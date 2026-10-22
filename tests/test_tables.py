"""LCM table tests.   python tests/test_tables.py           (fast)
                   python tests/test_tables.py --slow    (+ regenerates everything and compares)
They are also compatible with pytest."""
import json, os, subprocess, sys, tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lcm import load, position, cell, axis_of
from lcm.tense import shift
from lcm.walk import walk, path, WalkResult

T = load()
EXPECTED = {"subjects": 1391, "concepts": 7676, "connectors": 196}


def test_counts():
    for name, n in EXPECTED.items():
        assert len(T[name]) == n, (name, len(T[name]), n)


def test_weights_are_integers_0_to_10():
    for table in T:
        for n in table.nodes:
            for k in ("V", "I", "Lv", "Gv"):
                assert isinstance(n[k], int) and 0 <= n[k] <= 10, (table.name, n["word"], k, n[k])


def test_stored_geometry_matches_formulas():
    """Stored x, y, cell and tension == recomputed from the 4 weights (D1 and the cell rule)."""
    for table in T:
        for n in table.nodes:
            x, y = position(n["V"], n["I"], n["Lv"], n["Gv"])
            col, row = cell(x, y)
            assert (n["x"], n["y"], n["col"], n["row"], n["tension"]) == (x, y, col, row, min(n["V"], n["I"])), (table.name, n["word"])


def test_cells_are_inside_the_grid():
    for table in T:
        for (r, c) in table.occupied():
            assert 1 <= r <= 10 and 1 <= c <= 10


def test_drawing_anchors():
    """The words of the original drawing stay where they were drawn."""
    us = T["subjects"].get("us")[0]
    assert (us["row"], us["col"]) == (1, 5)                       # up, centered (We, Possibility)
    assert T["subjects"].get("you")[0]["col"] == 1                  # Adaptation extreme
    assert T["subjects"].get("i")[0]["col"] == 10                   # Imposition extreme
    assert T["subjects"].get("you")[0]["row"] == T["subjects"].get("i")[0]["row"]   # same height
    assert T["concepts"].get("dehumanization")[0]["row"] == 10   # down (World)
    dog = T["subjects"].get("dog")[0]
    assert dog["y"] > 0 and dog["x"] < 0                           # animals: up and toward Adaptation
    desert = T["concepts"].get("desert")[0]
    assert desert["y"] < 0 and desert["x"] > 0                     # desert: down and toward Imposition


def test_vertical_is_viability():
    """happiness/joy up; danger/sadness down."""
    c = T["concepts"]
    assert c.get("happy")[0]["y"] > 0 > c.get("sad")[0]["y"]
    assert c.get("hope")[0]["y"] > 0 > c.get("fear")[0]["y"]
    assert c.get("fire")[0]["row"] >= 9 and c.get("ghost")[0]["row"] >= 9


def test_adjective_axis_rules():
    c = T["concepts"]
    assert c.get("beautiful")[0]["x"] < 0     # judgment -> Adaptation (the hand-written list takes precedence over the generated one)
    assert c.get("loud")[0]["x"] > 0          # a single sense -> Imposition
    assert c.get("red")[0]["x"] > 0


def test_connectors_follow_the_time_axis():
    k = T["connectors"]
    us_cell = (T["subjects"].get("us")[0]["row"], T["subjects"].get("us")[0]["col"])
    assert (k.get("moreover")[0]["row"], k.get("moreover")[0]["col"]) == us_cell        # additive = we/us
    assert k.get("neither…nor")[0]["row"] == 10 and k.get("nor")[0]["row"] == 10       # negation = South
    assert axis_of(k.get("then")[0]["x"], k.get("then")[0]["y"]) == "North"             # relative future
    assert axis_of(k.get("before")[0]["x"], k.get("before")[0]["y"]) == "South"           # past
    for w in ("when", "while (temporal)", "and", "because", "although"):                   # present: band
        n = k.get(w)[0]
        assert abs(n["y"]) < 1.5, (w, n["y"])
    assert {n["construction"][0] for n in k.get("so that")} == {18}
    assert 21 in k.get("so (result)")[0]["construction"]


def test_ambiguous_connectors_are_two_nodes():
    k = T["connectors"]
    assert k.get("so (result)")[0]["row"] != k.get("so that")[0]["row"]
    assert len(k.get("though")) == 2 and k.get("though")[0]["current_axis"] != k.get("though")[1]["current_axis"]


def test_related_links_resolve_and_labels_are_separate():
    for table in T:
        for n in table.nodes:
            for target in n.get("related", []):
                assert T.find(target), (n["word"], target)        # every related entry is an existing node
            for lab in n.get("related_labels", []):
                assert not T.find(lab), (n["word"], lab)          # labels without a node are not nodes


def test_lookup_api():
    assert T.find("happy") and T.synonyms("happy")
    assert T["subjects"].near("dog", 1)
    assert T["subjects"].cell(1, 5)


def test_excel_matches_json():
    try:
        import openpyxl
    except ImportError:
        return
    wb = openpyxl.load_workbook(os.path.join(ROOT, "data", "LCM_tables.xlsx"), data_only=True)
    for name, sheet in (("subjects", "Subjects_nodes"), ("concepts", "Concepts_nodes"), ("connectors", "Connectors_nodes")):
        ws = wb[sheet]
        assert ws.max_row - 1 == len(T[name]), sheet
        for i, n in enumerate(T[name].nodes, start=2):
            assert [ws.cell(i, c).value for c in (1, 5, 6, 7, 8, 11, 12)] == [n["word"], n["V"], n["I"], n["Lv"], n["Gv"], n["col"], n["row"]], (sheet, i)


# ── v0.4: pronouns, split rule #10, related, opposites, tense, segmentation and walk ─────────────────────

def test_third_person_and_demonstratives():
    """Problem H: he/she/they/him/her/this/these are already nodes; 'them' is no longer duplicated in World."""
    s = T["subjects"]
    for w in ("he", "she", "they", "him", "her", "his", "their", "this", "these"):
        assert s.get(w), w
    for w in ("he", "she", "they"):
        n = s.get(w)[0]
        assert n["y"] > 0 and n["x"] < 0, w                      # third person: Context / Adaptation
    assert s.get("this")[0]["y"] < 0 < s.get("this")[0]["x"]     # near demonstrative: World / Imposition
    assert len(s.get("them")) == 1


def test_temporal_rule_is_split():
    """Decision #2: before/until -> South and lest -> South no longer 'differ'; no connector with a construction clashes."""
    k = T["connectors"]
    assert all(n["compat"] != "differs" for n in k.nodes)
    for w in ("before", "until", "lest"):
        n = k.get(w)[0]
        assert n["current_axis"] == "South" and n["compat"] == "matches", (w, n["current_axis"], n["compat"])
    assert k.get("when")[0]["current_axis"] == "West"             # simultaneous stays in the present band


def test_related_coverage_and_targets_exist():
    """Problem A: almost every node opens some link, and every target (WordNet, manual, opposite) is a node."""
    total = linked = 0
    for name in ("subjects", "concepts"):
        for n in T[name].nodes:
            total += 1
            linked += bool(n["related"] or n["related_wn"] or n["related_manual"])
            for target, via in n["related_wn"]:
                assert T.find(target) and via in T.LINK_WEIGHT, (n["word"], target, via)
            for target in n["related_manual"] + n["opposites"]:
                assert T.find(target), (n["word"], target)
    assert linked / total >= 0.95, (linked, total)
    for w in ("story", "jump", "dog", "happy"):                  # the 4 that had none in v0.3
        assert T.links(w), w


def test_links_are_ranked_and_deduplicated():
    ls = T.links("story")
    assert [round(w, 3) for *_, w in ls] == sorted((round(w, 3) for *_, w in ls), reverse=True)
    ids = [id(n) for n, *_ in ls]
    assert len(ids) == len(set(ids))
    assert ls[0][2] == "manual"                                  # the narrative seed weighs more than WordNet
    assert {n["word"] for _, n in T.related("story", sources={"manual"})} >= {"person", "hero", "tell"}


def test_story_person_jump_path_exists():
    """The guiding example: story -> person -> jump can be walked (each hop is manual)."""
    assert [w for w, _ in path(T, "story", "person", vias={"manual"})] == ["story", "person"]
    assert [w for w, _ in path(T, "person", "jump", vias={"manual"})] == ["person", "jump"]
    p = path(T, "story", "jump")
    assert p and p[0][0] == "story" and p[-1][0] == "jump" and len(p) <= 3
    assert path(T, "story", "zzzz-does-not-exist") is None


def test_opposites_and_mirror():
    """Decision #6: lexical opposite (WordNet) when there is one; geometric reflection for the rest."""
    assert "unhappy" in {n["word"] for _, n in T.opposites("happy")}
    assert sum(1 for n in T["concepts"].nodes if n["opposites"]) > 300
    happy = T["concepts"].get("happy")[0]
    for n in T.mirror("happy", radius=0):
        assert (n["row"], n["col"]) == (11 - happy["row"], 11 - happy["col"])


def test_tense_moves_weights_along_the_vertical():
    jump = T["concepts"].get("jump")[0]
    before = dict(jump)
    past_w, pre, fut = shift(jump, "past"), shift(jump, "present"), shift(jump, "future")
    assert past_w["y"] < pre["y"] < fut["y"]                        # past = South, future = North
    assert pre["x"] == jump["x"] and pre["y"] == jump["y"]       # the present moves nothing
    assert shift(jump, "present", negated=True)["y"] < pre["y"]  # negation also goes South
    assert (past_w["Lv"], past_w["Gv"]) == (jump["Lv"], jump["Gv"])    # tense does not touch the sensory/relational
    assert jump == before                                         # does not modify the node
    for t in ("past", "future"):
        for neg in (False, True):
            r = shift(jump, t, neg)
            assert all(0 <= r[k] <= 10 for k in ("V", "I", "Lv", "Gv"))
    try:
        shift(jump, "conditional")
        assert False
    except ValueError:
        pass


def _parse_or_skip():
    try:
        from lcm.parse import parse, _nlp
        _nlp()
        return parse
    except RuntimeError:
        return None                                              # without spaCy or without the model: skipped


def test_parse_guide_example():
    parse = _parse_or_skip()
    if parse is None:
        return
    f = parse("tell me a story", T)
    assert (f.action.lemma_text, f.recipient.node["word"], f.topic.lemma_text, f.quantity) == ("tell", "me", "story", 1)
    assert (f.mode, f.tense, f.negated) == ("imperative", "future", False)
    assert f.subject.lemma_text == "you" and not f.missing
    assert f.displaced_action["y"] > f.action.node["y"]         # imperative = future = North

    g = parse("Can you tell me a story about a brave king?", T)
    assert g.mode == "request" and g.tense == "future" and g.about_topic.lemma_text == "king" and g.about_qualities == ["brave"]

    p = parse("The dog jumped over the fence yesterday", T)
    assert p.tense == "past" and p.subject.lemma_text == "dog" and p.action.lemma_text == "jump"
    assert p.displaced_action["y"] < p.action.node["y"]         # past = South

    n = parse("I will not jump", T)
    assert n.tense == "future" and n.negated
    assert parse("Tell me two stories", T).quantity == 2


def test_walk_invariants():
    parse = _parse_or_skip()
    if parse is None:
        return
    f = parse("tell me a story", T)
    a, b = walk(T, f), walk(T, f)
    assert a.to_dict() == b.to_dict()                            # deterministic
    assert a.words[:3] == ["story", "tell", "me"]             # anchors come first (topic, action, recipient)
    assert len(set(a.words)) == len(a.words)               # no repeated nodes
    assert len(a.steps) - 5 <= 8 and 0.0 <= a.drift <= 1.0 and a.stop
    assert all(p.table != "connectors" for p in a.steps)
    assert all(p.from_word in a.words for p in a.steps if p.from_word)  # every node is opened from one already visited
    empty_value = walk(T, parse("Zzqx blorf", T))
    assert isinstance(empty_value, WalkResult) and "no anchors" in empty_value.stop or empty_value.steps


def test_walk_objective_reads_roles_and_theme_tense():
    """The walk's target uses the recipient's ROLE (Adaptation) and the past of 'story' (South).
    Consequence MEASURED in v0.4c: with that target (x = -3.2) the greedy walk no longer reaches 'jump'
    (Imposition, x = +3) even with 30 steps; that is why narrate.py chooses verbs through the protagonist's direct links."""
    parse = _parse_or_skip()
    if parse is None:
        return
    f = parse("tell me a story", T)
    r = walk(T, f, max_steps=14)
    assert -2.0 < r.target[0] < 0.0                            # Adaptation side, but balanced: I (+5) and you (-5) cancel out
    assert f.displaced_subject["x"] > 0 > f.displaced_recipient["x"]
    assert f.topic.node["y"] < 0                                  # story: South
    assert path(T, "story", "jump") is not None                  # the link path still exists


# ── v0.4b: from walk to TEXT (lcm/narrate.py) ─────────────────────────────────────────────────────────

def _tell(sentence, **kw):
    parse = _parse_or_skip()
    if parse is None:
        return None
    from lcm.narrate import narrate
    return narrate(T, parse(sentence, T), **kw)


def test_narrate_tell_me_a_story():
    """The goal: 'TELL ME A STORY' produces a story. The STRUCTURE is checked, not a fixed sentence."""
    r = _tell("TELL ME A STORY")
    if r is None:
        return
    assert r.error is None and 3 <= len(r.sentences) <= 5
    o = r.sentences
    assert o[0].text.startswith("There was a") and " who " in o[0].text
    assert all(x.text.endswith(".") and x.text[0].isupper() for x in o)
    verbs = [x.verb for x in o]
    assert len(set(verbs)) == len(verbs)                                   # does not repeat verbs
    assert all(T.find(v) and any(n.get("narrative_verb") for _, n in T.find(v)) for v in verbs)
    connectors = [x.connector for x in o[1:]]
    assert len(set(connectors)) == len(connectors)                           # does not repeat connectors
    assert all(T["connectors"].get(c) for c in connectors)                   # all come from the Connectors table
    ys = [T["concepts"].get(v)[0]["y"] if T["concepts"].get(v) else T["subjects"].get(v)[0]["y"] for v in verbs]
    assert ys[0] > ys[-1]                                                    # arc: starts up (North) and closes down (South)
    assert 0.0 < r.coherence <= 1.0
    assert r.text == _tell("TELL ME A STORY").text                       # deterministic


def test_narrate_pig_jumped_high():
    """MEASURED in v0.4b: with a pig as protagonist the project's example sentence comes out. If the seed or the
    weights change it may move; it is a finding of now, not a contract."""
    r = _tell("TELL ME A STORY ABOUT A PIG")
    if r is None:
        return
    assert r.sentences[0].text == "There was a pig who jumped high."
    assert r.sentences[-1].verb == "sleep"


def test_narrate_request_details():
    if _parse_or_skip() is None:
        return
    assert _tell("Tell me a story about a brave king").sentences[0].text.startswith("There was a brave king who")
    assert " he " in " " + _tell("Tell me a story about a king").sentences[1].text.lower() + " "   # king -> he
    assert _tell("Tell me a joke").error and "story" in _tell("Tell me a joke").error      # does not invent other constructions
    assert _tell("TELL ME A STORY", n_sentences=2).sentences[-1].text.split(",")[0] in ("Finally", "Lastly")
    from lcm.narrate import stories
    from lcm.parse import parse
    two_stories = stories(T, parse("Tell me two stories", T))
    assert len(two_stories) == 2 and two_stories[0].text != two_stories[1].text                    # 'two' -> two different stories
    assert len(stories(T, parse("Tell me a story", T))) == 1


def test_narrate_traza_explains_every_choice():
    r = _tell("TELL ME A STORY")
    if r is None:
        return
    for o in r.sentences:
        assert any(z.startswith("verb:") for z in o.trace)
    t0 = r.sentences[0].trace
    assert t0[0].startswith("tense:") and "South" in t0[0] and "There was" in t0[0]       # the tense is explained
    assert t0[1].startswith("protagonist:") and "story ->" in t0[1]



# ── v0.4c: topic tense (story = past = South) and recipient role (me = Adaptation) ───────────────────────

def test_story_family_is_past_in_the_data():
    """A story tells something that already happened: story/tale/legend/myth/memory fall in the South, like 'history'."""
    for w in ("story", "tale", "legend", "myth", "memory", "history"):
        n = T["concepts"].get(w)[0]
        assert n["y"] < 0 and n["V"] < n["I"], (w, n["y"])       # past = World (I) outweighs Context (V)
    assert T["concepts"].get("story")[0]["row"] > 5            # and it is no longer in the saturated cell (4, 3)
    # the data shift is the same one lcm/tense.py uses
    import re
    from lcm.tense import TENSE_SHIFT
    src = open(os.path.join(ROOT, "tools", "build_tables.py"), encoding="utf-8").read()
    m = re.search(r"PAST_SHIFT = \((-?\d+), (-?\d+), 0, 0\)", src)
    assert (int(m.group(1)), int(m.group(2))) == TENSE_SHIFT["past"]


def test_recipient_role_reads_me_as_adaptation():
    from lcm.tense import recipient_role, tense_of_position
    me = T["subjects"].get("me")[0]
    assert me["x"] > 0                                           # lexically 'me' is Imposition (like 'I')
    r = recipient_role(me)
    assert r["x"] < 0 and (r["Lv"], r["Gv"]) == (me["Gv"], me["Lv"]) and (r["V"], r["I"]) == (me["V"], me["I"])
    assert r["x"] == -me["x"] and r["y"] == me["y"]              # only the side changes
    assert (me["Lv"], me["Gv"]) == (10, 0)                       # does not modify the node
    assert [tense_of_position(y) for y in (-1.5, 0.0, 0.4, 2.0)] == ["past", "present", "present", "future"]
    parse = _parse_or_skip()
    if parse is None:
        return
    f = parse("TELL ME A STORY", T)
    assert f.displaced_recipient["x"] < 0
    g = parse("I will jump", T)                                  # 'I' as SUBJECT is untouched: it stays in Imposition
    assert g.subject.node["x"] > 0 and g.displaced_recipient is None


def test_narrate_tense_comes_from_the_theme_position():
    """'There was' is not fixed: it comes from the height of 'story'. With the same topic raised to the North or to the present
    band, the opening and the verbs change tense."""
    parse = _parse_or_skip()
    if parse is None:
        return
    import copy
    from lcm.narrate import narrate
    f = parse("TELL ME A STORY", T)
    base = narrate(T, f)
    assert base.tense == "past" and base.sentences[0].text.startswith("There was ")
    assert "South" in base.tense_reason and "future" in base.tense_reason        # 'tell' = future act, past content

    def with_y(y):
        g = copy.copy(f)
        g.topic = copy.copy(f.topic)
        g.topic.node = dict(f.topic.node, y=y)
        return narrate(T, g)
    fut, pres_w = with_y(+2.0), with_y(0.0)
    assert fut.tense == "future" and fut.sentences[0].text.startswith("There will be ") and " will " in fut.sentences[0].text
    assert pres_w.tense == "present" and pres_w.sentences[0].text.startswith("There is ")
    assert pres_w.sentences[0].text.split(" who ")[1].split(" ")[0].endswith("s")   # present: 'jumps', 'climbs'…


# ── v0.4d: randomizer for generic requests ("about an animal") ───────────────────────────────────────────

def test_sampling_module():
    from lcm.sampling import Sampler
    c = [("a", 5.0), ("b", 3.0), ("c", 1.0), ("d", 0.5)]
    # same seed = same results; different seeds = different sequences
    a1, a2 = Sampler(seed=7), Sampler(seed=7)
    assert [a1.choose(c)[0] for _ in range(30)] == [a2.choose(c)[0] for _ in range(30)]
    assert [Sampler(seed=1).choose(c)[0] for _ in range(1)] is not None
    # temperature 0 and top_k=1: always the highest score
    assert {Sampler(seed=i, temperature=0).choose(c)[0] for i in range(20)} == {"a"}
    assert {Sampler(seed=i, top_k=1).choose(c)[0] for i in range(20)} == {"a"}
    # the probabilities sum to 1, and temperature flattens or sharpens them
    def p_best(T):
        ps = Sampler(temperature=T).weights(c)
        assert abs(sum(p for _, p in ps) - 1.0) < 1e-9
        return dict(ps)["a"]
    assert p_best(0.5) > p_best(1.0) > p_best(3.0) > 0.25               # low T = peaked, high T = flat
    assert abs(p_best(1.0) - 5.0 / 9.5) < 1e-9                          # T = 1: proportional to the score
    # the observed frequency follows the probability (deterministic with a fixed seed)
    az, n = Sampler(seed=123), 4000
    counted = {k: 0 for k, _ in c}
    for _ in range(n):
        counted[az.choose(c)[0]] += 1
    assert abs(counted["a"] / n - 5.0 / 9.5) < 0.03 and counted["d"] > 0
    assert set(dict(Sampler(top_k=2).weights(c))) == {"a", "b"}              # top_k truncates
    assert dict(Sampler().weights([("x", 0.0), ("y", 0.0)])) == {"x": 0.5, "y": 0.5}   # all zero: uniform
    assert Sampler().seed != Sampler().seed or True and Sampler(seed=5).seed == 5
    for bad in (lambda: Sampler(temperature=-1), lambda: Sampler().choose([])):
        try:
            bad()
            assert False
        except ValueError:
            pass


def test_generic_nodes_and_their_species():
    """'animal' is generic (hundreds of descendants in WordNet); 'dog' is not. The candidates are concrete species."""
    from lcm.narrate import _descendants, GENERIC_MIN, EXCLUDE_FROM_DRAW
    animal = _descendants(T, T["subjects"].get("animal")[0])
    names = {n["word"].lower() for n in animal}
    assert len(animal) >= 30 and len(animal) >= GENERIC_MIN
    assert {"pig", "lion", "rabbit", "horse", "cat", "frog"} <= names
    assert all(n["category"] == "animal" and n["type"] == "biological" for n in animal)
    # there must be NO words whose main sense is not an animal, no plurals, no intermediates and no vulgar ones
    for bad in ("admiral", "amazon", "cardinal", "queen", "ass", "pussy", "cock", "mankind", "livestock", "beef"):
        assert bad not in names, bad
    assert not any(w.endswith("s") and w[:-1] in names for w in names)           # no plurals (snakes, ducks…)
    assert not ({"mammal", "bird", "fish", "insect"} & names)                       # no intermediates
    assert len(_descendants(T, T["subjects"].get("dog")[0])) < GENERIC_MIN          # dog: used as is
    assert len(_descendants(T, T["subjects"].get("pig")[0])) < GENERIC_MIN
    assert len(_descendants(T, T["subjects"].get("bird")[0])) >= GENERIC_MIN        # bird is also generic
    assert EXCLUDE_FROM_DRAW >= {"ass", "pussy", "cock"}


def test_narrate_random_for_generic_requests():
    parse = _parse_or_skip()
    if parse is None:
        return
    from lcm.narrate import narrate, stories
    f = parse("tell me a story about an animal", T)
    rs = [narrate(T, f, seed=i) for i in range(40)]
    assert all(r.error is None and 3 <= len(r.sentences) <= 5 for r in rs)
    protagonists = {r.sentences[0].protagonist for r in rs}
    assert len(protagonists) >= 10                                                   # really draws among many species
    for r in rs:
        t0 = r.sentences[0]
        assert t0.text.startswith(("There was a ", "There was an ")) and r.text.count(".") == len(r.sentences)
        assert T["subjects"].get(t0.protagonist)[0]["category"] == "animal"
        assert r.seed is not None and any("RANDOM" in z and str(r.seed) in z for z in t0.trace)
        assert len({o.verb for o in r.sentences}) == len(r.sentences)       # different verbs, also in drawn species
    # same seed = same story; the seed is reported so it can be repeated
    assert narrate(T, f, seed=42).text == narrate(T, f, seed=42).text
    assert narrate(T, f, seed=42).seed == 42
    # without a seed: one is generated and recorded
    free = narrate(T, f)
    assert isinstance(free.seed, int) and narrate(T, f, seed=free.seed).text == free.text
    # temperature 0: always the most probable species, whatever the seed
    assert len({narrate(T, f, seed=i, temperature=0).sentences[0].protagonist for i in range(15)}) == 1
    assert len({narrate(T, f, seed=i, top_k=3).sentences[0].protagonist for i in range(60)}) <= 3
    # 'two stories about an animal' -> two stories with derived seeds (seed, seed + 1)
    two_stories = stories(T, parse("tell me two stories about an animal", T), seed=10)
    assert [d.seed for d in two_stories] == [10, 11]
    assert two_stories[0].text == narrate(T, parse("tell me two stories about an animal", T), variant=0, seed=10).text


def test_no_random_when_request_is_concrete():
    """If something concrete is asked for (pig, dog, king) or nothing, there is NO draw: it is deterministic and the seed does not count."""
    parse = _parse_or_skip()
    if parse is None:
        return
    from lcm.narrate import narrate
    for sentence in ("tell me a story about a pig", "tell me a story about a dog", "tell me a story about a brave king", "tell me a story"):
        f = parse(sentence, T)
        a, b = narrate(T, f, seed=1), narrate(T, f, seed=999)
        assert a.text == b.text and a.seed is None, sentence
    # with randomize=True (random) it DOES draw (among the top 5) even if no protagonist is requested
    f = parse("tell me a story", T)
    random_stories = [narrate(T, f, seed=i, randomize=True) for i in range(30)]
    assert 2 <= len({r.sentences[0].protagonist for r in random_stories}) <= 5 and all(r.seed is not None for r in random_stories)


def test_person_sampling_is_safe():
    """The WordNet tree under 'person' brings slurs and nationalities; among people ONLY characters
    that the seed endorses are drawn. This test exists because without that restriction offensive and nationality terms came out."""
    parse = _parse_or_skip()
    if parse is None:
        return
    from lcm.narrate import narrate, GENERIC_WORDS
    f = parse("tell me a story about a person", T)
    protagonists = {narrate(T, f, seed=i).sentences[0].protagonist for i in range(150)}
    assert len(protagonists) >= 5
    banned = {"asshole", "bastard", "arab", "australian", "baptist", "amazon", "babe", "bitch", "whore", "slut", "friend"}
    assert not (protagonists & banned), protagonists & banned
    for w in protagonists:
        n = T["subjects"].get(w)[0]
        assert n["related_manual"] and w not in GENERIC_WORDS, w            # approved by the manual seed


def test_species_without_own_verbs_inherit_and_arc_rescales():
    """'buffalo' or 'deer' have no verbs in the seed: they inherit those of their ancestor (animal). And if there are few verbs,
    the arc is rescaled: the last sentence is still the lowest (before, it was cut and the closure was lost)."""
    parse = _parse_or_skip()
    if parse is None:
        return
    from lcm.narrate import narrate, _protagonist_verbs
    for w in ("buffalo", "deer", "owl"):
        n = T["subjects"].get(w)[0]
        assert not n["related_manual"]
        vs, de = _protagonist_verbs(T, n)
        assert vs and de in ("animal", "bird", "insect", "fish"), (w, de)
        assert all(v["narrative_verb"] for v, _ in vs)
    f = parse("tell me a story about an animal", T)
    seen = set()
    for i in range(60):
        r = narrate(T, f, n_sentences=6, seed=i)
        seen.add(len(r.sentences))
        assert r.sentences[-1].y_arc == -2.0 and r.sentences[0].y_arc == 2.0     # the arc always closes at the bottom
    assert min(seen) < 6                                                            # there were species with few verbs



# ── v0.4e: perspective (you -> I), mode (? = North, ! = South) and opinion answers ───────────────────────

def test_deixis_inverts_first_and_second_person():
    from lcm.deixis import invert, is_deictic
    from lcm.tense import recipient_role
    assert [invert(T, w)["word"] for w in ("you", "i", "me", "my", "your", "myself")] == ["i", "you", "you", "your", "my", "yourself"]
    assert invert(T, "duck") is None and not is_deictic("duck")
    assert invert(T, "you")["x"] > 0 > invert(T, "me")["x"]                  # you -> I (Imposition); me -> you (Adaptation)
    # what in v0.4c was an unnamed Lv<->Gv swap IS this inversion: 'me' with Lv<->Gv gives the weights of 'you'
    me, you = T["subjects"].get("me")[0], T["subjects"].get("you")[0]
    r = recipient_role(me)
    assert (r["V"], r["I"], r["Lv"], r["Gv"]) == (you["V"], you["I"], you["Lv"], you["Gv"])


def test_mode_shift_question_north_exclamation_south():
    n = T["concepts"].get("think")[0]
    base = shift(n, "present")["y"]
    assert shift(n, "present", form="question")["y"] > base > shift(n, "present", form="exclamation")["y"]
    assert shift(n, "future", form="question")["y"] > shift(n, "future")["y"]   # it combines with tense
    assert shift(n, "present", form=None) == shift(n, "present")
    try:
        shift(n, "present", form="susurro")
        assert False
    except ValueError:
        pass


def test_parse_questions_and_exclamations():
    parse = _parse_or_skip()
    if parse is None:
        return
    q = parse("WHAT DO YOU THINK ABOUT DUCKS?", T)
    assert (q.form, q.unknown, q.action.lemma_text, q.mode) == ("question", "what", "think", "interrogative")
    assert q.displaced_subject["word"] == "i" and q.displaced_subject["x"] > 0           # the user's 'you' is 'I' for the LCM
    assert q.about_topic.node["type"] == "biological" and q.about_topic.plural                          # 'ducks' = the animal, not 'to duck'
    assert q.displaced_action["y"] > shift(q.action.node, "present")["y"]                  # question: the verb goes up to the North
    assert parse("what do you think about ducks?", T).about_topic.node["word"] == q.about_topic.node["word"]   # uppercase = lowercase
    a = parse("What do you think about a duck?", T).about_topic
    assert a.node["type"] == "biological" and not a.plural and a.det == "a"                   # singular: uses the plural's node
    assert parse("What do you think about war?", T).about_topic.det is None                        # without an article none is invented
    s = parse("What is a story?", T)
    assert s.topic.lemma_text == "story" and s.unknown == "what" and s.subject is None
    e = parse("Ducks are great!", T)
    assert (e.mode, e.form) == ("exclamative", "exclamation")
    t1, t2 = parse("Tell me a story!", T), parse("Tell me a story", T)
    assert t1.mode == "imperative" and t1.form == "exclamation" and t2.form is None
    assert t1.displaced_action["y"] < t2.displaced_action["y"]                              # exclamation: the verb goes down to the South
    d = parse("tell me a story", T)                                                           # 'tell me': you -> I, me -> you
    assert d.displaced_subject["word"] == "i" and d.displaced_recipient["word"] == "you"


def _reply(sentence, **kw):
    parse = _parse_or_skip()
    if parse is None:
        return None
    from lcm.reply import reply
    return reply(T, parse(sentence, T), **kw)[0]


def test_responder_opinion_about_ducks():
    r = _reply("WHAT DO YOU THINK ABOUT DUCKS?", seed=3)
    if r is None:
        return
    pos = {n["word"] for n in T["concepts"].nodes if n.get("opinion") == "pos"}
    assert r.error is None and r.kind == "opinion" and r.stance == "pos" and r.seed == 3
    assert r.phrases[0].text.startswith("I think ducks are ") and r.phrases[0].text.rstrip(".").split()[-1] in pos
    assert r.phrases[1].text.startswith("They are birds and can ") and r.text.count(".") == 2
    z = " | ".join(r.trace)
    assert "perspective: 'you'" in z and "'i'" in z and "mode:" in z and "question" in z and "DRAW" in z and "seed=3" in z
    assert 0.0 < r.coherence <= 1.0
    # same seed = same answer; different seeds = different adjectives; T=0 = always the same
    assert _reply("What do you think about ducks?", seed=3).text == r.text
    adjs = {_reply("What do you think about ducks?", seed=i).phrases[0].text.rstrip(".").split()[-1] for i in range(40)}
    assert len(adjs) >= 5 and adjs <= pos
    assert len({_reply("What do you think about ducks?", seed=i, temperature=0).phrases[0].text for i in range(10)}) == 1


def test_responder_like_questions_and_determiners():
    if _reply("Do you like ducks?", seed=1) is None:
        return
    assert _reply("Do you like ducks?", seed=1).text.startswith("Yes, I like ducks. I think they are ")
    assert _reply("Do you love ducks?", seed=1).text.startswith("Yes, I love ducks.")
    assert _reply("Do you hate dogs?", seed=1).text.startswith("No, I don't hate dogs. I think they are ")   # pos + negative verb
    assert _reply("Do you like a duck?", seed=1).text.startswith("Yes, I like a duck. I think it is ")
    assert _reply("What do you think about the duck?", seed=1).text.startswith("I think the duck is ")
    assert _reply("What do you think about a duck?", seed=1).phrases[1].text.startswith("It is a bird and can ")


def test_responder_disposition_is_a_rule_not_data():
    """SentiWordNet gives 0.0 to war/poison and positive to disease: the 'amable' disposition only applies to animals. Without this
    test 'I think a war is friendly' came out."""
    if _reply("What do you think about war?") is None:
        return
    for w in ("war", "poison", "disease", "stories"):
        r = _reply(f"What do you think about {w}?", seed=2)
        assert r.stance == "neutral" and "I don't have a strong opinion about" in r.phrases[0].text and r.seed is None, w
    from lcm.reply import LIVING_ONLY
    neg = {n["word"] for n in T["concepts"].nodes if n.get("opinion") == "neg"}
    for seed_i in range(20):
        p = _reply("What do you think about pain?", seed=seed_i)                   # net sentiment -0.58: it does give an opinion, a bad one
        assert p.stance == "neg" and p.phrases[0].text.rstrip(".").split()[-1] in neg - LIVING_ONLY
    n = _reply("What do you think about ducks?", disposition="neutral")            # |-0.04| < 0.25: no stance
    assert n.phrases[0].text == "I don't have a strong opinion about ducks." and n.seed is None
    assert n.phrases[1].text.startswith("They are birds and can ")                    # what it knows it still says
    d = _reply("What do you think about danger?", seed=1)
    assert " can " not in d.text                                                       # 'danger can run' does not: abilities only for living beings


def test_responder_never_opines_about_people():
    """A randomly generated judgment about groups of people, nationalities or the asker is not acceptable."""
    if _reply("What do you think about people?") is None:
        return
    from lcm.reply import NO_OPINION
    for sentence in ("What do you think about people?", "What do you think about women?", "What do you think about children?",
                  "What do you think about me?", "Do you like me?", "Do you hate him?",
                  "What do you think about the king?", "What do you think about family?", "What do you think about teachers?"):
        r = _reply(sentence, seed=1)
        assert r.error is None and r.text == NO_OPINION and r.stance == "neutral" and r.seed is None, sentence


def test_responder_limits_and_dispatch():
    if _reply("What is the capital of Peru?") is None:
        return
    assert "OPINION" in _reply("What is the capital of Peru?").error                      # does not invent other constructions
    assert "I do not know" in _reply("What do you think about zzqx?").error
    h = _reply("TELL ME A STORY")
    assert h.kind == "story" and h.text.startswith("There was ") and h.error is None       # the same entry point tells stories
    assert _reply("Ducks are great!").error                                                # exclamations are not answered yet
    from lcm.reply import reply
    from lcm.parse import parse
    assert len(reply(T, parse("tell me two stories about an animal", T), seed=4)) == 2
    try:
        _reply("What do you think about ducks?", disposition="furiosa")
        assert False
    except ValueError:
        pass


# ── v0.4g: decomposer (roles by value, no fixed words) ───────────────────────────────────────────────────

def test_decomposer_tell_me_a_story():
    parse = _parse_or_skip()
    if parse is None:
        return
    from lcm.decompose import decompose
    fs = {f.role_obj: f for f in decompose(T, parse("TELL ME A STORY", T))}
    assert set(fs) == {"action", "agent", "recipient", "topic"}
    a, ag, de, te = fs["action"], fs["agent"], fs["recipient"], fs["topic"]
    assert a.axis == "North" and a.tense_by_height == "future" and a.tense_requested.startswith("future")   # asks for the act: North
    assert a.forms["past"] == "told" and {"told", "tells", "telling"} <= set(a.inflected)         # said/told come from the tense rule
    assert "say" in a.synonyms and "say" in a.related["manual"]
    assert te.axis == "South" and te.tense_by_height == "past" and "tale" in te.synonyms                # story = past = South
    assert "legend" in te.related["kind-of"] and any("quantity: 1" in n for n in te.notes)
    assert de.perspective == "you" and de.perspective_side.startswith("Adaptation")                    # me -> you (Adaptation)
    assert ag.perspective == "i" and ag.perspective_side.startswith("Imposition")                      # implied you -> I
    assert "implicit" in ag.word
    assert de.opposites == ["you"] and not te.opposites and not a.opposites                                # no lexical opposite: none is invented
    assert all(f.to_dict() for f in fs.values())


def test_decomposer_question_and_unknown_word():
    parse = _parse_or_skip()
    if parse is None:
        return
    from lcm.decompose import decompose
    fs = {f.role_obj: f for f in decompose(T, parse("WHAT DO YOU THINK ABOUT DUCKS?", T))}
    assert fs["agent"].perspective == "i" and "you" in fs["agent"].word.lower()
    assert "question" in fs["action"].tense_requested                                                      # the question form is recognized
    assert fs["about_topic"].node["type"] == "biological"
    z = {f.role_obj: f for f in decompose(T, parse("Tell me about zzqx", T))}
    assert z["about_topic"].node is None and "not a node" in z["about_topic"].notes[0]                         # the unknown is stated, not invented


# ── v0.4h: recomposer (rules by value, no words inside sentences) ────────────────────────────────────────

def _recomp(sentence, **kw):
    parse = _parse_or_skip()
    if parse is None:
        return None
    from lcm.recompose import recompose
    return recompose(T, parse(sentence, T), **kw)


def test_recomposer_every_word_comes_from_a_node_a_functional_or_conjugation():
    """The difference from a template: no word is written inside a sentence."""
    if _recomp("TELL ME A STORY", seed=1) is None:
        return
    from lcm.function_words import FUNCTION_NODES
    from lemminflect import getLemma
    for sentence in ("TELL ME A STORY", "tell me a story about an animal", "tell me a story about a brave king"):
        for seed_i in range(8):
            r = _recomp(sentence, seed=seed_i)
            assert r.error is None
            for p in r.pieces:
                w = p.text.lower()
                if w in {".", ","}:
                    assert T["connectors"].get(w) or w == ",", w
                    continue
                is_node = bool(T.find(w))
                es_func = w in FUNCTION_NODES or (w == "an" and "a" in FUNCTION_NODES)
                is_conj = any(T.find(getLemma(w, u)[0]) for u in ("VERB", "AUX") if getLemma(w, u)) or w.startswith("will ")
                assert is_node or es_func or is_conj, (sentence, w, p.rule)
                assert p.rule                                                 # and every word declares which rule it comes from


def test_recomposer_tense_closure_and_descent_come_from_values():
    r = _recomp("TELL ME A STORY ABOUT A PIG", seed=1)
    if r is None:
        return
    assert r.error is None and r.tense == "past" and r.text.startswith("There was a pig who ")           # R1 + R3
    assert 2 <= len(r.clauses) <= 6 and all(c[-1].text == "." for c in r.clauses)
    verbs = [next(p for p in c if p.node and p.node.get("type") != "pronoun" and "R4" in p.rule) for c in r.clauses]
    ys = [v.node["y"] for v in verbs]
    assert all(b <= a - 0.25 for a, b in zip(ys[1:], ys[2:])) and ys[0] > ys[-1]                                  # R4: descends
    assert len({v.text for v in verbs}) == len(verbs)
    assert ys[-1] <= -1.0 and "R5" in r.closed_by and all(y > -1.0 for y in ys[:-1])                               # R5: closes on reaching the South, not before
    assert r.pieces[-1].text == "." and r.pieces[-1].node["y"] < 0                                                 # the period is a South node
    assert 0.0 < r.coherence <= 1.0
    assert _recomp("TELL ME A STORY ABOUT A PIG", seed=1).text == r.text                                     # deterministic with a seed


def test_recomposer_tense_follows_the_theme_height():
    parse = _parse_or_skip()
    if parse is None:
        return
    import copy
    from lcm.recompose import recompose
    f = parse("TELL ME A STORY ABOUT A PIG", T)

    def with_y(y):
        g = copy.copy(f)
        g.topic = copy.copy(f.topic)
        g.topic.node = dict(f.topic.node, y=y)
        return recompose(T, g, seed=1)
    assert with_y(-1.5).text.startswith("There was a pig who ")
    fut, pres_w = with_y(+2.0), with_y(0.0)
    assert fut.text.startswith("There will be a pig who will ") and pres_w.text.startswith("There is a pig who ")


def test_recomposer_connectors_by_value_and_commas_from_the_corpus():
    r = _recomp("TELL ME A STORY ABOUT A PIG", seed=1)
    if r is None:
        return
    import lcm.recompose as R
    cons = [c[0] for c in r.clauses[1:]]
    assert all(c.node["relation"] in (R.MIDDLE_RELATION, R.CLOSING_RELATION) for c in cons)
    assert cons[-1].node["relation"] == R.CLOSING_RELATION and all(c.node["relation"] == R.MIDDLE_RELATION for c in cons[:-1])  # closing only in the one that closes
    assert len({c.text for c in cons}) == len(cons)                                                                       # no repetition
    # R7: the comma comes from the corpus. With the real data then/finally take NO comma; if we force the probability, it appears
    assert all(p.text != "," for p in r.pieces)
    before = dict(R._punctuation())
    try:
        R._PUNCTUATION_DATA = {k: dict(v, comma_after=0.9) for k, v in before.items()}
        with_conn = _recomp("TELL ME A STORY ABOUT A PIG", seed=1)
        assert any(p.text == "," for p in with_conn.pieces) and " ," not in with_conn.text
    finally:
        R._PUNCTUATION_DATA = before
    d = R._punctuation()
    assert d["however"]["comma_after"] >= 0.8 and d["then"]["comma_after"] < 0.5 and d["finally"]["comma_after"] < 0.5


def test_functional_nodes_are_chosen_by_value():
    from lcm.function_words import node, determiner, FUNCTION_NODES
    assert node("a")["y"] > node("the")["y"] and node("there")["y"] < 0                # new = possibility; exist = South
    assert determiner(True, "owl")[0] == "an" and determiner(True, "pig")[0] == "a" and determiner(False, "pig")[0] == "the"
    assert node("zzz") is None and all(len(v) == 5 for v in FUNCTION_NODES.values())     # every function word declares its reason


def test_recomposer_limits():
    parse = _parse_or_skip()
    if parse is None:
        return
    assert "communication" in _recomp("Tell me about ducks").error                       # only recomposes communication topics
    assert "there is no topic" in _recomp("Tell me about zzqx").error or "there is no" in _recomp("Tell me about zzqx").error
    r = _recomp("tell me a story about an animal", seed=3)
    assert r.seed == 3 and _recomp("tell me a story about an animal", seed=3).text == r.text


# ── v0.4i: questions about the LCM itself (you = I = Imposition) ─────────────────────────────────────────

def test_self_reference_parse():
    parse = _parse_or_skip()
    if parse is None:
        return
    h = parse("HOW ARE YOU?", T)
    assert h.unknown == "how" and h.form == "question" and h.displaced_subject["word"] == "i" and h.displaced_subject["x"] > 0
    w = parse("Who are you?", T)                                       # spaCy leaves 'you' as attribute: the inversion applies there too
    assert w.unknown == "who" and w.displaced_subject["word"] == "i"
    a = parse("Are you happy?", T)
    assert a.quality.node["word"] == "happy" and a.displaced_subject["word"] == "i"
    assert parse("tell me a story", T).unknown is None                # no unknown is invented


def test_responder_state_questions_about_the_lcm():
    if _reply("HOW ARE YOU?", seed=1) is None:
        return
    pos = {n["word"] for n in T["concepts"].nodes if n.get("state") == "pos"}
    r = _reply("HOW ARE YOU?", seed=1)
    adj = r.text.rstrip(".").split()[-1]
    assert r.kind == "state" and r.text.startswith("I am ") and adj in pos and r.seed == 1 and r.stance == "pos"
    z = " | ".join(r.trace)
    assert "perspective" in z and "'i'" in z and "first person" in z and "It is not a real state" in z and "DRAW" in z
    assert _reply("How are you doing?", seed=1).text == r.text                    # same words, another way of asking
    assert _reply("How do you feel?", seed=1).text == "I feel " + adj + "."        # repeats the verb that was asked for
    assert _reply("HOW ARE YOU?", seed=1).text == r.text                           # deterministic with a seed
    assert len({_reply("How are you?", seed=i).text for i in range(30)}) >= 4       # the draw varies
    assert all(_reply("How are you?", seed=i).text.rstrip(".").split()[-1] in pos for i in range(30))
    assert _reply("Are you happy?").text == "Yes, I am happy."
    assert _reply("Are you sad?").text == "No, I am not sad."
    assert _reply("Are you hungry?").text == "I don't know if I am hungry."           # no sign in the tables: neither affirms nor denies
    assert _reply("Were you happy?").text == "Yes, I was happy."                       # the tense of the question is respected
    assert _reply("How are you?", disposition="neutral").text == "I don't have a state to report."
    for q in ("Who are you?", "What are you?"):
        e = _reply(q)
        assert e.error and "I have no description of myself" in e.error                # does not invent an identity
    assert _reply("Where are you?").error
    # a 'why' presupposes the state: it is NOT a yes/no question ("Why are you sad?" must not answer "No, I am not sad.")
    w = _reply("Why are you sad?")
    assert w.error and "why" in w.error and "No, I" not in (w.text or "")
    assert "yourself" not in (_reply("What do you think about yourself?").text or "")
    assert "I have no description of myself" in _reply("What do you think about yourself?").error
    for q in ("Hello", "Thank you", "I am sad"):                          # not questions: it says so, and the message is the right one
        e = _reply(q).error
        assert e and "statement" in e and "OPINION" not in e, q
    from lcm.parse import parse
    from lcm.narrate import narrate
    from lcm.recompose import recompose
    f = parse("Tell me a story about you", T)
    assert "pronoun" in narrate(T, f).error and "pronoun" in recompose(T, f).error      # is not silently ignored
    assert _reply("What do you think about ducks?", seed=3).text.startswith("I think ducks are ")   # the above stays the same


# ── v0.4j: AI-generated corpus (prompt generator and validator) ──────────────────────────────────────────

def _tools():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import prepare_ai_batch as P
    return P


def test_ia_prompt_builder_quotas_and_determinism():
    P = _tools()
    assert P.quotas(40) == {"past": 8, "present": 8, "future": 6, "negative": 4, "question": 5, "exclamation": 3, "imperative": 3, "connector": 3}
    for n in (10, 20, 40, 60, 80, 100, 37):
        assert sum(P.quotas(n).values()) == n and all(v >= 0 for v in P.quotas(n).values())
    parts = P.split_quotas(P.quotas(40), 4)
    assert [sum(p.values()) for p in parts] == [10, 10, 10, 10]
    assert {e: sum(p.get(e, 0) for p in parts) for e in P.quotas(40)} == P.quotas(40)           # nothing is lost when splitting
    rows = [{"rank": 1, "word": "duck", "class": "noun", "pos": "NOUN", "note": ""},
             {"rank": 2, "word": "the", "class": "determiner", "pos": "DET", "note": ""}]
    a, b = P.build(rows), P.build(rows)
    assert a == b                                                                                 # deterministic
    ids = [c["id"] for c in a]
    assert len(ids) == len(set(ids))
    fr = {l: [c for c in a if c["lemma"] == l and c["type"] == "phrases"] for l in ("duck", "the")}
    assert (len(fr["duck"]), sum(c["n"] for c in fr["duck"])) == (4, 40)                            # content: 40 sentences in 4 calls of 10
    assert (len(fr["the"]), sum(c["n"] for c in fr["the"])) == (6, 60)                              # function: 60 sentences in 6 calls
    assert sum(1 for c in a if c["type"] == "card") == 2                                           # one card per word
    lib = [c for c in a if c["type"] == "free_calls"]
    assert [(c["lemma"], c["n"]) for c in lib] == [("duck", 10), ("the", 10)]                       # and 10 sentences WITHOUT per-word quotas (natural use)
    assert all("Do NOT force any tense" in c["prompt"] and '"free"' in c["prompt"] and "REQUIRED MIX" not in c["prompt"] for c in lib)
    assert not any(c["type"] == "free_calls" for c in P.build(rows, free_calls=0))                     # and they can be removed
    doms = [d for c in fr["duck"] + [x for x in lib if x["lemma"] == "duck"] for d in c["domains"]]
    assert len(doms) == len(set(doms)) == 10                                                        # 10 different contexts (4 calls with quotas + 1 free)
    for c in fr["duck"]:
        assert '"duck"' in c["prompt"] and "noun" in c["prompt"] and sum(c["quotas"].values()) == c["n"]
        assert all(f"- {e}: {v}" in c["prompt"] for e, v in c["quotas"].items())
    assert "function word" in fr["the"][0]["prompt"] and "determiner" in fr["the"][0]["prompt"]
    assert {"russian", "american", "christian", "catholic"} <= P.SKIP                              # nationalities and religions are not generated
    sheet = P.read_sheet(os.path.join(ROOT, "data", "vocabulary_sheet.xlsx"))
    assert len(sheet) == 1330 - 11 and not ({f["word"] for f in sheet} & P.SKIP)


def _validator():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import validate_documents as V
    try:
        V.nlp()
    except Exception:
        return None
    return V


def test_ia_validator_rejects_each_kind_of_failure():
    V = _validator()
    if V is None:
        return
    import json as _j
    call = {"id": "duck|NOUN|phrases|1", "type": "phrases", "lemma": "duck", "pos": "NOUN", "class": "noun", "n": 10}
    S = [("free", "A duck is waddling slowly along the muddy path again."),                  # free: good even if there is no quota
         ("present", "The duck swims across the pond every morning."),                   # good
         ("past", "She had to duck quickly under the low branch."),                      # different_pos (different part of speech; it is a verb)
         ("past", "John saw a duck near the river."),                                    # entity (rejected: names a person, place or group)
         ("present", "That damn duck ran off with my bread."),                           # banned_word (banned word)
         ("present", "The bird swam across the pond slowly."),                           # missing_word (word missing)
         ("question", "Where does the duck sleep at night?"),                            # good
         ("question", "The duck sleeps by the old tree."),                               # inconsistent_category (inconsistent category)
         ("present", "Duck!"),                                                           # longitud (length)
         ("past", "I saw 3 ducks by the lake yesterday."),                               # digitos (digits)
         ("present", "The duck swims across the pond every morning."),                   # duplicada (duplicate)
         ("future", "A duck will swim across the pond every morning soon."),             # near_duplicate (near duplicate)
         ("whisper", "The duck sleeps quietly near the pond today.")]                    # invalid_category (invalid category)
    raw = _j.dumps({"lemma": "duck", "pos": "noun", "sentences": [{"cat": c, "text": t} for c, t in S]})
    a, r, f, inf = V.validate_batch([call], {call["id"]: "```json\n" + raw + "\n```"})          # tolerates the code block
    assert [x["text"] for x in a] == ["A duck is waddling slowly along the muddy path again.", "The duck swims across the pond every morning.",
                                      "Where does the duck sleep at night?"]
    assert [(x["reason"]) for x in r] == ["different_pos", "entity", "banned_word", "missing_word", "inconsistent_category", "length",
                                         "digits", "duplicate", "near_duplicate", "invalid_category"]           # each bad sentence, with ITS reason, in order
    assert inf["accepted"] == 3 and inf["rejected"] == 10 and inf["pct_different_pos"] == round(100 / 13, 1)
    assert inf["rejections_by_reason"]["different_pos"] == 1 and inf["words_below_80pct"] == ["duck|NOUN: 3/10"]
    # broken json, different lemma, empty list (the model declares it is unsure) and missing answer are counted, they do not break
    a, r, f, inf = V.validate_batch([call], {call["id"]: "this is not json"})
    assert inf["invalid_json"] == 1 and a == []
    a, r, f, inf = V.validate_batch([call], {call["id"]: _j.dumps({"lemma": "cat", "sentences": [{"cat": "past", "text": "The cat slept all day long."}]})})
    assert [x["reason"] for x in r] == ["different_lemma"]
    a, r, f, inf = V.validate_batch([call], {call["id"]: _j.dumps({"lemma": "duck", "pos": "noun", "sentences": []})})
    assert inf["calls_with_empty_list"] == 1
    assert V.validate_batch([call], {})[3]["no_reply"] == 1


def test_ia_validator_cleans_dictionary_entries():
    V = _validator()
    if V is None:
        return
    card = {"id": "duck|NOUN|card|1", "type": "card", "lemma": "duck", "pos": "NOUN", "class": "noun"}
    ok, why = V.validate_card(card, {"lemma": "duck", "definition": "A water bird with a flat beak and webbed feet.",
                                      "synonyms": ["drake", "waterfowl", "mallard", "duck", "a bird", "drake"], "antonyms": []}, set())
    assert ok["synonyms"] == ["drake", "waterfowl", "mallard"] and ok["antonyms"] == [] and ok["definition"].startswith("A water bird")
    assert {"synonym_repeated", "synonym_is_not_a_word"} <= set(why)                         # the bad is removed and counted
    none, why = V.validate_card(card, {"lemma": "duck", "definition": "A duck is a bird.", "synonyms": [], "antonyms": []}, set())
    assert none is None and "invalid_definition" in why                                            # the definition cannot use the word
    ok, _ = V.validate_card({**card, "lemma": "hot"}, {"lemma": "hot", "definition": "Having a high temperature.", "synonyms": ["warm"], "antonyms": ["cold", "damn"]}, {"damn"})
    assert ok["antonyms"] == ["cold"]                                                               # antonyms (negative pins) go through the banned-words filter
    assert V.validate_card(card, {"lemma": "cat", "definition": "x"}, set())[0] is None
    assert V.parse_json("text before {\"a\": 1} text after") == {"a": 1} and V.parse_json("[1, 2]") is None


# ── v0.4k: fixes after trying the system with real inputs ────────────────────────────────────────────────

def test_k_negated_story_request_is_not_told():
    parse = _parse_or_skip()
    if parse is None:
        return
    from lcm.reply import reply
    r = reply(T, parse("Don't tell me a story", T), seed=1)[0]
    assert r.error and "negated" in r.error                       # "Don't tell me a story" no longer tells a story


def test_k_time_noun_is_not_the_theme():
    parse = _parse_or_skip()
    if parse is None:
        return
    f = parse("I am happy today", T)
    assert f.topic is None or f.topic.lemma_text != "today"              # a time noun is not the topic


def test_k_subject_hanging_from_attribute():
    parse = _parse_or_skip()
    if parse is None:
        return
    f = parse("Why is the sky blue?", T)
    assert f.subject is not None and f.subject.lemma_text == "sky"       # spaCy attaches "sky" to "blue"


def test_k_lets_and_let_me():
    parse = _parse_or_skip()
    if parse is None:
        return
    f = parse("Let's go to the park", T)
    assert f.action.lemma_text == "go" and f.tense == "future"
    g = parse("Let me tell you a secret", T)
    assert g.action.lemma_text == "tell" and g.topic.lemma_text == "secret"


def test_k_quality_of_theme_moves_the_objective():
    parse = _parse_or_skip()
    if parse is None:
        return
    from lcm.walk import target_of
    a = target_of(parse("Tell me a story about a king", T))
    b = target_of(parse("Tell me a sad story about a king", T))
    assert b[1] < a[1]                                           # "sad" lowers the target's y


def test_k_corpus_links_are_opt_in():
    from lcm.tables import Tables
    assert Tables.USE_CORPUS is False                           # related_oanc does not enter links() unless deliberately enabled



def test_rebuild_is_identical():
    """Regenerates everything from source_data/ and compares with data/LCM_tables.json (slow: ~1 min, needs nltk)."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_tables.py"), os.path.join(ROOT, "source_data"), tmp, "--write"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        new = json.load(open(os.path.join(tmp, "LCM_tables.json"), encoding="utf-8"))
        old = json.load(open(os.path.join(ROOT, "data", "LCM_tables.json"), encoding="utf-8"))
        for t in old["tables"]:
            assert new["tables"][t]["nodes"] == old["tables"][t]["nodes"], t


if __name__ == "__main__":
    slow = "--slow" in sys.argv
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and (slow or k != "test_rebuild_is_identical")]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"  OK    {name}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests pass" + ("" if slow else "  (use --slow for the regeneration test)"))
    sys.exit(1 if failed else 0)
