"""Realization: from a walk of nodes to TEXT.   \"TELL ME A STORY\"  ->  \"There was a pig who jumped high. …\"

What it does and what it does NOT do (read before trusting the result):

  * It is a NARRATIVE construction with blanks (a template), not a language model. The grammar (word order,
    \"there was a … who …\", the past) is written here. What the tables CHOOSE are the blanks: the protagonist,
    each verb, each adverb and each connector.
  * Choosing = walking links and measuring with the geometry (the same GDE as `lcm.walk`):
        protagonist   story -> (animal | child | …) -> species   ordered by link weight x closeness to the target
        verb          actions without an object linked to the protagonist / to the previous verb
        adverb        adverb of manner linked to the verb
        connector     from the Connectors table, the closest to the height (y) of the sentence
  * TENSE: it is not written in the template. It comes from the height of the TOPIC: story falls in the South (y < 0) = past, so
    the story opens with \"There was\" and its verbs go in the past. \"tell\" asks for the ACT (imperative = future, North),
    but what is told takes the tense of the topic. A topic in the North would give \"There will be…\", in the band \"There is…\".
  * ARC: the story goes down the vertical axis. The 1st sentence looks for the highest action (Context: possibility),
    the last the lowest (World: closure). It is the same idea as the connectors criterion (start = North,
    closure = South) applied to the story: then/next are at y = +2.5 and finally at y = −2.0.
  * The \"who can do what\" and \"how it is done\" links come from the manual seed (`tools/related_manual.py`);
    WordNet does not have them. Without that seed the protagonist would not know how to do anything.
  * RANDOMNESS: if the requested protagonist is GENERIC (\"about an animal\": hundreds of descendants in the WordNet tree),
    the tables say who the candidates are and with what score, and `lcm.sampling` DRAWS one (seed, temperature,
    top-k, like the knobs of an LLM but just a draw). A candidate counts only if the MAIN sense of its
    word is inside the tree: so 'animal' does not draw 'admiral' (butterfly), 'amazon' (parrot) or 'queen' (bee).
    A drawn species without verbs of its own INHERITS those of the closest ancestor that has them (zebra -> animal).
  * Only the \"story\" construction exists. Any other topic returns `error`, not an invented text.
  * English only. It needs nltk (WordNet) and lemminflect, in addition to spaCy for `parse`.
"""
import math
from dataclasses import dataclass, field
from functools import lru_cache

from collections import Counter

from .sampling import Sampler
from .tense import shift, tense_of_position
from .walk import walk, target_of, R_MAX

STORY_WORDS = {"story", "tale", "fable", "narrative", "yarn"}
GENERIC_WORDS = {"person", "character", "people", "animal", "being", "human", "family", "friend", "group"}   # not concrete protagonists
PROTAGONIST_CATEGORIES = {"animal", "person"}
PROTAGONIST_VIAS = {"manual", "kind-of"}
MALE_WORDS = {"boy", "man", "king", "father", "brother", "prince", "hero", "son"}
FEMALE_WORDS = {"girl", "woman", "queen", "mother", "sister", "princess", "heroine", "daughter"}
CONNECTOR_RELATIONS = ("succession (adverb)", "sequence (closes)")      # labels from the Connectors table
EXISTENTIAL = {"past": "There was", "present": "There is", "future": "There will be"}      # opening according to the topic's tense
EXCLUDED_CONNECTORS = {"subsequently"}                 # register too formal for a story
MAX_SENTENCES = 8
GENERIC_MIN = 8                  # a node with >= 8 concrete species below it (same type, main sense) is "generic" and gets drawn at random
LEAF_MIN = 3                      # a candidate with >= 3 candidates below it is intermediate (mammal, bird…) and is not drawn
INHERITANCE = 0.6                    # weight of a verb inherited from an ancestor vs. one of its own
INHERITED_PRIOR = 0.6              # a species that inherits verbs weighs less when drawing than one with its own verbs
# APPROVED ONLY: the WordNet tree under 'person' includes slurs and nationality or religion terms.
# Drawing them at random as protagonist is not acceptable, so among PEOPLE we only draw from the characters that
# the manual seed approves (child, boy, girl, man, woman, king, queen, hero…). With animals there is no such risk.
FREQ_EXP = 0.5                    # frequency bias: score x (1 + times the word appears in WordNet/SemCor) ** FREQ_EXP
# Never drawn. (1) vulgarity safety net (the sense filter already removes them); (2) tree words that are not an
# individual character (collectives, meat, mounts, obscure ones). Hand-written, editable list.
EXCLUDE_FROM_DRAW = {"ass", "pussy", "cock", "bitch", "whore", "slut",
                "mankind", "livestock", "beef", "mount", "bot", "pet", "cattle", "poultry", "game", "wildlife", "fauna"}
SECOND_HOP = 0.5                                   # extra weight for a verb reached through another verb (kept low so that, e.g., laugh does not beat jump)
STEP_ORDER = 0.12                                     # each position further down the seed list subtracts this weight (the first is the most typical)


# ── grammar utilities (WordNet / lemminflect, imported only when needed) ─────────────────────────────────
@lru_cache(maxsize=None)
def _intransitive(verb):
    """Does WordNet have a use of the verb without an object ('Somebody ----s' / 'Something ----s')?"""
    from nltk.corpus import wordnet as wn
    for s in wn.synsets(verb, "v")[:3]:
        for l in s.lemmas():
            if l.name().lower() == verb and set(l.frame_ids()) & {1, 2}:
                return True
    return False


@lru_cache(maxsize=None)
def _is_adverb(word):
    from nltk.corpus import wordnet as wn
    return bool(wn.synsets(word, "r"))


@lru_cache(maxsize=None)
def _form(verb, tense):
    """Conjugated verb: past (jumped), present (jumps) or future (will jump)."""
    from lemminflect import getInflection
    if tense == "future":
        return "will " + verb
    tag = "VBD" if tense == "past" else "VBZ"
    return (getInflection(verb, tag) or (verb + ("ed" if tense == "past" else "s"),))[0]


def _article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def _closeness(node, target, guide=0.5):
    d = math.hypot(node["x"] - target[0], node["y"] - target[1])
    return max(0.0, 1.0 - guide * d / R_MAX)


# ── result ──────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class StorySentence:
    text: str
    protagonist: str
    verb: str
    adverb: str = None
    connector: str = None
    y_arc: float = 0.0               # target height of this sentence in the arc
    nodes: list = field(default_factory=list)       # nodes used (to measure drift)
    trace: list = field(default_factory=list)       # why each word was chosen


@dataclass
class Story:
    input_text: str = ""
    sentences: list = field(default_factory=list)
    target: tuple = (0.0, 0.0)
    walk_result: object = None          # the lcm.walk WalkResult (walk) that set the target
    tense: str = None                # tense of the content, read from the topic's height
    tense_reason: str = None
    seed: int = None               # draw seed; None if nothing was drawn (the story did not depend on chance)
    error: str = None

    @property
    def text(self):
        return " ".join(o.text for o in self.sentences)

    @property
    def coherence(self):
        """1 - mean distance of the used nodes to the target, normalized (same measure as WalkResult.coherencia)."""
        ds = [math.hypot(n["x"] - self.target[0], n["y"] - self.target[1]) for o in self.sentences for n in o.nodes]
        return round(1.0 - min(1.0, sum(ds) / len(ds) / R_MAX), 3) if ds else 0.0

    def to_dict(self):
        return {"input": self.input_text, "text": self.text, "tense": self.tense, "tense_reason": self.tense_reason, "coherence": self.coherence, "seed": self.seed, "error": self.error,
                "sentences": [{"text": o.text, "y_arc": o.y_arc, "trace": o.trace} for o in self.sentences]}


# ── choices ─────────────────────────────────────────────────────────────────────────────────────────────
def _verbs_of(tables, node, via_ok=("manual",)):
    """[(node, weight)] story verbs (marked `narrative_verb` in the seed) without an object (WordNet) that `node` opens."""
    out = []
    for n, tb, via, w in tables.links(node["word"]):
        if tb != "connectors" and via in via_ok and n.get("narrative_verb") and _intransitive(n["word"].lower()):
            out.append((n, w))
    return out


@lru_cache(maxsize=None)
def _first_sense(word):
    from nltk.corpus import wordnet as wn
    ss = wn.synsets(word.replace(" ", "_"), "n")
    return ss[0] if ss else None


@lru_cache(maxsize=None)
def _is_plural(word):
    """'snakes', 'mice', 'giants'… (they have their own node, but a protagonist is singular: 'a snakes' does not exist)."""
    from lemminflect import getLemma
    return getLemma(word, "NOUN")[0] != word


@lru_cache(maxsize=None)
def _frequency(word):
    """How many times the word appears, in its main sense, in the WordNet corpus (SemCor). 0 = obscure."""
    sy = _first_sense(word)
    return sum(l.count() for l in sy.lemmas() if l.name().lower() == word) if sy else 0


@lru_cache(maxsize=None)
def _closing(word):
    """Synsets hanging (hyponyms, recursively) from the main sense of `word`."""
    sy = _first_sense(word)
    return frozenset(sy.closure(lambda x: x.hyponyms())) if sy else frozenset()


@lru_cache(maxsize=None)
def _ancestors(word):
    """Lemmas of the hypernyms of the main sense, from the closest to the farthest."""
    sy = _first_sense(word)
    dist = {}
    if sy:
        for path in sy.hypernym_paths():
            for d, anc in enumerate(reversed(path[:-1]), 1):
                for l in anc.lemmas():
                    w = l.name().lower().replace("_", " ")
                    dist[w] = min(dist.get(w, 99), d)
    return tuple(w for w, _ in sorted(dist.items(), key=lambda t: (t[1], t[0])))


_DESC = {}


def _descendants(tables, node, leaves_only=True):
    """Concrete species under `node` in the tree: biological nodes of its same category whose MAIN SENSE hangs from
    it, without the intermediates (mammal, bird…) unless `leaves_only=False`. If there are >= GENERIC_MIN, `node` is generic."""
    key = (id(tables), node["word"].lower(), leaves_only)
    if key in _DESC:
        return _DESC[key]
    closing = _closing(node["word"].lower())
    cands, sy = [], {}
    for tb in ("subjects", "concepts"):
        for n in tables[tb].nodes:
            w = n["word"].lower()
            if (n.get("type") != "biological" or n.get("category") != node.get("category") or n is node
                    or w in EXCLUDE_FROM_DRAW or _is_plural(w) or id(n) in sy):
                continue
            ps = _first_sense(w)
            if ps is not None and ps in closing:
                cands.append(n)
                sy[id(n)] = ps
    low = Counter(anc for n in cands for path in sy[id(n)].hypernym_paths() for anc in path[:-1])
    sheets = [n for n in cands if low[sy[id(n)]] < LEAF_MIN]
    _DESC[key] = (sheets or cands) if leaves_only else cands
    return _DESC[key]


def _protagonist_verbs(tables, node):
    """([(verb_node, weight)], inherited_from). The OWN verbs (manual seed); if it has none, those of the closest ancestor
    that has them, at weight x INHERITANCE (zebra knows nothing about zebras, but it is an animal: it walks, runs, jumps, sleeps)."""
    proper = _verbs_of(tables, node)
    if proper:
        return proper, None
    from lemminflect import getLemma
    sg = getLemma(node["word"], "NOUN")[0].lower()      # 'ducks' has no verbs of its own but 'duck' does (the seed uses the singular)
    if sg != node["word"].lower():
        for _, hit in tables.find(sg):
            vs = _verbs_of(tables, hit)
            if vs:
                return vs, None
    for anc in _ancestors(node["word"].lower()):
        for _, hit in tables.find(anc):
            vs = _verbs_of(tables, hit)
            if vs:
                return [(n, w * INHERITANCE) for n, w in vs], hit["word"]
    return [], None


def _table_of(tables, node):
    return next((tb for tb, n in tables.find(node["word"]) if n is node), "subjects")


def _choose_protagonist(tables, frame, target, variant, randomize, free_sampling=False):
    """(node, table, qualities, trace). Explicit request ('about a brave king') > walk from the topic.
    If the request is GENERIC ('about an animal') one is drawn among its species (lcm.sampling); among people, only among those
    approved by the seed (the approved-only rule)."""
    a = frame.about_topic
    if a is not None and a.node is not None and a.node.get("type") == "pronoun":
        return None, None, [], [f"no protagonist: I cannot tell a story about '{a.text}' (a pronoun: I/you/he are not characters I have)"]
    if a is not None and a.node is not None and a.node.get("type") == "biological":
        quals = [q for q in frame.about_qualities if any(h["type"] == "quality" for _, h in tables.find(q) if "type" in h)]
        sheets = _descendants(tables, a.node)
        if len(sheets) >= GENERIC_MIN:
            if a.node.get("category") == "person":      # approved characters: also those with subtypes (girl, man, queen…)
                sheets = _descendants(tables, a.node, leaves_only=False)
            cands = []
            for n in sheets:
                vs, inherited = _protagonist_verbs(tables, n)
                if n["word"] in GENERIC_WORDS:
                    continue
                if inherited and a.node.get("category") == "person":
                    continue            # PEOPLE: only characters approved by the seed (see the approved-only rule)
                if vs:
                    cands.append((n, _closeness(n, target) * (INHERITED_PRIOR if inherited else 1.0)
                                  * (1 + _frequency(n["word"].lower())) ** FREQ_EXP))
            if cands:
                n, p = randomize.choose(cands)
                return n, _table_of(tables, n), quals, [
                    f"protagonist: {n['word']}  (RANDOM: '{a.node['word']}' is generic, {len(cands)} candidate species; "
                    f"T={randomize.temperature:g}, seed={randomize.seed}; came out with p={p:.1%})"]
        return a.node, a.table, quals, [f"protagonist: {a.node['word']} (you asked for it: 'about …')"]

    topic = frame.topic.node
    best = {}                                       # word -> (score, node, table, path)
    boundary = [(topic, topic["word"], 1.0)]
    for level_value in range(2):
        fresh = []
        for node, route, weight in boundary:
            for n, tb, via, w in tables.links(node["word"]):
                if tb == "connectors" or via not in PROTAGONIST_VIAS:
                    continue
                pw = weight * w
                fresh.append((n, f"{route} -> {n['word']} ({via})", pw))
                if (n.get("type") == "biological" and n.get("category") in PROTAGONIST_CATEGORIES
                        and n["word"] not in GENERIC_WORDS and _verbs_of(tables, n)):
                    sc = pw * _closeness(n, target)
                    if n["word"] not in best or sc > best[n["word"]][0]:
                        best[n["word"]] = (sc, n, tb, route + f" -> {n['word']} ({via})")
        boundary = fresh
    if not best:
        return None, None, [], ["no protagonist: no living being reachable from the topic"]
    ranking = sorted(best.values(), key=lambda t: (-t[0], t[1]["word"]))
    if free_sampling:                                   # --randomize (random): draw among the top 5 instead of rotating by `variant` (variant)
        top = ranking[:5]
        (sc, n, tb, route), p = randomize.choose([(t, t[0]) for t in top])
        return n, tb, [], [f"protagonist: {n['word']}  ({route})  (RANDOM draw among the {len(top)} best; "
                           f"T={randomize.temperature:g}, seed={randomize.seed}; p={p:.1%})"]
    sc, n, tb, route = ranking[variant % min(len(ranking), 5)]
    return n, tb, [], [f"protagonist: {n['word']}  ({route})  score {sc:.2f}  [#{variant % min(len(ranking), 5) + 1} of {min(len(ranking), 5)}]"]


def _verb_score(n, w, y_arc, target, tense="past"):
    arc = max(0.0, 1.0 - abs(n["y"] - y_arc) / 6.0)
    pos = shift(n, tense)
    near = max(0.0, 1.0 - 0.5 * math.hypot(pos["x"] - target[0], pos["y"] - target[1]) / R_MAX)
    return w * arc * near


def _verb_sequence(tables, prot, ys, target, rank_index=0, pool_size=9, tense="past"):
    """Best sequence of DISTINCT verbs for the heights `ys`, optimized as a whole (not sentence by sentence:
    a greedy selector spends 'sleep' on the 3rd sentence and the last one is left with no low verb).

    Pool = protagonist's verbs (link weight) ∪ verbs those verbs open (weight x SECOND_HOP):
    what the protagonist does DIRECTLY weighs more than what is reached through another action. If the protagonist
    INHERITS its verbs, there is no second jump (a zebra does not fly because 'jump' opens 'fly').
    Returns [(node, score, trace, arc_y)] or [] if there are no verbs. If there are fewer verbs than sentences, the arc is
    RESCALED to the ones available (before, it was cut and the closure was lost). `rank_index` picks the 1st, 2nd… best sequence.
    """
    from itertools import permutations
    proper, inherited = _protagonist_verbs(tables, prot)
    from0 = prot["word"] + (f" (inherited from {inherited})" if inherited else "")
    pool = {n["word"]: (n, w, from0) for n, w in proper}
    if not inherited:
        for n0, w0, _ in list(pool.values()):
            for n, w in _verbs_of(tables, n0):
                if n["word"] not in pool or w0 * w * SECOND_HOP > pool[n["word"]][1]:
                    pool[n["word"]] = (n, w0 * w * SECOND_HOP, n0["word"])
    # the most promising ones for some point of the arc
    order = sorted(pool.values(), key=lambda t: (-max(_verb_score(t[0], t[1], y, target, tense) for y in ys), t[0]["word"]))
    order = order[:max(pool_size, len(ys))]
    if not order:
        return []
    if len(order) < len(ys):                         # rescale the arc +2 -> -2 to the number of available verbs
        m = len(order)
        ys = [2.0 - 4.0 * k / max(1, m - 1) for k in range(m)]
    table = [[_verb_score(n, w, y, target, tense) for y in ys] for n, w, _ in order]
    best_ones = []
    for perm in permutations(range(len(order)), len(ys)):
        best_ones.append((sum(table[i][k] for k, i in enumerate(perm)), tuple(order[i][0]["word"] for i in perm), perm))
    best_ones.sort(key=lambda t: (-t[0], t[1]))
    sc, _, perm = best_ones[rank_index % min(len(best_ones), 3)]
    return [(order[i][0], table[i][k], f"verb: {order[i][0]['word']}  (from {order[i][2]}; y={order[i][0]['y']:+.1f} vs arc {ys[k]:+.1f})  score {table[i][k]:.2f}", ys[k])
            for k, i in enumerate(perm)]


def _choose_adverb(tables, verb, target, used_items=()):
    order = verb.get("related_manual", [])             # the seed order expresses typicality
    cands = [(w * max(0.2, 1.0 - STEP_ORDER * order.index(n["word"])) * _closeness(n, target), n)
             for n, tb, via, w in tables.links(verb["word"])
             if via == "manual" and tb == "concepts" and n.get("type") == "quality" and n["word"] not in used_items
             and n["word"] in order and _is_adverb(n["word"].lower())]
    if not cands:
        return None, None
    cands.sort(key=lambda t: (-t[0], t[1]["word"]))
    sc, n = cands[0]
    return n, f"adverb: {n['word']}  (from {verb['word']}; score {sc:.2f})"


def _choose_connector(tables, y_arc, variant, last, used_items):
    """Connector from the Connectors table: the RELATION is fixed by the position (middle: succession; last: closure), and within
    it the one closest in height to the sentence is chosen, without repeating (then/next up, finally down)."""
    rel = CONNECTOR_RELATIONS[1] if last else CONNECTOR_RELATIONS[0]
    cands = [n for n in tables["connectors"].nodes if n["relation"] == rel and " " not in n["word"]
             and n["word"] not in used_items and n["word"] not in EXCLUDED_CONNECTORS]
    if not cands:                                     # exhausted: repetition is allowed
        cands = [n for n in tables["connectors"].nodes if n["relation"] == rel and " " not in n["word"]]
    best = min(abs(n["y"] - y_arc) for n in cands)
    neighbors = [n for n in cands if abs(n["y"] - y_arc) <= best + 1e-9]
    n = neighbors[variant % len(neighbors)]
    return n, f"connector: {n['word']}  (relation '{rel}', y={n['y']:+.1f} ~ arc {y_arc:+.1f})"


def _pronoun(tables, prot):
    p = "he" if prot["word"] in MALE_WORDS else "she" if prot["word"] in FEMALE_WORDS else "it"
    return p if tables["subjects"].get(p) else "it"


# ── API ────────────────────────────────────────────────────────────────────────────────────────────────────
def narrate(tables, frame, n_sentences=4, variant=0, seed=None, temperature=1.0, top_k=None, randomize=False):
    """A short story from a Frame (lcm.parse). `variant` rotates among the best alternatives (deterministic).

    Randomness: if the requested protagonist is generic ('about an animal') one is ALWAYS drawn among its species; with `azar=True`
    one is also drawn (among the top 5) when no protagonist was requested. `seed` (same seed = same story; if
    none is given, one is generated and kept in `Story.semilla`), `temperature` and `top_k` are the knobs of the draw (lcm.sampling).
    """
    r = Story(input_text=frame.input_text)
    draw = Sampler(seed, temperature, top_k)
    topic = frame.topic
    if topic is None or topic.node is None:
        r.error = "there is no topic anchored in the tables: I do not know what to tell about"
        return r
    is_story = topic.lemma_text in STORY_WORDS or topic.node["word"] in STORY_WORDS or \
        any(s in STORY_WORDS for s in topic.node.get("synonyms", []))
    if not is_story:
        r.error = f"only the story construction exists; the topic '{topic.text}' has no narrative construction"
        return r

    n_sentences = max(1, min(MAX_SENTENCES, int(n_sentences)))
    y_topic = topic.node["y"]
    r.tense = tense_of_position(y_topic)
    zone = {"past": "South", "present": "present band", "future": "North"}[r.tense]
    r.tense_reason = (f"{topic.node['word']} is at y = {y_topic:+.1f} ({zone}) -> {r.tense}: opens with '{EXISTENTIAL[r.tense]}'"
                    + (f"; '{frame.action.lemma_text}' asks for the act in {frame.tense}, but the content takes the tense of the topic"
                       if frame.action and frame.tense != r.tense else ""))
    r.walk_result = walk(tables, frame, max_steps=14)
    r.target = target_of(frame)
    prot, tb, quals, trace0 = _choose_protagonist(tables, frame, r.target, variant, draw, randomize)
    r.seed = draw.seed if draw.used else None
    if prot is None:
        r.error = trace0[0]
        return r

    pron = _pronoun(tables, prot)
    used_connectors, used_adverbs = set(), set()
    ys = [2.0 - 4.0 * k / max(1, n_sentences - 1) for k in range(n_sentences)]     # +2 -> -2
    seq = _verb_sequence(tables, prot, ys, r.target, rank_index=variant // 5, tense=r.tense)
    for k, (v, sc, tv, y) in enumerate(seq):
        adv, ta = _choose_adverb(tables, v, r.target, used_adverbs)
        if adv:
            used_adverbs.add(adv["word"])
        past_w = _form(v["word"].lower(), r.tense)
        queue = (" " + adv["word"]) if adv else ""
        nodes, trace = [prot, shift(v, r.tense) | {"word": v["word"]}], ([f"tense: {r.tense_reason}", *trace0] if k == 0 else []) + [tv]
        if adv:
            nodes.append(adv); trace.append(ta)
        if k == 0:
            q = (" ".join(quals) + " ") if quals else ""
            text = f"{EXISTENTIAL[r.tense]} {_article(q or prot['word'])} {q}{prot['word']} who {past_w}{queue}."
            with_conn = None
        else:
            with_conn, tc = _choose_connector(tables, y, variant + k, k == len(seq) - 1, used_connectors)
            used_connectors.add(with_conn["word"])
            trace.append(tc)
            nodes.append(with_conn)
            text = f"{with_conn['word'].capitalize()}, {pron} {past_w}{queue}."
        r.sentences.append(StorySentence(text, prot["word"], v["word"], adv["word"] if adv else None,
                                   with_conn["word"] if with_conn else None, y, nodes, trace))
    if not r.sentences:
        r.error = f"'{prot['word']}' has no linked actions without an object"
    return r


def stories(tables, frame, n_sentences=4, seed=None, **sampling_kw):
    """As many stories as the sentence asks for ('two stories' -> 2, 'stories' -> 3), each with another variant and, if there is a
    draw, another derived seed (seed + i), so that 'tell me two stories about an animal' gives two animals."""
    c = frame.quantity
    n = c if isinstance(c, int) and c > 0 else 3 if isinstance(c, str) and c in ("several", "some", "many") else 1
    return [narrate(tables, frame, n_sentences, variant=i, seed=None if seed is None else seed + i, **sampling_kw)
            for i in range(min(n, 5))]
