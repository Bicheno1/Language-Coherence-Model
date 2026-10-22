# build_tables.py — LCM: Subjects table and Concepts table
#
# Stepped weights (integers 0-10) per node: V (Context), I (World), Lv (Imposition), Gv (Adaptation).
# Position (GDE, reading D1):  x = (Lv - Gv)/2     y = (V - I)/2
# Cell in the 10 x 10 table: col = ceil(x + 5), row = ceil(5 - y), clipped to 1..10.
#
# Every number in this file is a DESIGN DECISION (first pass) and can be edited.
# Data sources: Unique Host data (source_data/db), SentiWordNet (valence) and the connectors table (connectors_data.py).
#
# Usage:  python tools/build_tables.py            on-screen diagnostics
#         python tools/build_tables.py --write    regenerates data/LCM_tables.xlsx and data/LCM_tables.json
#         python tools/build_tables.py [SRC] [OUT] --write     alternative paths

import sys, json, math
from collections import Counter, defaultdict

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
# By default it reads the Unique Host data that ships inside the package (source_data/) and writes to data/.
SRC = _args[0] if _args else os.path.join(_HERE, "..", "source_data")
OUT = _args[1] if len(_args) > 1 else os.path.join(_HERE, "..", "data")
sys.path.insert(0, SRC)

from db.db_lexicon import LEXICON
from db.db_concepts import CONCEPTS
import db.db_axis_lexicon as AX
from db.db_axis_lexicon_generated import ADJECTIVES_EAST, ADJECTIVES_WEST
from nltk.corpus import sentiwordnet as swn

N = 10  # table size (N x N)

# ── 1. DEFAULT WEIGHTS, INHERITED FROM THE TREE ──────────────────────────────
# Order of each tuple: (V, I, Lv, Gv)
#   V  = weight toward Context   (viable, open, possible, alive)
#   I  = weight toward World     (closed, dangerous, logical, inert, past)
#   Lv = weight toward Imposition (sensory, concrete, "I")
#   Gv = weight toward Adaptation (structural, relational, "you")

# Level 1: node type (biological=Subject/North, non_biological=Object/South, action=Sensory/East, quality=Structural/West)
TYPE_BASE = {
    "biological":     (6, 1, 4, 7),   # alive: up and toward Adaptation (NW quadrant)
    "non_biological": (2, 4, 6, 3),   # inert: down and toward Imposition (SE quadrant)
    "action":         (4, 2, 7, 3),   # sensory: toward Imposition
    "quality":        (3, 1, 3, 7),   # structural: toward Adaptation
}

# Level 2: category (ADDED to level 1)
CAT_ADJ = {
    # biological
    "person":        (1, 0, -1, 1),
    "animal":        (0, 1, -1, 1),
    "plant":         (0, 0, 1, -2),
    "body_part":     (-1, 1, 4, -4),
    "group":         (0, 1, -1, 1),
    # non_biological
    "object":        (0, 0, 2, -1),
    "substance":     (0, 1, 2, -1),
    "food":          (3, -3, 2, -1),
    "place":         (1, 0, 0, 1),
    "weather":       (0, 2, 1, 0),
    "time":          (0, 2, -4, 3),
    "communication": (3, -1, -2, 5),
    "cognition":     (3, -1, -3, 5),
    "attribute":     (2, 0, -1, 4),
    "state":         (0, 1, 0, 2),
    "event":         (0, 2, 1, 0),
    "act":           (2, 0, 1, 1),
    "possession":    (2, 0, -1, 3),
    "quantity":      (1, 1, -2, 4),
    "shape":         (1, 1, 2, 1),
    "relation":      (2, 0, -3, 5),
    "process":       (0, 2, 0, 1),
    "abstract":      (0, 2, -1, 2),
    "motive":        (3, 0, 0, 2),
    "emotion":       (1, -3, 2, -1),   # a feeling is not inert: low I unless valence raises it
    "substance_":    (0, 0, 0, 0),
    "action":        (0, 0, 0, 0),
    "quality":       (0, 0, 0, 0),
}

# Level 3: subtype, key (category, subtype)  (ADDED)
SUB_ADJ = {
    ("action", "communication"): (1, 0, -3, 5), ("action", "social"): (1, 0, -2, 5),
    ("action", "cognition"): (1, 0, -4, 5),     ("action", "possession"): (0, 1, -3, 3),
    ("action", "stative"): (0, 1, -4, 3),       ("action", "motion"): (1, 0, 1, -1),
    ("action", "contact"): (0, 1, 2, -1),       ("action", "perception"): (0, 0, 3, -2),
    ("action", "creation"): (2, -1, -1, 2),     ("action", "emotion"): (0, 0, 1, 1),
    ("action", "competition"): (0, 2, 0, 0),    ("action", "consumption"): (1, 0, 2, -1),
    ("action", "body"): (0, 0, 2, -1),          ("action", "change"): (0, 1, -1, 1),
    ("action", "weather"): (0, 1, 1, 0),
    ("quality", "sensory"): (0, 0, 5, -5),      ("quality", "physical"): (0, 0, 4, -4),
    ("quality", "speed"): (0, 0, 4, -4),        ("quality", "emotional"): (0, 0, 2, -2),
    ("quality", "temporal"): (0, 1, -1, 0),     ("quality", "quantity"): (0, 0, -1, 0),
    ("quality", "difficulty"): (-1, 2, 0, 0),   ("quality", "importance"): (1, 0, -1, 0),
    ("quality", "relational"): (0, 0, -1, 1),   ("quality", "evaluative"): (0, 0, -2, 2),
    ("quality", "personality"): (0, 0, -1, 1),  ("quality", "certainty"): (0, 0, -2, 1),
    ("quality", "comparative"): (0, 0, -2, 2),  ("quality", "cognitive"): (0, 0, -2, 1),
    ("quality", "social"): (0, 0, -2, 2),       ("quality", "aesthetic"): (1, 0, -2, 2),
    ("object", "weapon"): (-1, 3, 1, -1),       ("object", "drug"): (-1, 2, 1, -1),
    ("object", "vehicle"): (0, 0, 1, 0),        ("object", "clothing"): (1, -1, 1, 0),
    ("object", "container"): (0, 0, 1, 0),      ("object", "furniture"): (0, 0, 1, 0),
    ("object", "mechanism"): (0, 1, 1, 0),
    ("body_part", "external"): (0, 0, 1, -1),   ("body_part", "internal"): (-1, 1, -1, 0),
    ("body_part", "fluid"): (0, 1, 0, 0),       ("body_part", "genital"): (0, 0, 1, -1),
    ("place", "building"): (1, 0, 0, 1),        ("place", "landform"): (0, 1, 1, -1),
}

# Per-axis lists already written in Unique Host (db_axis_lexicon.py)
NORTH_CONCEPTS = {w.lower() for w in AX.AXIS_NORTH["concepts"]}
SOUTH_CONCEPTS = {w.lower() for w in AX.AXIS_SOUTH["concepts"]}
# v0.4c: words whose MEANING is something that already happened (a story tells what happened). The generator had given them
# the position of any 'communication' word (story: y = +1, North), so the narrator had to force the
# past. They get the same past shift as lcm.tense.TENSE_SHIFT["pasado"] = (V-2, I+3).
# 'history' is not listed: it already fell in the South (y = -4).
INTRINSIC_PAST = {"story", "tale", "legend", "myth", "fable", "anecdote", "memory", "memoir", "biography", "chronicle",
                     "saga", "folktale"}
PAST_SHIFT = (-2, 3, 0, 0)
HAND_EAST = {w.lower() for k, v in AX.AXIS_EAST.items() if k != "subjects" for w in v}
HAND_WEST = {w.lower() for k, v in AX.AXIS_WEST.items() if k != "subjects" for w in v}
GEN_EAST  = {w.lower() for w in ADJECTIVES_EAST}
GEN_WEST  = {w.lower() for w in ADJECTIVES_WEST}

# Loading of the hand-made concepts (danger, intrinsic emotion, state type)
EMOTION_ADJ = {"joy": (3, 0, 0, 0), "warmth": (3, 0, 0, 1), "pride": (2, 0, 2, 0), "awe": (1, 1, 1, 0),
               "anger": (-1, 3, 2, 0), "fear": (-1, 4, 0, 0), "sadness": (-1, 3, 0, 0), "shame": (-1, 2, 0, 1), "meta": (0, 0, 0, 0)}
STATUS_ADJ = {"threat": (-1, 4, 0, 0), "safety": (4, -1, 0, 0), "social": (0, 0, -1, 3), "judgment": (0, 0, -2, 4),
              "physical": (0, 0, 3, -1), "condition": (0, 1, 0, 1), "emotional": (0, 0, 2, 0)}
DANGER_ADJ = (-2, 4, 0, 0)
SENSE_ADJ = (0, 0, 2, -1)
VALENCE_SCALE = 6      # how much SentiWordNet weighs: V += round(6 * pos), I += round(6 * neg)

# WordNet lineage (first sense, same as the lexicon) to tell people apart: the "person" leaf of the tree
# has almost no words of its own, and the 998 people would share the same weights.
from nltk.corpus import wordnet as wn
LINEAGE_ADJ = [   # (lineage lemmas, adjustment (V, I, Lv, Gv)); the first matching group wins
    ({"bad_person", "wrongdoer", "criminal", "unwelcome_person", "adversary", "enemy", "killer", "tyrant", "villain"}, (-2, 3, 2, -2)),   # threat
    ({"leader", "head", "principal", "serviceman", "lawman", "contestant", "defender", "administrator", "politician",
      "spiritual_leader", "capitalist", "boss", "ruler", "soldier", "warrior", "fighter", "athlete"}, (0, 0, 3, -3)),                   # imposes itself
    ({"relative", "ancestor", "genitor", "progenitor", "parent", "offspring", "child", "juvenile", "domestic_partner",
      "peer", "associate", "friend", "lover", "spouse"}, (2, 0, -2, 2)),                                                                # close bond
    ({"preserver", "health_professional", "medical_practitioner", "communicator", "intellectual", "expert", "professional",
      "teacher", "educator", "servant", "helper", "negotiator", "representative", "creator", "entertainer", "performer"}, (1, 0, -1, 2)),  # cares, teaches, communicates
    ({"unfortunate"}, (-1, 2, 0, 1)),
    ({"imaginary_being", "spiritual_being"}, (0, 2, 2, 0)),
]
ANIMAL_LINEAGE_ADJ = [({"domestic_animal"}, (1, 0, 0, 1)), ({"carnivore", "predator", "bird_of_prey", "big_cat"}, (-1, 2, 2, -1))]
_lin_cache = {}
def lineage(word, pos="n"):
    if word not in _lin_cache:
        ss = wn.synsets(word.replace(" ", "_"), pos)
        _lin_cache[word] = {l.name().lower() for h in (ss[0].closure(lambda s: s.hypernyms()) if ss else []) for l in h.lemmas()}
    return _lin_cache[word]

# Pronouns per axis (db_axis_lexicon.py, "subjects" lists), with fixed weights that reproduce the diagram
PRONOUNS = {
    "Context (We)":      (["we", "us", "our", "ours", "ourselves"],  (10, 0, 5, 5)),
    "Imposition (I)":    (["i", "me", "my", "mine", "myself"],       (6, 2, 10, 0)),
    "Adaptation (You)":  (["you", "your", "yours", "yourself"],      (6, 2, 0, 10)),
    "World (That)":      (["that", "it", "those", "itself"], (0, 10, 5, 5)),
    # v0.4 (problem H): third person and demonstratives that were missing. Design weights, editable.
    # Third person: a living being outside the dialogue -> viable and relational, but less than 'you' (not present).
    "They (3rd person)": (["he", "she", "they", "him", "her", "his", "hers", "their", "theirs", "them",
                            "himself", "herself", "themselves"], (7, 2, 3, 7)),
    # Demonstratives of the near: like 'that' (World) but toward Imposition (what the speaker points at).
    "This (near)":    (["this", "these"], (1, 8, 8, 2)),
}

# ── 2. UTILITIES ────────────────────────────────────────────────────────────
def add(a, b):
    return tuple(x + y for x, y in zip(a, b))

def clamp(v):
    return max(0, min(10, int(math.floor(v + 0.5))))

_senti_cache = {}
def senti(word, pos):
    key = (word, pos)
    if key not in _senti_cache:
        ss = list(swn.senti_synsets(word.replace(" ", "_"), pos))[:3]
        _senti_cache[key] = (sum(s.pos_score() for s in ss) / len(ss), sum(s.neg_score() for s in ss) / len(ss)) if ss else (0.0, 0.0)
    return _senti_cache[key]

def cell_of(V, I, Lv, Gv):
    x, y = (Lv - Gv) / 2.0, (V - I) / 2.0
    return x, y, max(1, min(N, math.ceil(x + 5))), max(1, min(N, math.ceil(5 - y)))

SWN_POS = {"biological": "n", "non_biological": "n", "action": "v", "quality": "a"}

def weights(word, ntype, cat, sub, concept=None):
    w = TYPE_BASE[ntype]
    w = add(w, CAT_ADJ.get(cat, (0, 0, 0, 0)))
    w = add(w, SUB_ADJ.get((cat, sub), (0, 0, 0, 0)))
    wl = word.lower()
    # horizontal axis of adjectives: verified East (sensory) / West (judgment, relation) lists
    # (the hand-written lists take precedence over the generated ones: e.g. "beautiful" is West even though the heuristic gave East)
    if ntype == "quality":
        if wl in HAND_WEST or (wl not in HAND_EAST and wl in GEN_WEST):
            w = (w[0], w[1], min(w[2], 2), max(w[3], 8))
        elif wl in HAND_EAST or wl in GEN_EAST:
            w = (w[0], w[1], max(w[2], 8), min(w[3], 2))
    if cat in ("person", "animal") and ntype == "biological":
        lin = lineage(wl)
        for lemmas, adj in (LINEAGE_ADJ if cat == "person" else ANIMAL_LINEAGE_ADJ):
            if lin & lemmas:
                w = add(w, adj)
                break
    if wl in NORTH_CONCEPTS:
        w = add(w, (3, -1, 0, 0))
    if wl in SOUTH_CONCEPTS:
        w = add(w, (-1, 3, 0, 0))
    if wl in INTRINSIC_PAST and ntype == "non_biological":
        w = add(w, PAST_SHIFT)
    # valence: SentiWordNet (positive -> V, negative -> I), independent of each other
    pos, neg = senti(wl, SWN_POS[ntype])
    if concept is not None and concept.get("type") == "status":   # emotions/states: take the POS with the most signal
        for p in ("n", "a"):
            a, b = senti(wl, p)
            if a + b > pos + neg:
                pos, neg = a, b
    w = add(w, (round(VALENCE_SCALE * pos), round(VALENCE_SCALE * neg), 0, 0))
    # loading of the hand-made concepts
    if concept is not None:
        if concept.get("intrinsic_danger"):
            w = add(w, DANGER_ADJ)
        if concept.get("intrinsic_emotion") in EMOTION_ADJ:
            w = add(w, EMOTION_ADJ[concept["intrinsic_emotion"]])
        if concept.get("type") == "status" and concept.get("subtype") in STATUS_ADJ:
            w = add(w, STATUS_ADJ[concept["subtype"]])
        if concept.get("type") == "illogical":
            w = add(w, (-2, 4, 1, 0))
        if concept.get("type") == "tone":
            w = add(w, (0, 0, 4, -2))
        if concept.get("type") == "music":
            w = add(w, (2, 0, 3, 0))
        if concept.get("sense"):
            w = add(w, SENSE_ADJ)
    return tuple(clamp(v) for v in w), round(pos, 2), round(neg, 2)

def concept_class(c):
    """(ntype, category, table) according to the type of the hand-made concept."""
    t, st = c["type"], c.get("subtype")
    if t == "subject":
        return "biological", {"animal": "animal", "body_part": "body_part", "group": "group"}.get(st, "person"), "subjects"
    if t == "object":      return "non_biological", OBJ_SUB_TO_CAT.get(st, "object"), "concepts"
    if t == "action":      return "action", "action", "concepts"
    if t == "quality":     return "quality", "quality", "concepts"
    if t == "status":      return "non_biological", ("emotion" if st == "emotion" else "state"), "concepts"
    if t == "illogical":   return "non_biological", "abstract", "concepts"
    if t == "environment": return "non_biological", "place", "concepts"
    return "non_biological", "attribute", "concepts"      # tone, music

OBJ_SUB_TO_CAT = {"object": "object", "act": "act", "state": "state", "emotion": "emotion", "hazard": "event",
                  "injury": "state", "event": "event", "weather": "weather"}

# ── 3. BUILDING THE NODES ────────────────────────────────────────────────────
nodes = {"subjects": [], "concepts": []}
seen = set()

def make_node(table, word, ntype, cat, sub, W, pos, neg, source, syn, concept):
    V, I, Lv, Gv = W
    x, y, col, row = cell_of(V, I, Lv, Gv)
    syn = [s for s in (syn or []) if s.lower() != word.lower()]
    nodes[table].append({
        "word": word, "type": ntype, "category": cat, "subtype": sub,
        "V": V, "I": I, "Lv": Lv, "Gv": Gv, "x": x, "y": y, "col": col, "row": row,
        "tension": min(V, I), "source": source, "sent_pos": pos, "sent_neg": neg,
        "synonyms": syn, "_related": list((concept or {}).get("related", {}).keys()),
    })
    seen.add(word.lower())

# 3a. pronouns (Subjects)
for label, (words, W) in PRONOUNS.items():
    for w in words:
        make_node("subjects", w, "pronoun", label, None, W, 0.0, 0.0, "pronoun", [x for x in words if x != w], None)

# 3b. full lexicon: biological -> Subjects; the rest -> Concepts
for word, e in LEXICON.items():
    if word.lower() in seen:
        continue
    c = CONCEPTS.get(word)
    ntype, cat, sub = e["node_type"], e["category"], e.get("subtype")
    if c is not None and c["type"] in ("illogical", "status", "environment", "tone", "music"):
        ntype, cat, _ = concept_class(c)       # e.g. "ghost": WordNet gave it as "cognition"; the concept says illogical
    W, pos, neg = weights(word, ntype, cat, sub, c)
    table = "subjects" if ntype == "biological" else "concepts"
    make_node(table, word, ntype, cat, sub, W, pos, neg, "concept+lexicon" if c else "lexicon", e.get("synonyms"), c)

# 3c. hand-made concepts that are NOT in the lexicon
SKIPPED = Counter()
for name, c in CONCEPTS.items():
    if name.lower() in seen:
        continue
    if c.get("inflection_of"):
        SKIPPED["inflection of another word"] += 1
        continue
    t, st = c["type"], c.get("subtype")
    if t == "language":
        SKIPPED["language (questions/pronouns: they will go to the Connectors table)"] += 1
        continue
    ntype, cat, table = concept_class(c)
    W, pos, neg = weights(name, ntype, cat, None, c)
    make_node(table, name, ntype, cat, st, W, pos, neg, "concept", c.get("synonyms"), c)

# 3d. word from the diagram that is not in the lexicon
make_node("concepts", "dehumanization", "non_biological", "abstract", None, (0, 10, 5, 5), 0.0, 0.0, "drawing", [], None)

# 3e. related: separate links to nodes (they cross tables) from state labels without a node
all_words = {n["word"].lower() for t in nodes.values() for n in t}
for t in nodes.values():
    for n in t:
        rel = n.pop("_related")
        n["related"] = [r for r in rel if r.lower() in all_words and r.lower() != n["word"].lower()]
        n["related_labels"] = [r for r in rel if r.lower() not in all_words]

# 3e2. v0.4 — related derived from WordNet, manual seed and opposites
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import related_wn as RW
from related_manual import MANUAL as RELATED_MANUAL, NARRATIVE_VERBS, OPINION_ADJECTIVES, STATE_ADJECTIVES
_by_word = {n["word"].lower(): n for t in nodes.values() for n in t}
_derived = RW.derive(_by_word)
for w, n in _by_word.items():
    n["related_wn"], n["opposites"] = _derived[w]
    n["related_manual"] = [r for r in RELATED_MANUAL.get(w, []) if r in _by_word and r != w]
    for _sign, _join_words in OPINION_ADJECTIVES.items():
        if w in _join_words and n["type"] == "quality":
            n["opinion"] = _sign              # 'pos' | 'neg': adjective the system can evaluate with (lcm/reply.py)
    for _sign, _join_words in STATE_ADJECTIVES.items():
        if w in _join_words and n["type"] == "quality":
            n["state"] = _sign               # 'pos' | 'neg': state adjective the system can answer 'how are you' with (lcm/reply.py)
    if w in NARRATIVE_VERBS:
        n["narrative_verb"] = True        # only these can be the verb of a generated sentence (lcm/narrate.py)

# 3f. Connectors table — see connectors_data.py
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import connectors_data as CD
nodes["connectors"] = []

def primary_axis(x, y):
    """Main axis: vertical if |y| >= 1.5 (North future / South past); otherwise, the side of the present (West / East)."""
    return "North" if y >= 1.5 else "South" if y <= -1.5 else ("West" if x <= -1 else "East" if x >= 1 else "Present")

def proposed_axis(x, y):
    base = "North" if y >= 1.5 else "South" if y <= -1.5 else "Present"
    return base + (" · West" if x <= -1 else " · East" if x >= 1 else "")

def conn_node(word, rel, cls, cons, amb, note, W, group, rel_label):
    V, I, Lv, Gv = W
    x, y, col, row = cell_of(V, I, Lv, Gv)
    actual = sorted({CD.CONS_AXIS_SPLIT.get((c, rel), CD.CONS_AXIS[c]) for c in cons if c in CD.CONS_AXIS})
    prop = proposed_axis(x, y)
    compat = "no construction" if not actual else ("matches" if primary_axis(x, y) in actual else "differs")
    nodes["connectors"].append({
        "word": word, "group": group, "relation": rel_label, "class": cls,
        "V": V, "I": I, "Lv": Lv, "Gv": Gv, "x": x, "y": y, "col": col, "row": row, "tension": min(V, I),
        "construction": list(cons), "current_axis": "/".join(actual), "proposed_axis": prop, "compat": compat,
        "multiword": (" " in word and word not in ("( )", '" "')) or "…" in word, "ambiguous": amb, "note": note,
        "source": "connectors_data.py",
    })

for word, rel, cls, cons, amb, note in CD.CONNECTORS:
    conn_node(word, rel, cls, cons, amb, note, CD.REL[rel], CD.GROUP_OF[rel], CD.REL_LABEL[rel])
for sign, name, V, I, Lv, Gv, cons, note in CD.PUNCT:
    conn_node(sign, "punct", "PUNCT", cons, False, f"{name}: {note}", (V, I, Lv, Gv), "punctuation", "punctuation")

# ── 4. DIAGNOSTICS ───────────────────────────────────────────────────────────
def find(table, word):
    return next((n for n in nodes[table] if n["word"].lower() == word), None)

if __name__ == "__main__":
    print("nodes:", {k: len(v) for k, v in nodes.items()})
    print("omitted:", dict(SKIPPED))
    for k, v in nodes.items():
        q = Counter(("Context" if n["y"] > 0 else "World" if n["y"] < 0 else "center") + "/" +
                    ("Imposition" if n["x"] > 0 else "Adaptation" if n["x"] < 0 else "axis") for n in v)
        print(f"\n[{k}] quadrants:", dict(q))
        rows = Counter(n["row"] for n in v); cols = Counter(n["col"] for n in v)
        print("  by row:", [rows.get(i, 0) for i in range(1, N + 1)])
        print("  by col :", [cols.get(i, 0) for i in range(1, N + 1)])
    cn = nodes["connectors"]
    print("\nconnectors:", len(cn), "| compat:", dict(Counter(n["compat"] for n in cn)))
    print("  by proposed axis:", dict(Counter(n["proposed_axis"].split(" ")[0] for n in cn)))
    print("  differ from the current axis:", [(n["word"], n["current_axis"], n["proposed_axis"]) for n in cn if n["compat"] == "differs"])
    print("\n— drawing anchors —")
    for tb, w in [("subjects", "us"), ("subjects", "you"), ("subjects", "i"), ("subjects", "animal"), ("concepts", "desert"), ("concepts", "dehumanization")]:
        n = find(tb, w)
        print(f"  {w:15s} V{n['V']} I{n['I']} Lv{n['Lv']} Gv{n['Gv']} -> x={n['x']:+.1f} y={n['y']:+.1f}  col {n['col']} row {n['row']}")
    print("\n— samples —")
    for tb, w in [("subjects", "person"), ("subjects", "dog"), ("subjects", "puppy"), ("subjects", "intruder"), ("concepts", "story"), ("concepts", "jump"),
                  ("concepts", "happy"), ("concepts", "sad"), ("concepts", "fear"), ("concepts", "ghost"), ("concepts", "fire"), ("concepts", "stone"),
                  ("concepts", "beautiful"), ("concepts", "loud"), ("concepts", "red"), ("concepts", "kill"), ("concepts", "love"), ("concepts", "hope"), ("concepts", "history")]:
        n = find(tb, w)
        print(f"  {w:10s}" + (f" {n['type'][:6]:6s}/{n['category'][:8]:8s} V{n['V']:2d} I{n['I']:2d} Lv{n['Lv']:2d} Gv{n['Gv']:2d} -> col {n['col']:2d} row {n['row']:2d}" if n else " — does not exist"))


# ── 5. OUTPUTS: JSON and Excel ─────────────────────────────────────────────────
def write_outputs():
    import os
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter as L

    os.makedirs(OUT, exist_ok=True)
    PRIORITY = {"pronoun": 0, "drawing": 1, "concept+lexicon": 2, "concept": 3, "lexicon": 4, "connectors_data.py": 5}
    order = {id(n): i for t in nodes.values() for i, n in enumerate(t)}   # lexicon order ~ frequency
    for t in nodes.values():
        t.sort(key=lambda n: (n["row"], n["col"], PRIORITY[n["source"]], order[id(n)]))

    # ---------- JSON ----------
    data = {
        "version": "0.4",
        "weight_scale": "integers 0-10 (0 = no weight, 10 = maximum)",
        "axes": {"V": "Context (We, Possibility) - up", "I": "World (That, logical) - down",
                 "Lv": "Imposition (I, Sensory) - right", "Gv": "Adaptation (You, structural) - left"},
        "position": {"x": "(Lv - Gv) / 2", "y": "(V - I) / 2"},
        "cell_rule": f"col = ceil(x + 5), row = ceil(5 - y), clipped to 1..{N}. Row 1 = Context, row {N} = World; col 1 = Adaptation, col {N} = Imposition",
        "tables": {},
    }
    for name, lst in nodes.items():
        cells = defaultdict(list)
        for n in lst:
            cells[f"{n['row']},{n['col']}"].append(n["word"])
        data["tables"][name] = {"nodes": lst, "cells": dict(cells)}
    with open(f"{OUT}/LCM_tables.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    # ---------- Excel ----------
    FN = "Arial"
    f_norm, f_b = Font(name=FN, size=10), Font(name=FN, size=10, bold=True)
    f_title = Font(name=FN, size=14, bold=True)
    f_in = Font(name=FN, size=10, color="0000FF")          # entradas / supuestos escritos a mano
    hdr = PatternFill("solid", fgColor="1F3864"); f_hdr = Font(name=FN, size=10, bold=True, color="FFFFFF")
    fill_ctx, fill_wld = PatternFill("solid", fgColor="DDEBF7"), PatternFill("solid", fgColor="FCE4D6")
    fill_imp, fill_ada = PatternFill("solid", fgColor="E2EFDA"), PatternFill("solid", fgColor="FFF2CC")
    thin = Side(style="thin", color="999999"); box = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap_top = Alignment(wrap_text=True, vertical="top"); center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    wb = Workbook()
    # --- Nodes sheet ---
    NODE_COLS = ["Word", "Type", "Category", "Subtype", "V", "I", "Lv", "Gv", "x", "y", "Col", "Row", "τ", "Source",
                 "Sent+", "Sent−", "Synonyms", "Related (nodes)", "Related (labels without a node)",
                 "Related (WordNet)", "Related (manual)", "Opposites"]
    node_sheets, last_row = {}, {}
    CONN_COLS = ["Word", "Group", "Relation", "Class", "V", "I", "Lv", "Gv", "x", "y", "Col", "Row", "τ",
                 "Current construction", "Current axis", "Proposed axis", "Compare", "Multi-word", "Ambiguous", "Note"]
    for name in ("subjects", "concepts", "connectors"):
        ws = wb.create_sheet(f"{name.capitalize()}_nodes")
        node_sheets[name] = ws
        cols_ = CONN_COLS if name == "connectors" else NODE_COLS
        ws.append(cols_)
        for c in range(1, len(cols_) + 1):
            cell = ws.cell(1, c); cell.font, cell.fill, cell.alignment = f_hdr, hdr, center
        for i, n in enumerate(nodes[name], start=2):
            geo = [f"=(G{i}-H{i})/2", f"=(E{i}-F{i})/2", f"=MAX(1,MIN({N},ROUNDUP(I{i}+5,0)))", f"=MAX(1,MIN({N},ROUNDUP(5-J{i},0)))", f"=MIN(E{i},F{i})"]
            if name == "connectors":
                ws.append([n["word"], n["group"], n["relation"], n["class"], n["V"], n["I"], n["Lv"], n["Gv"], *geo,
                           ", ".join(f"#{c}" for c in n["construction"]), n["current_axis"], n["proposed_axis"], n["compat"],
                           "yes" if n["multiword"] else "", "yes" if n["ambiguous"] else "", n["note"]])
            else:
                ws.append([n["word"], n["type"], n["category"], n["subtype"], n["V"], n["I"], n["Lv"], n["Gv"], *geo,
                           n["source"], n["sent_pos"], n["sent_neg"], ", ".join(n["synonyms"][:8]),
                           ", ".join(n["related"]), ", ".join(n["related_labels"]),
                           ", ".join(f"{w} ({v})" for w, v in n["related_wn"]), ", ".join(n["related_manual"]), ", ".join(n["opposites"])])
            for c in (5, 6, 7, 8):
                ws.cell(i, c).font = f_in           # weights = design decisions (blue)
        last_row[name] = len(nodes[name]) + 1
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if cell.font != f_in:
                    cell.font = f_norm
        widths = [20, 14, 24, 10, 5, 5, 5, 5, 6, 6, 6, 6, 5, 14, 11, 20, 16, 11, 9, 60] if name == "connectors" else \
                 [18, 15, 14, 13, 5, 5, 5, 5, 6, 6, 6, 6, 5, 15, 7, 7, 40, 34, 26, 50, 34, 24]
        for c, w in enumerate(widths, start=1):
            ws.column_dimensions[L(c)].width = w
        ws.freeze_panes = "B2"; ws.auto_filter.ref = f"A1:{L(len(cols_))}{last_row[name]}"

    # --- Grids ---
    def grid_sheet(name, title):
        ws = wb.create_sheet(title)
        cells = defaultdict(list)
        for n in nodes[name]:
            cells[(n["row"], n["col"])].append(n["word"])
        ws["A1"] = f"Table of {title} ({N} × {N})"; ws["A1"].font = f_title
        ws["A2"] = ("Each cell shows how many words it has and the first ones (hand-made concepts first). "
                    f"Full list in the sheet {title}_nodes. Position: x = (Lv − Gv)/2, y = (V − I)/2.")
        ws["A2"].font = f_norm
        ws.merge_cells(start_row=4, start_column=3, end_row=4, end_column=2 + N // 2)
        ws.merge_cells(start_row=4, start_column=3 + N // 2, end_row=4, end_column=2 + N)
        ws["C4"] = "◄ Adaptation (You, structural)"; ws.cell(4, 3 + N // 2).value = "Imposition (I, Sensory) ►"
        for c in (3, 3 + N // 2):
            ws.cell(4, c).font, ws.cell(4, c).alignment = f_b, center
        ws.cell(4, 3).fill, ws.cell(4, 3 + N // 2).fill = fill_ada, fill_imp
        for k in range(1, N + 1):
            for r0 in (5, 5 + N + 4):
                c = ws.cell(r0, 2 + k, k); c.font, c.alignment = f_b, center
        ws["A6"] = "Context\n(We, Possibility)\n▲"; ws[f"A{5 + N}"] = "World\n(That, logical)\n▼"
        for a in ("A6", f"A{5 + N}"):
            ws[a].font, ws[a].alignment = f_b, center
        ws["A6"].fill, ws[f"A{5 + N}"].fill = fill_ctx, fill_wld
        for r in range(1, N + 1):
            ws.cell(5 + r, 2, r).font = f_b; ws.cell(5 + r, 2).alignment = center
            ws.row_dimensions[5 + r].height = 118
            for c in range(1, N + 1):
                ws_words = cells.get((r, c), [])
                cell = ws.cell(5 + r, 2 + c)
                cell.value = (f"{len(ws_words)}\n" + ", ".join(ws_words[:6]) + (f" … +{len(ws_words) - 6}" if len(ws_words) > 6 else "")) if ws_words else None
                cell.font, cell.alignment, cell.border = f_norm, wrap_top, box
        # counts with formulas
        r0 = 5 + N + 3
        ws.cell(r0, 1, f"Count per cell (COUNTIFS formulas over {title}_nodes)").font = f_b
        ns, lr = f"{title}_nodes", last_row[name]
        for r in range(1, N + 1):
            ws.cell(r0 + 1 + r, 2, r).font = f_b
            for c in range(1, N + 1):
                cell = ws.cell(r0 + 1 + r, 2 + c, f"=COUNTIFS({ns}!$K$2:$K${lr},{L(2 + c)}${r0 + 1},{ns}!$L$2:$L${lr},$B{r0 + 1 + r})")
                cell.font, cell.alignment, cell.border = f_norm, center, box
        tot = r0 + N + 2
        ws.cell(tot, 2, "Total").font = f_b
        ws.cell(tot, 3, f"=SUM(C{r0 + 2}:{L(2 + N)}{r0 + 1 + N})").font = f_b
        ws.cell(tot, 4, "nodes in the sheet:").font = f_norm
        ws.cell(tot, 5, f"=COUNTA({ns}!A2:A{lr})").font = f_b
        ws.cell(tot, 6, f'=IF(C{tot}=E{tot},"✔ matches","✖ does not match")').font = f_b
        ws.conditional_formatting.add(f"C{r0 + 2}:{L(2 + N)}{r0 + 1 + N}",
                                      ColorScaleRule(start_type="min", start_color="FFFFFF", end_type="max", end_color="5B9BD5"))
        ws.column_dimensions["A"].width = 17; ws.column_dimensions["B"].width = 6
        for c in range(1, N + 1):
            ws.column_dimensions[L(2 + c)].width = 24
        ws.sheet_view.zoomScale = 85
        return ws

    grid_subjects = grid_sheet("subjects", "Subjects")
    grid_concepts = grid_sheet("concepts", "Concepts")
    grid_connectors = grid_sheet("connectors", "Connectors")

    # --- Readme ---
    ws = wb.active; ws.title = "Readme"
    rows = [
        ("Language Coherence Model — Subjects, Concepts and Connectors tables (v0.3)", f_title),
        ("First pass generated by build_tables.py. The weights are design decisions (blue) and can be edited.", f_norm),
        ("", f_norm),
        ("1. The four axes", f_b),
    ]
    for t, f in rows: ws.append([t]); ws.cell(ws.max_row, 1).font = f
    ws.append(["Pole", "Label", "Pronoun", "Character", "Weight", "What goes there"])
    for c in range(1, 7): ws.cell(ws.max_row, c).font, ws.cell(ws.max_row, c).fill = f_hdr, hdr
    axes = [("Up", "Context", "We", "Possibility", "V", "viable, open, possible, alive, future, good"),
            ("Down", "World", "That", "logical", "I", "dangerous, sad, closed, inert, past, negation"),
            ("Right", "Imposition", "I", "Sensory", "Lv", "raw perception, concrete, body, physical action"),
            ("Left", "Adaptation", "You", "structural", "Gv", "relation, judgment, standard, language, the social")]
    for a in axes: ws.append(list(a))
    for l in [
        "", "2. Scale and position",
        "Each node has 4 independent weights, integers from 0 to 10 (0 = no weight). Happiness, for example, weighs in V, Lv and Gv and nothing in I.",
        "x = (Lv − Gv) / 2   →  x > 0 toward Imposition, x < 0 toward Adaptation.     y = (V − I) / 2   →  y > 0 toward Context, y < 0 toward World.",
        f"Cell ({N} × {N}):  column = ROUNDUP(x + 5),  row = ROUNDUP(5 − y), clipped to 1..{N}.  Row 1 = Context, row {N} = World; column 1 = Adaptation, column {N} = Imposition.",
        "⚠ With even N there is no central cell: x = 0 falls in column 5 and y = 0 in row 5 (Adaptation / Context side). It is a provisional rule.",
        "τ (tension) = MIN(V, I): separates the ambivalent from the neutral, which the centroid collapses to the same point.",
        "", "3. How weights are assigned (inheritance down the tree)",
        "type (biological / non_biological / action / quality) → category → subtype: each level ADDS an adjustment. See the Weight_rules sheet.",
        "Valence: SentiWordNet. V += round(6 × positivity), I += round(6 × negativity). They are independent of each other.",
        "Per-axis lists from Unique Host: sensory adjectives → Imposition, judgment or relation → Adaptation (the hand-written list takes precedence over the generated one).",
        "Hand-made concepts: intrinsic danger, intrinsic emotion and state type adjust V/I/Lv/Gv; they take precedence over the first WordNet sense (ghost = illogical).",
        "People and animals: separated by their WordNet lineage (threat, imposes itself, close bond, cares or teaches; domestic, predator).",
        "Pronouns: we/us up and centered; I on the right; you on the left; that/it down and centered (they reproduce the drawing).",
        "", "4. What each table contains",
    ]:
        ws.append([l]); ws.cell(ws.max_row, 1).font = f_b if l[:2] in ("2.", "3.", "4.", "5.", "6.") else f_norm
    ws.append(["Table", "Nodes", "Contents"])
    for c in range(1, 4): ws.cell(ws.max_row, c).font, ws.cell(ws.max_row, c).fill = f_hdr, hdr
    ws.append(["Subjects", f"=COUNTA(Subjects_nodes!A2:A{last_row['subjects']})", "biological lexicon (people, animals, plants, body, groups) + subject concepts + pronouns"])
    ws.append(["Concepts", f"=COUNTA(Concepts_nodes!A2:A{last_row['concepts']})", "non_biological, action and quality lexicon + hand-made concepts (state, illusory, environment, tone, music)"])
    ws.append(["Connectors", f"=COUNTA(Connectors_nodes!A2:A{last_row['connectors']})", "conjunctions, connective adverbs, transition phrases and punctuation"])
    for l in [
        "", "5. Connectors: time runs vertically",
        "North = future, possibility, condition, purpose and 'we/us' (join, include).  South = past, negation, closure, result, exception.",
        "East–West = present: the horizontal band (V = I, y ≈ 0). West = structure (joins, compares, explains); East = the speaker imposes (emphasis, stance).",
        "Each relation in the document has a base weight (Weight_rules sheet). If a word has two roles, they are two nodes: 'so (result)' and 'so that'; 'though (concessive)' and 'though (sentence-initial)'.",
        "The 'Compare' column of Connectors_nodes contrasts the proposed axis with the one of the Unique Host construction.",
        "", "6. What is NOT there (limits of this pass)",
        "• The 7 concepts of type 'language' (who, which, where…) were left out of Subjects and Concepts: relatives and interrogatives now live in Connectors.",
        "• 261 concepts that are inflections of another word are not separate nodes.",
        "• 'he, she, they, him, her, this, these' are not in any Unique Host pronoun list and were not placed.",
        "• Links (v0.4): 'Related (nodes)' comes from the 597 hand-made concepts; 'Related (WordNet)' is derived from WordNet (is-a, kind-of, part, derived, similar, definition); 'Related (manual)' is a hand-written narrative seed (tools/related_manual.py); 'Opposites' are WordNet antonyms. State labels (hostile, defenseless…) are listed separately because they are not nodes.",
        "• People are still concentrated: the tree has almost no words in its person leaves (237 in total, no sub-leaves).",
        "• SentiWordNet valence is noisy for nouns; the extremes (rows 1 and 10) should be reviewed by hand.",
        "• Only base forms of verbs exist: tense (past = World, future = Context) does not yet move the weight of an action.",
    ]:
        ws.append([l]); ws.cell(ws.max_row, 1).font = f_b if l[:2] == "6." else f_norm
    for row in ws.iter_rows():
        for cell in row:
            if cell.font == Font(): cell.font = f_norm
            cell.alignment = Alignment(wrap_text=False, vertical="top")
    for col, w in zip("ABCDEF", [24, 22, 14, 12, 8, 58]):
        ws.column_dimensions[col].width = w
    ws.column_dimensions["A"].width = 30

    # --- Weight_rules (weight rules sheet) ---
    wr = wb.create_sheet("Weight_rules")
    def table(title, header, rows_):
        wr.append([title]); wr.cell(wr.max_row, 1).font = f_b
        wr.append(header)
        for c in range(1, len(header) + 1): wr.cell(wr.max_row, c).font, wr.cell(wr.max_row, c).fill = f_hdr, hdr
        for r in rows_:
            wr.append(list(r))
            for c in range(1, len(r) + 1):
                wr.cell(wr.max_row, c).font = f_in if isinstance(r[c - 1], int) else f_norm
        wr.append([])
    wr.append(["Default weights (blue = design assumption, editable in build_tables.py). They are ADDED in this order: type → category → subtype → per-axis lists → valence → concept."])
    wr.cell(1, 1).font = f_b; wr.append([])
    table("Level 1 — node type", ["Type", "V", "I", "Lv", "Gv", "Reason"],
          [(k, *v, r) for (k, v), r in zip(TYPE_BASE.items(), ["alive: up and toward Adaptation (NW quadrant)", "inert: down and toward Imposition (SE quadrant)", "sensory: toward Imposition", "structural: toward Adaptation"])])
    table("Level 2 — category (adjustment)", ["Category", "ΔV", "ΔI", "ΔLv", "ΔGv"], [(k, *v) for k, v in CAT_ADJ.items() if v != (0, 0, 0, 0)])
    table("Level 3 — subtype (adjustment)", ["Category", "Subtype", "ΔV", "ΔI", "ΔLv", "ΔGv"], [(k[0], k[1], *v) for k, v in SUB_ADJ.items()])
    table("Hand-made concept — intrinsic emotion (adjustment)", ["Emotion", "ΔV", "ΔI", "ΔLv", "ΔGv"], [(k, *v) for k, v in EMOTION_ADJ.items()])
    table("Hand-made concept — state subtype (adjustment)", ["Subtype", "ΔV", "ΔI", "ΔLv", "ΔGv"], [(k, *v) for k, v in STATUS_ADJ.items()])
    table("Other adjustments", ["Rule", "ΔV", "ΔI", "ΔLv", "ΔGv"],
          [("intrinsic danger (violence, hazard, weapon…)", *DANGER_ADJ), ("has a sensory channel (sight, hearing)", *SENSE_ADJ),
           ("illogical (occurrences)", -2, 4, 1, 0), ("tone (voice, volume)", 0, 0, 4, -2), ("music", 2, 0, 3, 0),
           ("North 'concepts' list (hope, plan, dream…)", 3, -1, 0, 0), ("South 'concepts' list (failure, past, history…)", -1, 3, 0, 0)])
    table("WordNet lineage — people (the first group wins; adjustment)", ["Group (lemma examples)", "ΔV", "ΔI", "ΔLv", "ΔGv"],
          [(", ".join(sorted(g)[:5]) + " …", *a) for g, a in LINEAGE_ADJ])
    table("WordNet lineage — animals (adjustment)", ["Group", "ΔV", "ΔI", "ΔLv", "ΔGv"], [(", ".join(sorted(g)), *a) for g, a in ANIMAL_LINEAGE_ADJ])
    table("Connectors — base weight per relation (V, I, Lv, Gv)", ["Relation", "Group", "V", "I", "Lv", "Gv"],
          [(CD.REL_LABEL[k], CD.GROUP_OF[k], *v) for k, v in CD.REL.items()])
    table("Pronouns (fixed weights)", ["Axis", "Words", "V", "I", "Lv", "Gv"], [(k, ", ".join(w), *W) for k, (w, W) in PRONOUNS.items()])
    for col, w in zip("ABCDEF", [58, 22, 8, 8, 8, 52]): wr.column_dimensions[col].width = w

    wb._sheets = [wb[n] for n in ["Readme", "Subjects", "Concepts", "Connectors", "Subjects_nodes", "Concepts_nodes", "Connectors_nodes", "Weight_rules"]]
    wb.save(f"{OUT}/LCM_tables.xlsx")
    print("written:", f"{OUT}/LCM_tables.xlsx", f"{OUT}/LCM_tables.json")


if __name__ == "__main__" and "--write" in sys.argv:
    write_outputs()
