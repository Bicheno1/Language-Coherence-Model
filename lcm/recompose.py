"""Recomposer: the second half of the decompose/recompose pair. It recomposes the text from the ROLES, by value rules.

Difference from `lcm.narrate` (which is a template with blanks): here NO word is written inside a sentence. Each word comes
from (a) a node of the tables, (b) one of the 3 provisional functional nodes (lcm.function_words: there, a, the), or (c) conjugating a node
(morphology). What is written are RULES that act on values:

  R1 TENSE        the communication topic (story) takes the tense its height marks: South = past. (Only communication topics:
                  the height of a concrete noun is possibility, not tense.) Verbs are conjugated in that tense.
  R2 PROTAGONIST  the living being that the topic opens by association (related) and closeness; if the request is generic, a draw (lcm.sampling).
  R3 INTRODUCTION what is NEW is introduced with the existential rule: [there][be in R1][determiner by value][being][relative by
                  animacy][action]. The determiner is the functional with the highest y if it is new (a) and the lowest y if it already exists (the).
  R4 DESCENT      each following action is the one associated with the protagonist whose height drops ~1 relative to the previous one. The sentence
                  starts up (Context) and goes down to the South (World), as the corpus showed (first word y=0.84; last 0.35).
  R5 CLOSURE      the walk CLOSES when an action falls in the South (y <= y_close) or there is no action left that descends: the
                  value decides, not a number of sentences.
  R6 CONNECTOR    from the Connectors table, the one whose height is closest to the action; the closing ones (finally, lastly) only for the
                  clause that closes (R5), the succession ones (then, next, later) for the others. Without repeating.
  R7 PUNCTUATION  comma after the connector if, in the corpus, it takes one more than half the time (data/connector_punctuation.json:
                  however yes, then/next/finally no). Each clause ends in the '.' node of the tables (South).

What it is NOT yet: there are still three hand-written lists that come from the manual seed (who can do what, how, which
protagonists) and two gender sets; the functional words are provisional; it only recomposes communication topics (stories).
"""
import json
import math
import os
from dataclasses import dataclass, field

from .sampling import Sampler
from .decompose import decompose
from .function_words import determiner, node as function_node
from .narrate import (MALE_WORDS, FEMALE_WORDS, SECOND_HOP, _closeness, _choose_adverb, _choose_protagonist, _form, _verbs_of, _protagonist_verbs,
                      EXCLUDED_CONNECTORS)
from .walk import target_of

Y_START = 3.0                    # the descent starts here: the first action looks for y ~ 2
Y_STEP = 1.0                      # the minimum amount each following action descends
Y_CLOSE = -1.0                   # an action in the South closes the walk
MIN_DROP = 0.25                 # an action only counts as "next" if it descends at least this much
MAX_CLAUSES = 6
COMMA_THRESHOLD = 0.5                 # comma if the corpus puts it half the time or more
MIDDLE_RELATION, CLOSING_RELATION = "succession (adverb)", "sequence (closes)"
RELATIVE_WORDS = {"animate": "who", "inanimate": "which"}   # R3: relatives are REL nodes in the table; which one is decided by animacy

_PUNCTUATION_DATA = None


def _punctuation():
    global _PUNCTUATION_DATA
    if _PUNCTUATION_DATA is None:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "connector_punctuation.json")
        try:
            _PUNCTUATION_DATA = json.load(open(filepath, encoding="utf-8"))["connectors"]
        except OSError:
            _PUNCTUATION_DATA = {}
    return _PUNCTUATION_DATA


@dataclass
class Piece:
    text: str
    rule: str
    node: dict = None


@dataclass
class Recomposition:
    input_text: str = ""
    clauses: list = field(default_factory=list)       # list of lists of Piece (pieces)
    target: tuple = (0.0, 0.0)
    tense: str = None
    closed_by: str = None
    seed: int = None
    error: str = None

    @property
    def pieces(self):
        return [p for c in self.clauses for p in c]

    @property
    def text(self):
        out = []
        for c in self.clauses:
            s = ""
            for p in c:
                s += p.text if p.text in {".", ","} else (" " if s else "") + p.text
            out.append(s[:1].upper() + s[1:])
        return " ".join(out)

    @property
    def coherence(self):
        ns = [p.node for p in self.pieces if p.node is not None and "x" in p.node]
        if not ns:
            return 0.0
        return round(1.0 - min(1.0, sum(math.hypot(n["x"] - self.target[0], n["y"] - self.target[1]) for n in ns) / len(ns) / (5 * math.sqrt(2))), 3)


def _pool(tables, prot):
    proper, inherited = _protagonist_verbs(tables, prot)
    pool = {n["word"]: (n, w) for n, w in proper}
    if not inherited:
        for n0, w0 in list(pool.values()):
            for n, w in _verbs_of(tables, n0):
                if n["word"] not in pool or w0 * w * SECOND_HOP > pool[n["word"]][1]:
                    pool[n["word"]] = (n, w0 * w * SECOND_HOP)
    return pool, inherited


def _next_action(pool, used_items, y_prev, target, first):
    """R4: the associated action whose height comes closest to (y_prev - Y_STEP); from the 2nd on it must really descend."""
    y_target = y_prev - Y_STEP
    cands = []
    for n, w in pool.values():
        if n["word"] in used_items or (not first and n["y"] > y_prev - MIN_DROP):
            continue
        cands.append((w * max(0.0, 1.0 - abs(n["y"] - y_target) / 6.0) * _closeness(n, target), n))
    if not cands:
        return None, 0.0
    cands.sort(key=lambda t: (-t[0], t[1]["word"]))
    return cands[0][1], cands[0][0]


def _connector(tables, y, is_closing, used_items, draw):
    """R6: node of the Connectors table whose height is closest to `y`; the set is fixed by R5 (does this clause close?)."""
    rel = CLOSING_RELATION if is_closing else MIDDLE_RELATION
    all_items = [n for n in tables["connectors"].nodes if n["relation"] == rel and " " not in n["word"] and n["word"] not in EXCLUDED_CONNECTORS]
    free_calls = [n for n in all_items if n["word"] not in used_items] or all_items
    m = min(abs(n["y"] - y) for n in free_calls)
    tied = [n for n in free_calls if abs(abs(n["y"] - y) - m) < 1e-9]
    return (draw.choose([(n, 1.0) for n in tied])[0] if len(tied) > 1 else tied[0]), rel


def _pronoun(tables, prot):
    w = "he" if prot["word"] in MALE_WORDS else "she" if prot["word"] in FEMALE_WORDS else "it"
    hit = tables["subjects"].get(w)
    return hit[0] if hit else None


def recompose(tables, frame, seed=None, temperature=1.0, top_k=None, randomize=False, y_close=Y_CLOSE, max_clauses=MAX_CLAUSES):
    r = Recomposition(input_text=frame.input_text)
    cards = {f.role_obj: f for f in decompose(tables, frame)}
    te = cards.get("topic")
    if te is None or te.node is None:
        r.error = "there is no topic anchored in the tables: there is nothing to recompose"
        return r
    if te.node.get("category") != "communication":
        r.error = f"I only recompose communication topics (a story); '{te.word}' is {te.node.get('category')}"
        return r
    r.tense = te.tense_by_height                                        # R1
    r.target = target_of(frame)
    draw = Sampler(seed, temperature, top_k)
    prot, tb, quals, tz = _choose_protagonist(tables, frame, r.target, 0, draw, randomize)    # R2
    if prot is None:
        r.error = tz[0]
        return r
    pool, inherited = _pool(tables, prot)
    animate = prot.get("type") == "biological"
    pron = _pronoun(tables, prot)

    used_items, used_connectors, used_adverbs = set(), set(), set()
    y_prev, k = Y_START, 0
    while k < max_clauses:
        v, sc = _next_action(pool, used_items, y_prev, r.target, first=(k == 0))
        if v is None:
            r.closed_by = "no action that descends is left: the walk closes" if k else None
            break
        used_items.add(v["word"])
        closes = v["y"] <= y_close
        adv, _ = _choose_adverb(tables, v, r.target, used_adverbs)
        if adv:
            used_adverbs.add(adv["word"])
        clause_list = []
        rule_v = f"R4 descent: y={v['y']:+.1f} (target {y_prev - Y_STEP:+.1f})" + (f", inherited from {inherited}" if inherited else "")
        form = _form(v["word"].lower(), r.tense)                  # R1
        if k == 0:
            there = function_node("there")
            cop = tables["concepts"].get("be")[0]
            clause_list.append(Piece("there", "R3 existential: something new enters the world; World functional", there))
            clause_list.append(Piece(_form("be", r.tense), f"R1 tense {r.tense} (y of '{te.word}'={te.node['y']:+.1f}) on the node 'be'", cop))
            q = " ".join(quals)
            det, ndet = determiner(True, q or prot["word"])
            clause_list.append(Piece(det, "R3 determiner by value: new = higher y (possibility)", ndet))
            for qq in quals:
                clause_list.append(Piece(qq, "quality requested by the user"))
            clause_list.append(Piece(prot["word"], "R2 protagonist: " + tz[0].replace("protagonist: ", "")[:110], prot))
            rel = tables["connectors"].get(RELATIVE_WORDS["animate" if animate else "inanimate"])[0]
            clause_list.append(Piece(rel["word"], f"R3 relative by animacy ({'animate' if animate else 'inanimate'}): REL node", rel))
        else:
            with_conn, relc = _connector(tables, v["y"], closes, used_connectors, draw)
            used_connectors.add(with_conn["word"])
            clause_list.append(Piece(with_conn["word"], f"R6 connector '{relc}' with y closest to the action ({with_conn['y']:+.1f} ~ {v['y']:+.1f})", with_conn))
            p = _punctuation().get(with_conn["word"], {}).get("comma_after", 0.0)
            if p >= COMMA_THRESHOLD:
                clause_list.append(Piece(",", f"R7 comma: in the corpus {with_conn['word']} takes it {p:.0%} of the time"))
            else:
                clause_list[-1].rule += f"; no comma (the corpus puts it {p:.0%})"
            clause_list.append(Piece(pron["word"], "pronoun by animacy and gender", pron))
        clause_list.append(Piece(form, rule_v + f"; R1 conjugated in {r.tense}", v))
        if adv:
            clause_list.append(Piece(adv["word"], f"adverb associated with '{v['word']}' (manual seed), the most typical one not yet used", adv))
        point = tables["connectors"].get(".")[0]
        clause_list.append(Piece(".", f"R7 period: '.' node of the tables (South, y={point['y']:+.1f}) closes the clause", point))
        r.clauses.append(clause_list)
        y_prev = v["y"]
        k += 1
        if closes:
            r.closed_by = f"R5: the action '{v['word']}' fell in the South (y={v['y']:+.1f} <= {y_close:+.1f})"
            break
    else:
        r.closed_by = f"the maximum of {max_clauses} clauses was reached without getting to the South"
    r.seed = draw.seed if draw.used else None
    if not r.clauses:
        r.error = f"'{prot['word']}' has no associated actions"
    return r
