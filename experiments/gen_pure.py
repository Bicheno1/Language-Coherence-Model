"""PURE generator (experiment): no templates, no lists of verbs/adverbs/adjectives. Only the plane.

What it does (everything is values going in, nothing of a sentence skeleton):
  * STATE: a point (x, y) that starts at the target of the input sentence (the anchors of 'tell me a story', with tense,
    mode and the you->I inversion already applied by lcm.parse) and moves toward each emitted word (moving average).
  * NEXT WORD: drawn (lcm.sampling) among the nodes closest to the reference point, weighted by exp(-(d/sigma)^2).
        \"cercano\" mode  reference = the state                (close association)
        \"opuesto\" mode  reference = the mirror (-x, -y)       (opposite association)
        \"alterno\" mode  close, opposite, close…               (both)
  * The POOL is ALL the nodes: subjects, concepts, connectors and punctuation marks (the '.' lives in the South, the '?' in the North).
  * CLOSURE: cierre=\"puntuacion\": the sentence closes BY ITSELF if the draw lands on a closing mark (., ?, !): it comes from the geometry.
            cierre=\"equilibrio\": in addition, with >= 4 words, if the state returns near the origin (|state| < eps) the '.' is emitted.
  * Only surface \"correction rule\": verbs (type action) are conjugated according to the tense of the TOPIC (past if the topic is
    in the South), the first word is capitalized and punctuation is attached to the previous word.

It is an experiment to SEE what comes out without templates; it is not part of the package and is not tested like the rest.
"""
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lcm import load
from lcm.sampling import Sampler
from lcm.parse import parse
from lcm.tense import tense_of_position
from lcm.walk import target_of

CLOSERS = {".", "?", "!"}


def _pool(t):
    nodes = [("concepts", n) for n in t["concepts"].nodes] + [("subjects", n) for n in t["subjects"].nodes]
    for n in t["connectors"].nodes:
        if not n["multiword"] and (n["group"] != "punctuation" or n["word"] in CLOSERS | {","}):
            nodes.append(("connectors", n))
    return nodes


def _form(node, tense):
    """Surface: only the conjugation of verbs according to tense (a correction rule, not a content one)."""
    w = node["word"]
    if node.get("type") == "action" and tense != "present":
        from lemminflect import getInflection
        if tense == "past":
            return (getInflection(w, "VBD") or (w,))[0]
        return "will " + w
    return w


def generate(t, sentence, mode="near", closing="punctuation", seed=None, temperature=1.0, k=25, sigma=1.5, max_pal=14, eps=0.8, alpha=0.5):
    f = parse(sentence, t)
    obj = target_of(f)
    if obj is None:
        return None, "no anchors"
    tense = tense_of_position(f.topic.node["y"]) if f.topic and f.topic.node else "present"
    randomize = Sampler(seed, temperature, k)
    pool = _pool(t)
    state, used_items, output, trace = list(obj), set(), [], []
    for i in range(max_pal):
        if mode == "near" or (mode == "alternating" and i % 2 == 0):
            ref, tag = (state[0], state[1]), "near"
        else:
            ref, tag = (-state[0], -state[1]), "opposite"
        cands = []
        for tb, n in pool:
            if n["word"] in used_items and n["word"] not in CLOSERS:
                continue
            d = math.hypot(n["x"] - ref[0], n["y"] - ref[1])
            cands.append(((tb, n), math.exp(-(d / sigma) ** 2)))
        (tb, n), p = randomize.choose(cands)
        output.append(n)
        used_items.add(n["word"])
        trace.append(f"{n['word']}[{tag} ({ref[0]:+.1f},{ref[1]:+.1f}) -> ({n['x']:+.1f},{n['y']:+.1f}) p={p:.2f}]")
        if n["word"] in CLOSERS:
            return _surface(output, tense), trace, "closed by itself (the draw landed on a closing mark)", randomize.seed
        state[0] = (1 - alpha) * state[0] + alpha * n["x"]
        state[1] = (1 - alpha) * state[1] + alpha * n["y"]
        if closing == "equilibrium" and len(output) >= 4 and math.hypot(*state) < eps:
            point = t["connectors"].get(".")[0]
            output.append(point)
            trace.append(f".[equilibrium: |state|={math.hypot(*state):.2f} < {eps}]")
            return _surface(output, tense), trace, "closed by equilibrium (|state| < eps)", randomize.seed
    return _surface(output, tense), trace, f"cut off after {max_pal} words (never closed)", randomize.seed


def _surface(nodes, tense):
    out = ""
    for i, n in enumerate(nodes):
        w = _form(n, tense)
        if i == 0:
            w = w[:1].upper() + w[1:]
        out += (w if n["word"] in CLOSERS | {","} else (" " if out else "") + w)
    return out


if __name__ == "__main__":
    T = load()
    sentence = sys.argv[1] if len(sys.argv) > 1 else "TELL ME A STORY"
    for mode in ("near", "opposite", "alternating"):
        for closing in ("punctuation", "equilibrium"):
            print(f"\n=== mode {mode} · closure {closing} ===")
            for s in range(1, 5):
                txt, trace, reason, seed_i = generate(T, sentence, mode, closing, seed=s)
                print(f"  [{s}] {txt}    ({reason})")
