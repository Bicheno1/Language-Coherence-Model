"""Experiment v0.4f: do the LCM tables encode association between neighboring words, and does closure by equilibrium appear?

Two questions, with REAL text (Brown corpus, narrative genres), without generating anything:

  Q1 ASSOCIATION. If two consecutive words of a real sentence are closer in the plane than the same words
     in another order (or than two random words with the same concentration), the geometry encodes something of syntax/usage.
     If not, closeness is no use for choosing \"the next word\" and the weights must be recalibrated before generating.

  Q2 CLOSURE. Project hypothesis: a sentence closes when its accumulated state returns near an equilibrium point
     (Imposition ~ Adaptation, x ~ 0) and a paragraph closes when the state crosses to the South (y < 0). It is measured whether real text
     shows it: |x| of the accumulated state along the sentence, and y of the last sentence versus the first of each paragraph.

Controls: (a) reordering NULL: the same words of the same sentence in random order (keeps vocabulary and
concentration, breaks order); (b) NULL of random pairs with the corpus word distribution. Confidence intervals
by bootstrap over sentences (fixed seed). Only words that are nodes count: the coverage is reported.

Usage:  python experiments/exp_geometry.py        (a few minutes; writes experiments/results_geometry.json)
"""
import json
import math
import os
import random
import sys
from collections import defaultdict
from functools import lru_cache

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lcm import load
from lemminflect import getLemma

T = load()
GENDERS = ["fiction", "adventure", "romance", "mystery", "science_fiction", "humor"]
DEFAULT_SEED = 20261003
RNG = random.Random(DEFAULT_SEED)
UPOS_TO_LEMMA = {"NOUN": "NOUN", "VERB": "VERB", "ADJ": "ADJ", "ADV": "ADV"}      # PRON, DET, etc.: only the surface form


@lru_cache(maxsize=None)
def anchor_all(word, upos):
    """Table node for a tagged word, or None. Same preferences as lcm.parse: a noun is not an
    action, a verb prefers the action; if the singular only exists as a verb the plural is tried (duck -> ducks)."""
    w = word.lower()
    if not w.isalpha():
        return None
    keys = [w]
    if upos in UPOS_TO_LEMMA:
        r = getLemma(w, UPOS_TO_LEMMA[upos])
        lem = r[0].lower() if r else w
        keys = [lem, w] if upos in ("NOUN", "VERB", "ADJ") else [w, lem]
    order = ("connectors", "subjects", "concepts") if upos in ("CONJ", "ADP", "PRT") else \
            ("subjects", "concepts", "connectors") if upos == "PRON" else ("concepts", "subjects", "connectors")
    for k in keys:
        hits = [h for tb in order for h in T[tb].get(k)]
        if not hits:
            continue
        if upos == "NOUN":
            no_acc = [h for h in hits if h.get("type") != "action"]
            if no_acc:
                return no_acc[0]
        elif upos == "VERB":
            acc = [h for h in hits if h.get("type") == "action"]
            return (acc or hits)[0]
        else:
            return hits[0]
    if upos == "NOUN" and w.isalpha():
        from lemminflect import getInflection
        pl = (getInflection(w, "NNS") or (None,))[0]
        for tb in ("subjects", "concepts"):
            for h in (T[tb].get(pl) if pl else []):
                if h.get("type") != "action":
                    return h
    return None


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def ic(values_list, n_boot=2000):
    """(mean, ci95_low, ci95_high) by bootstrap over the list (fixed seed)."""
    r = random.Random(DEFAULT_SEED + 1)
    n = len(values_list)
    if n == 0:
        return (float("nan"),) * 3
    m = sum(values_list) / n
    bs = sorted(sum(values_list[r.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return m, bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]


def load_corpus():
    from nltk.corpus import brown
    sents = []
    for cat in GENDERS:
        for s in brown.tagged_sents(categories=[cat], tagset="universal"):
            toks = [(w, t) for w, t in s]
            pal = [w for w, t in toks if t != "."]
            if 5 <= len(pal) <= 14 and all(w.isalpha() or w in ",.;:'-" or w.replace("'", "").isalpha() for w, _ in toks):
                sents.append([(w, t, anchor_all(w, t)) for w, t in toks if t != "."])
    return sents


def load_paragraphs():
    """Paragraphs as lists of sentences, each sentence = list of anchored nodes (in order)."""
    from nltk.corpus import brown
    out = []
    for cat in GENDERS:
        tagged = brown.tagged_paras(categories=[cat], tagset="universal")
        for paragraph_tokens in tagged:
            sents_list = []
            for s in paragraph_tokens:
                nodes = [anchor_all(w, t) for w, t in s if t != "."]
                nodes = [n for n in nodes if n is not None]
                if len(nodes) >= 3:
                    sents_list.append(nodes)
            if len(sents_list) >= 3:
                out.append(sents_list)
    return out


def p1_association(sents):
    """Distance between consecutive neighbors (anchored only) versus nulls."""
    real, shuf, real_pair_counts, shuf_pair_counts = [], [], [0, 0], [0, 0]
    by_real_sentence, by_shuffled_sentence = [], []
    pool = [n for s in sents for _, _, n in s if n is not None]
    randomize = []
    by_type = defaultdict(lambda: [[], []])
    for s in sents:
        an = [(t, n) for _, t, n in s if n is not None]
        if len(an) < 3:
            continue
        d_r = [dist(a[1], b[1]) for a, b in zip(an, an[1:])]
        shuffled = an[:]
        RNG.shuffle(shuffled)
        d_s = [dist(a[1], b[1]) for a, b in zip(shuffled, shuffled[1:])]
        by_real_sentence.append(sum(d_r) / len(d_r))
        by_shuffled_sentence.append(sum(d_s) / len(d_s))
        real_pair_counts[0] += sum(1 for a, b in zip(an, an[1:]) if (a[1]["row"], a[1]["col"]) == (b[1]["row"], b[1]["col"]))
        real_pair_counts[1] += len(d_r)
        shuf_pair_counts[0] += sum(1 for a, b in zip(shuffled, shuffled[1:]) if (a[1]["row"], a[1]["col"]) == (b[1]["row"], b[1]["col"]))
        shuf_pair_counts[1] += len(d_s)
        for a, b in zip(an, an[1:]):
            by_type[f"{a[0]}->{b[0]}"][0].append(dist(a[1], b[1]))
        for a, b in zip(shuffled, shuffled[1:]):
            by_type[f"{a[0]}->{b[0]}"][1].append(dist(a[1], b[1]))
        randomize.append(dist(RNG.choice(pool), RNG.choice(pool)))
    dif = [r - s_ for r, s_ in zip(by_real_sentence, by_shuffled_sentence)]
    res = {
        "sentences": len(by_real_sentence),
        "real_neighbor_distance": ic(by_real_sentence),
        "reordered_neighbor_distance": ic(by_shuffled_sentence),
        "real_minus_reordered": ic(dif),
        "sentences_where_real_is_smaller": round(sum(1 for d in dif if d < 0) / len(dif), 3),
        "random_pair_distance": ic(randomize),
        "same_cell_real": round(real_pair_counts[0] / real_pair_counts[1], 3),
        "same_cell_reordered": round(shuf_pair_counts[0] / shuf_pair_counts[1], 3),
        "by_pair_type": {k: {"n": len(v[0]), "real": round(sum(v[0]) / len(v[0]), 3), "reordered": round(sum(v[1]) / len(v[1]), 3)}
                            for k, v in sorted(by_type.items(), key=lambda kv: -len(kv[1][0])) if len(v[0]) >= 400},
    }
    return res


def p2_closure(sents, paragraphs):
    out = {}
    # (a) along the sentence: |x| of the accumulated state (centroid) in quartiles, and x,y of the first and last anchored node
    quartiles = {q: [] for q in (0.25, 0.5, 0.75, 1.0)}
    first_x, last_x, first_y, last_y = [], [], [], []
    for s in sents:
        an = [n for _, _, n in s if n is not None]
        if len(an) < 4:
            continue
        for q in quartiles:
            k = max(1, round(q * len(an)))
            quartiles[q].append(abs(sum(n["x"] for n in an[:k]) / k))
        first_x.append(abs(an[0]["x"])); last_x.append(abs(an[-1]["x"]))
        first_y.append(an[0]["y"]); last_y.append(an[-1]["y"])
    out["sentence_abs_x_accumulated_state"] = {str(q): ic(v) for q, v in quartiles.items()}
    out["sentence_abs_x_first_vs_last"] = {"first": ic(first_x), "last": ic(last_x),
                                              "diff_last_minus_first": ic([b - a for a, b in zip(first_x, last_x)])}
    out["sentence_y_first_vs_last"] = {"first": ic(first_y), "last": ic(last_y),
                                          "diff_last_minus_first": ic([b - a for a, b in zip(first_y, last_y)])}
    out["sentence_n"] = len(first_x)
    # (b) paragraph: centroid of each sentence; first vs. last
    py_p, py_u, px_p, px_u, ds = [], [], [], [], []
    for sents_list in paragraphs:
        centroids = [(sum(n["x"] for n in o) / len(o), sum(n["y"] for n in o) / len(o)) for o in sents_list]
        px_p.append(abs(centroids[0][0])); px_u.append(abs(centroids[-1][0]))
        py_p.append(centroids[0][1]); py_u.append(centroids[-1][1])
    out["paragraph_n"] = len(py_p)
    out["paragraph_abs_x_first_vs_last"] = {"first": ic(px_p), "last": ic(px_u),
                                              "diff_last_minus_first": ic([b - a for a, b in zip(px_p, px_u)])}
    out["paragraph_y_first_vs_last"] = {"first": ic(py_p), "last": ic(py_u),
                                          "diff_last_minus_first": ic([b - a for a, b in zip(py_p, py_u)])}
    return out


def coverage():
    from nltk.corpus import brown
    tot, anc, by_pos = 0, 0, defaultdict(lambda: [0, 0])
    for s in brown.tagged_sents(categories=GENDERS, tagset="universal"):
        for w, t in s:
            if t == "." or not w.isalpha():
                continue
            tot += 1
            n = anchor_all(w, t)
            by_pos[t][1] += 1
            if n is not None:
                anc += 1
                by_pos[t][0] += 1
    return {"words": tot, "anchored": anc, "pct": round(100 * anc / tot, 1),
            "by_category": {t: round(100 * a / b, 1) for t, (a, b) in sorted(by_pos.items(), key=lambda kv: -kv[1][1])}}


if __name__ == "__main__":
    print("loading sentences…", flush=True)
    sents = load_corpus()
    print(f"{len(sents)} simple sentences (5-14 words). Coverage:", flush=True)
    cov_info = coverage()
    print(" ", cov_info, flush=True)
    r1 = p1_association(sents)
    print("P1:", json.dumps(r1, indent=1, ensure_ascii=False), flush=True)
    paras = load_paragraphs()
    r2 = p2_closure(sents, paras)
    print("P2:", json.dumps(r2, indent=1, ensure_ascii=False), flush=True)
    with open(os.path.join(ROOT, "experiments", "results_geometry.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": DEFAULT_SEED, "genders": GENDERS, "coverage": cov_info, "P1_association": r1, "P2_closure": r2}, f, indent=1, ensure_ascii=False)


# ── Second pass (v0.4f-b): missing controls and the OPPOSITE association ────────────────────────────────
def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va and vb else float("nan")


def p1b_correlation_y_opposite(sents):
    """(1) correlation between the x (and the y) of consecutive words: >0 close association, <0 opposite, 0 none.
    (2) distance of the next word to the MIRROR (-x, -y) of the previous one: smaller than the reordered one's = opposite association."""
    real = {"x": ([], []), "y": ([], [])}
    shuffled = {"x": ([], []), "y": ([], [])}
    d_mir_real, d_mir_shuffled = [], []
    for s in sents:
        an = [n for _, _, n in s if n is not None]
        if len(an) < 3:
            continue
        m = an[:]
        RNG.shuffle(m)
        for seq, dest, mir in ((an, real, d_mir_real), (m, shuffled, d_mir_shuffled)):
            for a, b in zip(seq, seq[1:]):
                dest["x"][0].append(a["x"]); dest["x"][1].append(b["x"])
                dest["y"][0].append(a["y"]); dest["y"][1].append(b["y"])
                mir.append(math.hypot(b["x"] + a["x"], b["y"] + a["y"]))
    out = {"pairs": len(d_mir_real)}
    for axis in ("x", "y"):
        out[f"correlation_{axis}"] = {"real": round(_pearson(*real[axis]), 4), "reordered": round(_pearson(*shuffled[axis]), 4)}
    out["distance_to_mirror"] = {"real": round(sum(d_mir_real) / len(d_mir_real), 4), "reordered": round(sum(d_mir_shuffled) / len(d_mir_shuffled), 4)}
    # CI of the real correlation by pair bootstrap
    r = random.Random(DEFAULT_SEED + 2)
    xs, ys_ = real["x"], real["y"]
    n = len(xs[0])
    bs = {"x": [], "y": []}
    for _ in range(300):
        idx = [r.randrange(n) for _ in range(n)]
        bs["x"].append(_pearson([xs[0][i] for i in idx], [xs[1][i] for i in idx]))
        bs["y"].append(_pearson([ys_[0][i] for i in idx], [ys_[1][i] for i in idx]))
    for axis in ("x", "y"):
        v = sorted(bs[axis])
        out[f"correlation_{axis}"]["ic95_real"] = [round(v[int(0.025 * 300)], 4), round(v[int(0.975 * 300)], 4)]
    return out


def p2_controls(sents):
    """Same P2 calculation over the SAME words in random order: what comes out by simple averaging, without order."""
    quartile_bins = {0.25: [], 0.5: [], 0.75: [], 1.0: []}
    dif_y, dif_x = [], []
    for s in sents:
        an = [n for _, _, n in s if n is not None]
        if len(an) < 4:
            continue
        m = an[:]
        RNG.shuffle(m)
        for q in quartile_bins:
            k = max(1, round(q * len(m)))
            quartile_bins[q].append(abs(sum(n["x"] for n in m[:k]) / k))
        dif_y.append(m[-1]["y"] - m[0]["y"])
        dif_x.append(abs(m[-1]["x"]) - abs(m[0]["x"]))
    return {"abs_x_accumulated_state_reordered": {str(q): round(sum(v) / len(v), 3) for q, v in quartile_bins.items()},
            "y_last_minus_first_reordered": ic(dif_y), "abs_x_last_minus_first_reordered": ic(dif_x)}


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "controls":
    pass
