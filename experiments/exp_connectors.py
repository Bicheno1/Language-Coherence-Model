"""Experiment v0.4f-c: where do connectors go and when do they take a comma? (CORRECTION rules, measured on real text)

For each one-word connector in the Connectors table, the complete Brown corpus (57,000 sentences, all
genres) is used to count: how many times it appears, what fraction opens the sentence, what fraction takes a comma AFTER it and what fraction a comma BEFORE.
It is aggregated by table CLASS (SC subordinating, CC coordinating, CA conjunctive adverb, TP transition, PREP…).
It is the empirical basis for deciding the placement and punctuation rules; it decides nothing by itself.
Usage:  python experiments/exp_connectors.py     (writes experiments/results_connectors.json)
"""
import json, os, sys
from collections import defaultdict
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lcm import load
from nltk.corpus import brown

T = load()
connectors_by_word = {n["word"].lower(): n for n in T["connectors"].nodes if not n["multiword"] and n["class"] not in ("PUNCT",)}
cont = defaultdict(lambda: {"n": 0, "start": 0, "comma_after": 0, "comma_before": 0, "final": 0, "middle": 0})
for s in brown.sents():
    toks = [w.lower() for w in s]
    # word indices (without punctuation) to tell whether it opens the sentence
    for i, w in enumerate(toks):
        if w not in connectors_by_word:
            continue
        c = cont[w]
        c["n"] += 1
        prev = toks[i - 1] if i else None
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        opens = i == 0 or all(not t.isalpha() for t in toks[:i])
        c["start"] += opens
        c["comma_after"] += nxt == ","
        c["comma_before"] += prev == ","
        c["final"] += nxt in (".", "?", "!", None)
        c["middle"] += (not opens) and nxt not in (".", "?", "!", None)

by_class = defaultdict(lambda: defaultdict(int))
by_class_words = defaultdict(list)
for w, c in cont.items():
    cl = connectors_by_word[w]["class"]
    for k, v in c.items():
        by_class[cl][k] += v
    by_class_words[cl].append((c["n"], w))
res = {}
for cl, c in by_class.items():
    n = c["n"]
    res[cl] = {"occurrences": n, "pct_opens_sentence": round(100 * c["start"] / n, 1), "pct_comma_after": round(100 * c["comma_after"] / n, 1),
               "pct_comma_before": round(100 * c["comma_before"] / n, 1), "pct_sentence_final": round(100 * c["final"] / n, 1),
               "most_frequent": [w for _, w in sorted(by_class_words[cl], reverse=True)[:8]]}
detail = {w: {**c, "class": connectors_by_word[w]["class"], "group": connectors_by_word[w]["group"], "relation": connectors_by_word[w]["relation"]} for w, c in cont.items() if c["n"] >= 30}
json.dump({"by_class": res, "detail": detail}, open(os.path.join(ROOT, "experiments", "results_connectors.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
