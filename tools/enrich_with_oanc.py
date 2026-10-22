# -*- coding: utf-8 -*-
"""
Connects the training bases (pyramids.json, general_measures.json) with the generator's tables.

  input:   data/LCM_tables.json   +  pyramids.json   (+ general_measures.json, optional)
  output:  data/LCM_tables_oanc.json   (copy of the tables with two new fields per node)
           data/oanc_report.csv       (to REVIEW: what the corpus supports and what it proposes)

New fields in each node (they do not touch any of the existing ones):
  related_oanc  [[word, score], ...]   words that are NODES and that the corpus associates with this one (higher = more associated)
  verbs_oanc   [[verb, score], ...]   only those that are actions (concepts table, type 'action')
score = tree value x idf, where idf = ln(N / (1 + trees)) with 'trees' from general_measures.json.
Only words that are NODES of the tables enter (the, of, a... are not) and very general ones weigh little because of the idf.

The generator does NOT use these fields to choose verbs for a story (that choice belongs only to the manual seed, on purpose).
The flow is: review oanc_report.csv -> move by hand whatever is useful into related_manual (tools/related_manual.py).

    python tools/enrich_with_oanc.py --pyramids pyramids.json --general general_measures.json
"""
import argparse, collections, csv, json, math, os
import ijson

def _num(v):
    if isinstance(v, bool): return None
    if isinstance(v, (int, float)): return v
    if isinstance(v, (list, tuple)): return next((r for r in (_num(x) for x in v) if r is not None), None)
    if isinstance(v, dict):
        for k in ("trees", "arboles", "árboles", "n_trees"):
            if k in v: return _num(v[k])
        return next((r for r in (_num(x) for x in v.values()) if r is not None), None)
    return None

def read_trees(path):
    """general_measures.json = {\"meta\": {\"fields\": [...]}, \"<block>\": {word: [trees, definitions, confirmed, confirmation]}}"""
    gm = json.load(open(path, encoding="utf-8"))
    fields = (gm.get("meta") or {}).get("fields") or ["trees"]; idx = fields.index("trees") if "trees" in fields else 0
    blocks = {k: v for k, v in gm.items() if k != "meta" and isinstance(v, dict)}
    words = max(blocks.values(), key=len) if blocks else gm            # the block with the most entries is the words
    out = {}
    for w, v in words.items():
        out[w] = (v[idx] if isinstance(v, (list, tuple)) and len(v) > idx else _num(v)) or 0
    print(f"general_measures: {len(out):,} words (fields: {fields}).  example trees:", {w: out[w] for w in ("dog", "animal", "man", "the", "of") if w in out})
    return out, max(len(out), 1)

def detect_prefix(path):
    with open(path, "rb") as f:
        for pref, ev, val in ijson.parse(f):
            if ev == "map_key" and val == "pyramid":
                return pref.rsplit(".", 1)[0] if "." in pref else ""
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-path", default=os.path.join("data", "LCM_tables.json"))
    ap.add_argument("--pyramids", required=True); ap.add_argument("--general", default=None)
    ap.add_argument("--output", default=os.path.join("data", "LCM_tables_oanc.json"))
    ap.add_argument("--report", default=os.path.join("data", "oanc_report.csv"))
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--k-verbs", type=int, default=10)
    ap.add_argument("--idf-min", type=float, default=0.5, help="discards words that are too general; raise the value to be stricter")
    ap.add_argument("--min-pins", type=int, default=2)
    a = ap.parse_args()
    d = json.load(open(a.tables_path, encoding="utf-8"))
    nodes = collections.defaultdict(list)                                   # word -> [(table, node)]
    for tb, t in d["tables"].items():
        for n in t["nodes"]: nodes[n["word"].lower()].append((tb, n))
    connector_words = {n["word"].lower() for n in d["tables"]["connectors"]["nodes"]}
    is_action = lambda w: any(tb == "concepts" and n.get("type") == "action" for tb, n in nodes.get(w, []))
    trees, N = {}, 0
    if a.general:
        trees, N = read_trees(a.general)
    idf = lambda w: max(0.0, math.log(N / (1 + trees.get(w, 0)))) if N else 1.0
    pref = detect_prefix(a.pyramids); print(f"words at the level: '{pref or '(root)'}'   nodes to enrich: {len(nodes):,}")
    facts = found_words = 0; arboles = {}
    with open(a.pyramids, "rb") as f:
        for word, info in ijson.kvitems(f, pref, use_float=True):
            w = word.lower()
            if w not in nodes: continue
            found_words += 1; cands = {}
            for e in info.get("pyramid", []):
                t, v, pins = e[0].lower(), e[1], (e[2] if len(e) > 2 else 0)
                if t == w or t not in nodes or t in connector_words or pins < a.min_pins: continue
                i = idf(t)
                if i < a.idf_min: continue
                cands[t] = max(cands.get(t, 0.0), v * i)
            arboles[w] = sorted(cands.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"words of the tables found in pyramids.json: {found_words:,} de {len(nodes):,}")
    row = []
    for tb, t in d["tables"].items():
        for n in t["nodes"]:
            lst = arboles.get(n["word"].lower())
            if lst is None: continue
            n["related_oanc"] = [[w, round(s, 4)] for w, s in lst[:a.k]]
            vs = [(w, s) for w, s in lst if is_action(w)]
            n["verbs_oanc"] = [[w, round(s, 4)] for w, s in vs[:a.k_verbs]]
            if n.get("related_manual"):
                man = n["related_manual"]; score = dict(lst)
                sin = [m for m in man if m not in score]
                new_items = [w for w, _ in vs[:a.k_verbs] if w not in man]
                narrative = [w for w in new_items if any(nn.get("narrative_verb") for _, nn in nodes[w])]
                row.append([n["word"], tb, " ".join(man), " ".join(m for m in man if m in score),
                             " ".join(sin), " ".join(new_items), " ".join(narrative)])
            facts += 1
    d["oanc"] = {"pyramids": os.path.basename(a.pyramids), "k": a.k, "idf_min": a.idf_min, "min_pins": a.min_pins}
    json.dump(d, open(a.output, "w", encoding="utf-8"), ensure_ascii=False)
    with open(a.report, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["word", "table", "manual_seed", "manual_with_corpus_support", "manual_NO_support", "verbs_proposed_by_corpus_not_in_tables", "of_those_already_narrative"])
        w.writerows(row)
    print(f"nodes enriched: {facts:,}  ->  {a.output}\nnodes with a manual seed compared: {len(row)}  ->  {a.report}")

if __name__ == "__main__":
    main()
