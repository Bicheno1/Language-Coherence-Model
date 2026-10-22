"""Loading and querying of the tables (data/LCM_tables.json)."""
import json
import os
from collections import defaultdict

from .geometry import position, cell, tension, axis_of, N

_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "LCM_tables.json")


def _keys(word):
    """Lookup keys of a node: the full word and, if it carries a role in parentheses, the simple form."""
    w = word.lower()
    out = {w}
    if " (" in w:
        out.add(w.split(" (")[0])
    return out


class Table:
    """An n x n table: nodes with weights, position and cell."""

    def __init__(self, name, nodes, n=N):
        self.name, self.n = name, n
        self.nodes = nodes
        self._index = defaultdict(list)
        self._cells = defaultdict(list)
        for node in nodes:
            for k in _keys(node["word"]):
                self._index[k].append(node)
            self._cells[(node["row"], node["col"])].append(node)

    def __len__(self):
        return len(self.nodes)

    def get(self, word):
        """Nodes with that word (there can be several if it has several roles, e.g. 'so')."""
        return list(self._index.get(word.lower(), []))

    def cell(self, row, col):
        """Nodes of a cell (row 1 = Context up, col 1 = Adaptation on the left)."""
        return list(self._cells.get((row, col), []))

    def occupied(self):
        """{(row, col): count} of the cells that have nodes."""
        return {k: len(v) for k, v in self._cells.items()}

    def near(self, word, radius=1, limit=None):
        """Nodes in cells at distance <= radius (Chebyshev) from the word, not including itself."""
        out, seen = [], set()
        for node in self.get(word):
            r0, c0 = node["row"], node["col"]
            for r in range(r0 - radius, r0 + radius + 1):
                for c in range(c0 - radius, c0 + radius + 1):
                    for other in self._cells.get((r, c), []):
                        key = id(other)
                        if other is not node and key not in seen:
                            seen.add(key)
                            out.append(other)
        return out[:limit] if limit else out


class Tables:
    """All the tables (subjects, concepts, connectors)."""

    def __init__(self, data):
        self.version = data.get("version")
        self.meta = {k: v for k, v in data.items() if k != "tables"}
        self.tables = {name: Table(name, t["nodes"]) for name, t in data["tables"].items()}

    def __getitem__(self, name):
        return self.tables[name]

    def __iter__(self):
        return iter(self.tables.values())

    def find(self, word):
        """[(table, node)] across all tables."""
        return [(t.name, n) for t in self for n in t.get(word)]

    def synonyms(self, word):
        """Synonyms (WordNet) of all the nodes with that word."""
        out = []
        for _, n in self.find(word):
            out.extend(s for s in n.get("synonyms", []) if s not in out)
        return out

    # Weight of each link type when choosing the next node (higher = more reliable association).
    LINK_WEIGHT = {"manual": 1.0, "concept": 0.9, "is-a": 0.7, "kind-of": 0.7, "part": 0.6,
                   "derived": 0.5, "similar": 0.5, "corpus": 0.4, "definition": 0.2}
    USE_CORPUS = False        # True: `related_oanc` links (from tools/enrich_with_oanc.py) enter links(); off by default

    def links(self, word):
        """[(target_node, table, via, weight)] — all the links that `word` opens, without duplicating targets.

        via: 'manual' (narrative seed), 'concept' (related of the hand-made concepts) or the type
        derived from WordNet. If a target is reached by several routes, the one with the highest weight is kept.
        """
        best = {}
        for _, n in self.find(word):
            cands = [(t, "manual") for t in n.get("related_manual", [])] + [(t, "concept") for t in n.get("related", [])] \
                    + [(t, via) for t, via in n.get("related_wn", [])] \
                    + ([(t, "corpus") for t, _ in n.get("related_oanc", [])] if self.USE_CORPUS else [])
            for target, via in cands:
                for tb, node in self.find(target):
                    key = id(node)
                    w = self.LINK_WEIGHT.get(via, 0.1)
                    if key not in best or w > best[key][3]:
                        best[key] = (node, tb, via, w)
        return sorted(best.values(), key=lambda t: (-t[3], t[0]["word"]))

    def related(self, word, sources=None):
        """Nodes that `word` opens onto (they cross tables). `sources`: filters by route (e.g. {'manual', 'concept'})."""
        return [(tb, node) for node, tb, via, _ in self.links(word) if sources is None or via in sources]

    def opposites(self, word):
        """Opposites: WordNet antonyms that are nodes. Nodes without an antonym have `mirror()`."""
        out = []
        for _, n in self.find(word):
            for o in n.get("opposites", []):
                out.extend(hit for hit in self.find(o) if hit not in out)
        return out

    def mirror(self, word, radius=0):
        """Geometric opposite (decision #6): the reflection (x, y) -> (-x, -y) of each node with that word.

        Returns nodes of the same table in the reflected cell (or at distance <= radius from it). It is the
        alternative for the ~90 % of words without a lexical antonym: the 'most opposite' of fire is what
        is left on the opposite side of the grid, not necessarily a word with an antonym in WordNet.
        """
        out = []
        for tb, n in self.find(word):
            t = self[tb]
            r0, c0 = t.n + 1 - n["row"], t.n + 1 - n["col"]
            for r in range(r0 - radius, r0 + radius + 1):
                for c in range(c0 - radius, c0 + radius + 1):
                    out.extend(o for o in t.cell(r, c) if o is not n and all(o is not x for x in out))
        return out

    def stats(self):
        res = {}
        for t in self:
            occ = t.occupied()
            top = max(occ.items(), key=lambda kv: kv[1])
            res[t.name] = {"nodes": len(t), "occupied_cells": len(occ), "largest_cell": {"row": top[0][0], "col": top[0][1], "nodes": top[1]}}
        return res


def load(path=None):
    """Loads data/LCM_tables.json (or the given path)."""
    with open(path or _DEFAULT, encoding="utf-8") as f:
        return Tables(json.load(f))
