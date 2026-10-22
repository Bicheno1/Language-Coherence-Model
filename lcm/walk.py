"""Walk of the tables: opens one node from another through its links, guided and measured by the geometry (GDE).

Design decisions it implements (#10 and #11; they are a FIRST PASS, editable below):

  #10  The GDE does both things: it GUIDES (each candidate is scored by its closeness to the target point) and MEASURES
       (the drift of the walk = mean distance to the target; coherence is 1 - normalized drift).
  #11  Choice among candidates: highest score = link_weight x closeness  (alphabetical tie-break, deterministic).
       Stop: `max_steps` was reached, or no candidate is left with score >= `min_score`, or the drift exceeds
       `max_drift`.

Target point = centroid of the sentence's anchors (topic and about-topic count double; the action counts with the
tense/negation shift of lcm.tense). What comes out is a WALK OF NODES, not text: the realization
into sentences (grammar, order) does not exist yet.
"""
import math
from collections import deque
from dataclasses import dataclass, field

R_MAX = 5 * math.sqrt(2)          # reasonable maximum distance to the target (corner of the plane)

MAX_STEPS = 8
MIN_SCORE = 0.15
MAX_DRIFT = 0.75                 # fraction of R_MAX; above it the walk is cut
GUIDE = 0.5                        # 0 = ignore geometry (only the link weighs); 1 = pure closeness to the target
DIVERSITY = 0.6                  # each node already visited in the SAME cell multiplies the score by this factor
TYPE_DIVERSITY = 0.7             # same, per already-visited node of the SAME type (biological / action / quality / non_biological)


@dataclass
class Step:
    word: str
    table: str
    via: str                      # manual | concept | is-a | … | anchor
    from_word: str                    # node that opened it (None for anchors)
    x: float
    y: float
    cell: tuple
    distance: float              # to the target
    score: float = None

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class WalkResult:
    target: tuple
    steps: list = field(default_factory=list)
    stop: str = ""

    @property
    def words(self):
        return [p.word for p in self.steps]

    @property
    def drift(self):
        """Mean distance to the target, normalized to 0..1 (0 = everything on the target)."""
        if not self.steps:
            return 0.0
        return min(1.0, sum(p.distance for p in self.steps) / len(self.steps) / R_MAX)

    @property
    def coherence(self):
        return round(1.0 - self.drift, 3)

    def to_dict(self):
        return {"target": self.target, "stop": self.stop, "drift": round(self.drift, 3),
                "coherence": self.coherence, "steps": [p.to_dict() for p in self.steps]}


def _dist(x, y, target):
    return math.hypot(x - target[0], y - target[1])


def target_of(frame):
    """Centroid (x, y) of the frame's anchors. Topic and about-topic weigh double; the action is already shifted by tense."""
    pts = []
    if frame.action and frame.action.node:
        d = frame.displaced_action or frame.action.node
        pts.append((d["x"], d["y"], 1))
    for role in (frame.topic, frame.about_topic):
        if role and role.node:
            pts.append((role.node["x"], role.node["y"], 2))
    for q in getattr(frame, "topic_quality_nodes", []):      # "a SAD story": the topic's quality pulls the target
        pts.append((q["x"], q["y"], 2))
    if frame.recipient and frame.recipient.node:       # the recipient counts by its ROLE (Adaptation), not by its lexicon
        d = frame.displaced_recipient or frame.recipient.node
        pts.append((d["x"], d["y"], 1))
    if frame.subject and frame.subject.node:                    # the subject, after perspective inversion (you -> I)
        s = frame.displaced_subject or frame.subject.node
        pts.append((s["x"], s["y"], 1))
    if not pts:
        return None
    w = sum(p[2] for p in pts)
    return (sum(p[0] * p[2] for p in pts) / w, sum(p[1] * p[2] for p in pts) / w)


def _step(node, table, via, from_word, target, score=None):
    return Step(node["word"], table, via, from_word, node["x"], node["y"], (node["row"], node["col"]),
                round(_dist(node["x"], node["y"], target), 3), score)


def walk(tables, frame, max_steps=MAX_STEPS, min_score=MIN_SCORE, max_drift=MAX_DRIFT,
         guide=GUIDE, diversity=DIVERSITY, type_diversity=TYPE_DIVERSITY):
    """Walks the tables from the anchors of `frame`. Returns a WalkResult.

    score = link_weight x (1 - guide x distance/R_MAX) x diversity^(same cell) x type_diversity^(same type)
    With guide = 1 and diversity = 1 it is pure closeness: it never crosses from Adaptation to Imposition, so
    story -> person -> jump is not reached (measured). guide < 1 lets the walk cross the plane.
    """
    target = target_of(frame)
    if target is None:
        return WalkResult((0.0, 0.0), [], "no anchors: no word of the input is a node")
    rec = WalkResult(target=tuple(round(v, 3) for v in target))
    visited = {}                                           # word -> node
    for role in (frame.topic, frame.about_topic, frame.action, frame.recipient, frame.subject):
        if role and role.node and role.node["word"] not in visited:
            visited[role.node["word"]] = role.node
            rec.steps.append(_step(role.node, role.table, "anchor", None, target))

    types = {}                                               # word -> node type (for per-type diversity)
    for p0 in rec.steps:
        types[p0.word] = visited[p0.word].get("type")
    n_anchors = len(rec.steps)
    while len(rec.steps) - n_anchors < max_steps:
        best = None
        for p in list(rec.steps):
            for node, tb, via, w in tables.links(p.word):
                if tb == "connectors" or node["word"] in visited:
                    continue
                near = max(0.0, 1.0 - guide * _dist(node["x"], node["y"], target) / R_MAX)
                in_cell = sum(1 for q in rec.steps if q.cell == (node["row"], node["col"]))
                in_type = sum(1 for q in rec.steps if types.get(q.word) == node.get("type"))
                score = w * near * (diversity ** in_cell) * (type_diversity ** in_type)
                if best is None or score > best[0] or (score == best[0] and node["word"] < best[1]["word"]):
                    best = (score, node, tb, via, p.word)
        if best is None:
            rec.stop = "no candidates"
            break
        score, node, tb, via, from_word = best
        if score < min_score:
            rec.stop = f"score {score:.2f} < {min_score}"
            break
        visited[node["word"]] = node
        types[node["word"]] = node.get("type")
        rec.steps.append(_step(node, tb, via, from_word, target, round(score, 3)))
        if rec.drift > max_drift:
            rec.stop = f"drift {rec.drift:.2f} > {max_drift}"
            break
    else:
        rec.stop = f"{max_steps} steps"
    return rec


def path(tables, start, end, vias=None, max_hops=4):
    """Shortest path from `origin` to `end` following links (BFS). `vias` filters the link type.

    Returns [(word, via)] starting at the origin, or None. It serves to check that a walk
    like story -> person -> jump exists with the loaded links.
    """
    start, end = start.lower(), end.lower()
    queue, previous = deque([start]), {start: None}
    depth = {start: 0}
    while queue:
        w = queue.popleft()
        if w == end:
            break
        if depth[w] >= max_hops:
            continue
        for node, tb, via, _ in tables.links(w):
            nxt = node["word"].lower()
            if tb == "connectors" or nxt in previous or (vias and via not in vias):
                continue
            previous[nxt] = (w, via)
            depth[nxt] = depth[w] + 1
            queue.append(nxt)
    if end not in previous:
        return None
    out, w = [], end
    while w is not None:
        link = previous[w]
        out.append((w, link[1] if link else "origin"))
        w = link[0] if link else None
    return out[::-1]
