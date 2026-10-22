"""LCM geometry: weights -> position (GDE, reading D1) -> cell.

Each node carries four integer weights 0-10:
    V  -> Context     (up)         I  -> World       (down)
    Lv -> Imposition  (right)      Gv -> Adaptation  (left)

    x = (Lv - Gv) / 2      x > 0 toward Imposition, x < 0 toward Adaptation
    y = (V  - I ) / 2      y > 0 toward Context,    y < 0 toward World

Cell of an n x n table:  col = ceil((x + 5) * n / 10),  row = ceil((5 - y) * n / 10),
clipped to 1..n. Row 1 = Context, row n = World; col 1 = Adaptation, col n = Imposition.
(With even n there is no central cell: x = 0 falls in column n/2 and y = 0 in row n/2.)
"""
import math

N = 10
WEIGHT_MIN, WEIGHT_MAX = 0, 10

AXES = {
    "V":  {"pole": "Context",    "side": "up",    "pronoun": "We",   "character": "Possibility"},
    "I":  {"pole": "World",      "side": "down",     "pronoun": "That", "character": "logical"},
    "Lv": {"pole": "Imposition", "side": "right",   "pronoun": "I",    "character": "Sensory"},
    "Gv": {"pole": "Adaptation", "side": "left", "pronoun": "You",  "character": "structural"},
}


def position(V, I, Lv, Gv):
    """D1 centroid of the four weights: (x, y)."""
    return (Lv - Gv) / 2.0, (V - I) / 2.0


def cell(x, y, n=N):
    """(column, row) of the n x n grid, starting at 1."""
    col = max(1, min(n, math.ceil((x + 5) * n / 10.0)))
    row = max(1, min(n, math.ceil((5 - y) * n / 10.0)))
    return col, row


def tension(V, I):
    """Ambivalence: the centroid collapses fire (V=I=8) and stone (V=I=1) to the same point; this does not."""
    return min(V, I)


def axis_of(x, y, threshold=1.5):
    """Main axis: vertical if |y| >= threshold (North = future / South = past); otherwise, the side of the present."""
    if y >= threshold:
        return "North"
    if y <= -threshold:
        return "South"
    return "West" if x <= -1 else "East" if x >= 1 else "Present"
