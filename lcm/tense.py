"""Verb tense and negation as a shift of a node's weights.

Time runs vertically: past = South (World, weight I), future = North (Context, weight V); the present is the horizontal
band and moves nothing. Negation also goes South (connectors criterion). This replaces
Unique Host's VERB_TENSES, which was a table of surface forms generated blindly (wassed, hadded…) and which
never moved the weight of an action.

Standard library only. The shifts are DESIGN DECISIONS (first pass), editable here.
"""
from .geometry import position, cell, WEIGHT_MIN, WEIGHT_MAX

# (dV, dI): how much is added to Context (V) and World (I). Lv/Gv are untouched: tense does not change the sensory/relational.
TENSE_SHIFT = {
    "past":   (-2, +3),
    "present": (0, 0),
    "future":   (+3, -2),
}
# Mode: QUESTIONS live in possibility (North, Context) and EXCLAMATIONS in the South (World): a question leaves open
# what an exclamation takes as closed. Same orders of magnitude as tense; design decision, editable.
MODE_SHIFT = {
    "question":    (+2, -2),
    "exclamation": (-2, +2),
}
NEGATION_SHIFT = (-3, +3)      # "not doing" closes the action: toward World, like nor / neither…nor


def _clamp(v):
    return max(WEIGHT_MIN, min(WEIGHT_MAX, v))


# Role: whoever RECEIVES adapts instead of imposing. In "tell ME a story", 'me' is an Imposition pronoun (Lv 10, Gv 0)
# but, as a recipient, it is read in Adaptation: Lv and Gv are swapped (everything else stays the same).
# It is a ROLE rule (of the sentence), not a lexical one: 'I' as subject stays in Imposition. Design decision, editable.
def recipient_role(node):
    """Weights, position and cell of `node` read as a recipient (Lv <-> Gv). It does not modify the node."""
    Lv, Gv = node["Gv"], node["Lv"]
    x, y = position(node["V"], node["I"], Lv, Gv)
    col, row = cell(x, y)
    return {"V": node["V"], "I": node["I"], "Lv": Lv, "Gv": Gv, "x": x, "y": y, "col": col, "row": row,
            "tension": min(node["V"], node["I"])}


def tense_of_position(y, band=0.5):
    """Tense marked by a node's height: South (y < -band) = past, central band = present, North = future."""
    return "past" if y < -band else "future" if y > band else "present"


def shift(node, tense="present", negated=False, form=None):
    """Weights, position and cell of `node` after applying tense, negation and mode (`form`: question | exclamation). It does not modify the node.

    Returns a dict with V, I, Lv, Gv, x, y, col, row, tension.
    """
    if tense not in TENSE_SHIFT:
        raise ValueError(f"unknown tense: {tense!r} (use {sorted(TENSE_SHIFT)})")
    dV, dI = TENSE_SHIFT[tense]
    if negated:
        dV, dI = dV + NEGATION_SHIFT[0], dI + NEGATION_SHIFT[1]
    if form:
        if form not in MODE_SHIFT:
            raise ValueError(f"unknown form: {form!r} (use {sorted(MODE_SHIFT)})")
        dV, dI = dV + MODE_SHIFT[form][0], dI + MODE_SHIFT[form][1]
    V, I = _clamp(node["V"] + dV), _clamp(node["I"] + dI)
    Lv, Gv = node["Lv"], node["Gv"]
    x, y = position(V, I, Lv, Gv)
    col, row = cell(x, y)
    return {"V": V, "I": I, "Lv": Lv, "Gv": Gv, "x": x, "y": y, "col": col, "row": row, "tension": min(V, I)}
