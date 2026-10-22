"""Provisional FUNCTIONAL nodes: words the recomposer needs that are NOT in the tables.

The tables have almost no function words (determiners, prepositions, particles: 2-26 % coverage in the corpus). Until
they exist in the database, they are declared here as nodes with weights (V, I, Lv, Gv) assigned by what each axis means, and are chosen
BY VALUE, not hard-coded inside a sentence. They are design decisions, provisional, and should live in the tables.

    Context (V, up)  = possibility: the not-yet-identified, what opens
    World   (I, down) = the logical, the established, the past, what already exists
    Imposition (Lv)   = the speaker's (sensory); Adaptation (Gv) = the listener's (structural)

  a / an   indefinite determiner: introduces a NEW, not yet identified unit              -> Context (possibility)   y = +3
  the      definite determiner: what is already identified and shared                    -> World (established)     y = -2
  there    existential expletive: puts something into existence, far from the speaker, in the world -> World, not the speaker's   y = -3
"""
from .geometry import cell, position

# word -> (V, I, Lv, Gv, reason)
FUNCTION_NODES = {
    "a":     (8, 2, 5, 5, "indefinite: new unit, not yet identified = possibility (Context)"),
    "the":   (3, 7, 5, 5, "definite: already identified and shared = established (World)"),
    "there": (2, 8, 2, 6, "existential: exists in the world, far from the speaker (World, not Imposition)"),
}


def node(word):
    """Node (with x, y, cell) of a function word, or None."""
    if word not in FUNCTION_NODES:
        return None
    V, I, Lv, Gv, rationale = FUNCTION_NODES[word]
    x, y = position(V, I, Lv, Gv)
    col, row = cell(x, y)
    return {"word": word, "type": "function", "V": V, "I": I, "Lv": Lv, "Gv": Gv, "x": x, "y": y, "col": col, "row": row, "rationale": rationale}


def determiner(is_new, following):
    """Determiner chosen BY VALUE: the new (possibility) takes the one with the highest y; the already mentioned, the one with the lowest y.
    The variant 'an' before a vowel is morphology, not content."""
    cands = [node("a"), node("the")]
    n = max(cands, key=lambda c: c["y"]) if is_new else min(cands, key=lambda c: c["y"])
    word = ("an" if following[:1].lower() in "aeiou" else "a") if n["word"] == "a" else n["word"]
    return word, n
