"""Perspective inversion (deixis): what the user calls 'you' is 'I' for the LCM.

The user speaks FROM their own position: their 'I/me' is the system's second person, and their 'you' is the system itself. Therefore:

    \"What do you think about ducks?\"   'you' -> 'I'   (Imposition: the system is the one answering)
    \"Tell me a story\"                  'me'  -> 'you' (Adaptation: the receiver)  and the implicit 'you' of 'tell' -> 'I'

It is not a new weight rule: it uses the NODE that is already in the Subjects table (*you* = V6 I2 Lv0 Gv10, cell (3,1);
*I* = V6 I2 Lv10 Gv0, cell (3,10)). In v0.4c the recipient was read in Adaptation by swapping Lv and Gv; that was this same
inversion without a name (for 'me' it gives exactly the weights of 'you'). Standard library only.
"""

INVERSION = {
    "i": "you", "me": "you", "my": "your", "mine": "yours", "myself": "yourself",
    "you": "i", "your": "my", "yours": "mine", "yourself": "myself",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself"}
SECOND_PERSON = {"you", "your", "yours", "yourself"}


def invert(tables, word):
    """Subjects node that the LCM uses instead of `word` (you -> i, me -> you…), or None if it is not 1st/2nd person."""
    w = INVERSION.get(word.lower())
    hits = tables["subjects"].get(w) if w else None
    return hits[0] if hits else None


def is_deictic(word):
    return word.lower() in INVERSION
