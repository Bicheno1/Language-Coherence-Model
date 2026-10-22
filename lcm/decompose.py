"""Decomposer: recognizes the ROLES of an input by value, without carrying any hard-coded word.

A TEMPLATE carries words (\"There was a X who…\"). A RULE recognizes and decomposes by value (this is an action, this is the
speaker, this is past) and recomposes by value. This module is the first half: DECOMPOSE. Each word of the input is
turned into a card with what the tables and the dynamic rules say about it:

  position and axis   x, y, cell, axis (North = future, South = past, West/East = present)
  tense by height     the one marked by its y (lcm.tense.tense_of_position); for an action also the tense the sentence asks for
  perspective         the you<->I, me<->you inversion (lcm.deixis): who speaks and who listens
  forms               for an action, its form in the tense it calls for (base conjugation) and the inflected nodes that exist
  related             links by type (manual, es-un, tipo-de, derivada…), synonyms and lexical opposite, if there is one
  opposite            the lexical one (WordNet) when it exists; for pronouns, the perspective inversion. The geometric mirror is NOT used:
                      the cell is so wide that the opposite of 'story' comes out 'jam, soup, vodka' (measured).

It generates nothing: it is the input of a recomposer that does not exist yet.
"""
from dataclasses import dataclass, field

from .deixis import invert
from .geometry import axis_of
from .tense import tense_of_position, shift


@dataclass
class Card:
    role_obj: str                          # action | agent | recipient | topic | about-topic
    word: str
    node: dict = None
    axis: str = None
    tense_by_height: str = None         # tense marked by its y
    tense_requested: str = None          # actions only: the tense the sentence asks for (imperative = future)
    perspective: str = None           # word the LCM uses in its place (you -> i, me -> you)
    perspective_side: str = None      # its pole (Imposition / Adaptation)
    forms: dict = field(default_factory=dict)       # {'past': 'told', 'present': 'tells', 'future': 'will tell'}
    inflected: list = field(default_factory=list)  # inflected nodes of the same word that exist in the tables
    related: dict = field(default_factory=dict)      # via -> [words]
    synonyms: list = field(default_factory=list)
    opposites: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "node" and v not in (None, {}, [])}
        if self.node:
            d["cell"] = (self.node["row"], self.node["col"])
            d["xy"] = (self.node["x"], self.node["y"])
        return d


def _inflection_nodes(tables, lemma_text):
    """Inflected forms of a verb that are already nodes (told, said…): lemminflect gives the forms, the tables say whether they exist."""
    from lemminflect import getAllInflections
    out = {}
    for tag, forms in getAllInflections(lemma_text, upos="VERB").items():
        for f in forms:
            if f != lemma_text and tables.find(f):
                out.setdefault(tag, []).append(f)
    return out


def _forms(lemma_text):
    from lemminflect import getInflection
    return {"past": (getInflection(lemma_text, "VBD") or (lemma_text,))[0], "present": (getInflection(lemma_text, "VBZ") or (lemma_text,))[0],
            "future": "will " + lemma_text}


def _card(tables, role_obj, role, frame=None, node_override=None):
    node = node_override or (role.node if role is not None else None)
    f = Card(role_obj=role_obj, word=(role.text if role is not None else "(implicit)"), node=node)
    if node is None:
        f.notes.append("it is not a node of the tables: it cannot be decomposed")
        return f
    w = node["word"]
    f.axis, f.tense_by_height = axis_of(node["x"], node["y"]), tense_of_position(node["y"])
    inv = invert(tables, w)
    if inv is not None:
        f.perspective, f.perspective_side = inv["word"], inv["category"]
        f.opposites = [inv["word"]]
        f.notes.append(f"'{w}' is deictic: the LCM reads it as '{inv['word']}' ({inv['category']})")
    if node.get("type") == "action":
        if frame is not None:
            f.tense_requested = frame.tense + (" (negated)" if frame.negated else "") + (f", {frame.form}" if frame.form else "")
        f.forms = _forms(role.lemma_text if role is not None else w)
        f.inflected = sorted({x for v in _inflection_nodes(tables, role.lemma_text if role is not None else w).values() for x in v})
    grouped = {}
    for n, _, via, _ in tables.links(w):
        grouped.setdefault(via, []).append(n["word"])
    f.related = {via: ws[:8] for via, ws in grouped.items() if via != "definition"}
    f.synonyms = (node.get("synonyms") or [])[:8]
    if not f.opposites:
        f.opposites = [n["word"] for _, n in tables.opposites(w)]
    return f


def decompose(tables, frame):
    """Cards for the roles of the Frame (lcm.parse): action, agent, recipient, topic, about-topic, and the quantity in notes."""
    cards = []
    if frame.action is not None:
        cards.append(_card(tables, "action", frame.action, frame))
    if frame.subject is not None:
        # agent: the subject AS THE LCM READS IT (you -> i). The implied subject of an imperative is also 'you' -> 'i'.
        f = _card(tables, "agent", frame.subject, node_override=frame.subject.node)
        if frame.subject.text == "(you)":
            f.word = "(implicit)"
            f.notes.append("implicit subject of the imperative: whoever answers")
        cards.append(f)
    if frame.recipient is not None:
        cards.append(_card(tables, "recipient", frame.recipient))
    if frame.topic is not None:
        t = _card(tables, "topic", frame.topic)
        if frame.quantity is not None:
            t.notes.append(f"quantity: {frame.quantity}")
        cards.append(t)
    if frame.about_topic is not None:
        cards.append(_card(tables, "about_topic", frame.about_topic))
    return cards
