"""Input segmentation: text -> roles (action, subject, recipient, quantity, topic, tense) -> nodes.

Guiding example:   \"tell me a story\"
    action = tell   recipient = me   quantity = 1 (a)   topic = story   tense = futuro (imperative)

The roles come from spaCy's dependency analysis; each one is ANCHORED in the tables (its node is looked up), and the
action receives the tense/negation shift of `lcm.tense`. If a word is not a node, the role stays with
`nodo = None` and is listed in `Frame.faltan` (nothing is invented).

Requires:  pip install spacy && python -m spacy download en_core_web_sm      (English only: open decision #1)
"""
from dataclasses import dataclass, field

from .deixis import INVERSION, invert
from .tense import shift, recipient_role

_NLP = None

QUANTITY = {  # determiners and quantifiers -> quantity (number or label)
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10,
    "some": "some", "several": "several", "many": "many", "few": "few", "all": "all_items",
    "every": "each", "each": "each", "any": "any", "another": 1, "no": 0,
}
TIME_WORDS = {  # time adverbs that override verb morphology
    "yesterday": "past", "ago": "past", "before": "past", "earlier": "past", "once": "past", "previously": "past",
    "tomorrow": "future", "later": "future", "soon": "future", "next": "future", "tonight": "future", "eventually": "future",
    "now": "present", "today": "present", "currently": "present",
}
REQUEST_AUX = {"can", "could", "would", "will"}      # "can you tell me…?" is a request, not a question about ability
FUTURE_AUX = {"will", "shall", "'ll", "wo"}          # "wo" = won't after spaCy tokenization


def _nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except (ImportError, OSError) as e:              # pragma: no cover
            raise RuntimeError("parse needs spaCy and the model: pip install spacy && python -m spacy download en_core_web_sm") from e
    return _NLP


@dataclass
class Role:
    """A role of the sentence: the word, its lemma and the node where it is anchored (or None)."""
    text: str
    lemma_text: str
    table: str = None
    node: dict = None
    plural: bool = False              # 'ducks' (NNS): the node is the animal, but the surface form is plural
    det: str = None                   # determiner the user wrote ('a', 'the'…); None if none ('about war')

    def to_dict(self):
        d = {"text": self.text, "lemma": self.lemma_text, "anchored": self.node is not None}
        if self.node:
            d.update({"table": self.table, "cell": (self.node["row"], self.node["col"]),
                      "weights": {k: self.node[k] for k in ("V", "I", "Lv", "Gv")}})
        return d


@dataclass
class Frame:
    """Result of segmenting an input."""
    input_text: str
    action: Role = None
    subject: Role = None
    recipient: Role = None
    topic: Role = None
    about_topic: Role = None               # "about X" / "of X" / "on X": what the topic is about (a story ABOUT a king)
    quantity: object = None           # number or label (1, 2, 'some'…); None if not stated
    tense: str = "present"          # past | present | future
    tense_reason: str = "by default"   # where the tense comes from (for auditing)
    negated: bool = False
    mode: str = "declarative"         # imperative | interrogative | declarative
    quality: Role = None                  # attributive adjective of the subject: "Are you HAPPY?" (acomp)
    qualities: list = field(default_factory=list)   # adjectives of the topic
    about_qualities: list = field(default_factory=list)   # adjectives of the about-topic (brave king -> ['brave'])
    topic_quality_nodes: list = field(default_factory=list)   # nodes of the topic's qualities ('sad story' -> node sad): they move the target
    displaced_action: dict = None    # weights/position of the action after tense and negation
    displaced_recipient: dict = None   # the recipient after perspective inversion ('me' -> node 'you': Adaptation)
    displaced_subject: dict = None         # the subject after perspective inversion ('you' -> node 'i': Imposition)
    unknown: str = None                  # 'what', 'who'… when the question has an interrogative word that is what is being asked
    form: str = None                      # 'question' (?) | 'exclamation' (!) | None: punctuation, which shifts the position (lcm.tense)
    missing: list = field(default_factory=list)      # words without a node

    def roles(self):
        return {k: getattr(self, k) for k in ("action", "subject", "recipient", "topic", "about_topic", "quality") if getattr(self, k)}

    def to_dict(self):
        return {"input": self.input_text, "mode": self.mode, "form": self.form, "unknown": self.unknown, "tense": self.tense, "tense_reason": self.tense_reason,
                "negated": self.negated, "quantity": self.quantity, "qualities": self.qualities,
                "about_qualities": self.about_qualities,
                **{k: r.to_dict() for k, r in self.roles().items()}, "missing": self.missing,
                **({"displaced_action": {k: self.displaced_action[k] for k in ("V", "I", "x", "y", "row", "col")}}
                   if self.displaced_action else {}),
                **({"displaced_recipient": {k: self.displaced_recipient[k] for k in ("Lv", "Gv", "x", "y", "row", "col")}}
                   if self.displaced_recipient else {})}


def _anchor(tables, tok, prefer, lemma_first=False):
    """Looks up the node of a token: surface form and lemma (the lemma first if `lemma_first`), among tables in order of
    preference. A NOUN is not anchored on an action if there is another option: 'ducks' is the animal, not 'to duck' (the lemma
    'duck' only has the action node; the plural 'ducks' has the animal's)."""
    keys = (tok.text.lower(), tok.lemma_.lower())
    order = keys[::-1] if lemma_first else keys
    plural = tok.tag_ in ("NNS", "NNPS")
    det = next((c.lower_ for c in tok.children if c.dep_ == "det"), None)
    found = []                                   # (table, node) by key, in order
    for key in order:
        hits = [(tb, h) for tb in prefer for h in tables[tb].get(key)]
        if hits:
            if tok.pos_ in ("NOUN", "PROPN"):
                no_action = [x for x in hits if x[1].get("type") != "action"]
                if no_action:
                    return Role(tok.text, tok.lemma_.lower(), no_action[0][0], no_action[0][1], plural, det)
            elif tok.pos_ == "VERB":
                actions = [x for x in hits if x[1].get("type") == "action"]
                return Role(tok.text, tok.lemma_.lower(), *(actions or hits)[0], plural, det)
            found.append(hits[0])
    if tok.pos_ in ("NOUN", "PROPN"):
        # 'a duck': in the tables 'duck' is only the verb (to duck) and the animal only exists as 'ducks'. A noun for which only the
        # action was found tries its plural: the node is the animal's, the surface form stays singular.
        from lemminflect import getInflection
        pl = (getInflection(tok.lemma_.lower(), "NNS") or (None,))[0]
        hits = [(tb, h) for tb in prefer for h in (tables[tb].get(pl) if pl else []) if h.get("type") != "action"]
        if hits:
            return Role(tok.text, tok.lemma_.lower(), hits[0][0], hits[0][1], plural, det)
    if found:
                return Role(tok.text, tok.lemma_.lower(), found[0][0], found[0][1], plural, det)
    return Role(tok.text, tok.lemma_.lower(), plural=plural, det=det)


def _tense(root, doc):
    """(tense, why): time adverb > future auxiliary > morphology > imperative > present."""
    for t in doc:
        if t.lower_ in TIME_WORDS and t.dep_ in ("advmod", "npadvmod", "amod", "advcl"):
            return TIME_WORDS[t.lower_], f"adverb '{t.text}'"
    aux = [c for c in root.children if c.dep_ in ("aux", "auxpass")]
    if any(a.lower_ in FUTURE_AUX for a in aux) or any(a.lower_ == "going" for a in aux):
        return "future", "future auxiliary"
    if "Past" in root.morph.get("Tense") or any("Past" in a.morph.get("Tense") for a in aux if a.lemma_ in ("have", "be", "do")):
        return "past", "verb morphology"
    if "Imp" in root.morph.get("Mood") or (root.tag_ == "VB" and not any(c.dep_ == "nsubj" for c in root.children)):
        return "future", "imperative: the requested action has not happened yet"
    return "present", "by default"


def parse(text, tables):
    """Segments `text` and anchors each role in `tables` (the result of `lcm.load()`)."""
    # spaCy misreads UPPERCASE TEXT ("DUCKS" -> proper noun with lemma "DUCKS"): it is analyzed in lowercase
    doc = _nlp()(text.lower() if text.isupper() else text)
    sents = list(doc.sents)
    root = next((t for t in sents[0] if t.dep_ == "ROOT"), sents[0][0]) if sents else doc[0]
    f = Frame(input_text=text)
    es_let = root.lemma_ == "let" and any(c.dep_ in ("ccomp", "xcomp") for c in root.children)
    if es_let:
        root = next(c for c in root.children if c.dep_ in ("ccomp", "xcomp"))

    # action: the root if it is a verb; in a copular "is X" the root is the attribute and the action is 'be'
    f.action = _anchor(tables, root, ("concepts", "subjects"), lemma_first=True)   # jumped -> jump
    f.tense, f.tense_reason = _tense(root, doc)
    if es_let:
        f.tense, f.tense_reason = "future", "imperative: let's / let me"
    f.negated = any(c.dep_ == "neg" for c in root.children)

    kids = {}
    for c in root.children:
        kids.setdefault(c.dep_, []).append(c)

    wh_adv = next((c for c in root.children if c.tag_ == "WRB"), None)            # "HOW are you?", "WHERE are you?"
    if wh_adv is not None:
        f.unknown = wh_adv.lower_
    imperative = "future" == f.tense and "imperative" in f.tense_reason
    f.mode = "imperative" if imperative else "interrogative" if text.strip().endswith("?") else "declarative"
    f.form = "question" if text.strip().endswith("?") else "exclamation" if text.strip().endswith("!") else None
    if f.form == "exclamation" and f.mode == "declarative":
        f.mode = "exclamative"

    # subject: explicit, or implied 'you' in an imperative
    subj = (kids.get("nsubj") or kids.get("nsubjpass") or [None])[0]
    if subj is None:
        for c in kids.get("acomp", []) + kids.get("attr", []):
            subj = next((x for x in c.children if x.dep_ == "nsubj"), None)
            if subj is not None:
                break
    if subj is not None:
        f.subject = _anchor(tables, subj, ("subjects", "concepts"))
    elif imperative:
        f.subject = Role("(you)", "you", "subjects", (tables["subjects"].get("you") or [None])[0])

    # recipient: dative / indirect object, or a to/for complement
    dative = (kids.get("dative") or kids.get("iobj") or [None])[0]
    if dative is None:
        for prep in kids.get("prep", []):
            if prep.lower_ in ("to", "for"):
                dative = next((c for c in prep.children if c.dep_ == "pobj"), None)
                break
    if dative is not None:
        f.recipient = _anchor(tables, dative, ("subjects", "concepts"))

    # topic: direct object; if none, the attribute; otherwise the first noun that is neither subject nor recipient
    obj = (kids.get("dobj") or kids.get("obj") or kids.get("attr") or kids.get("oprd") or [None])[0]
    if obj is not None and obj.tag_ in ("WP", "WDT", "WRB"):
        if root.lemma_ == "be" and subj is not None:         # "What is a story?": the topic is the subject
            f.unknown = obj.lower_
            obj, f.subject, f.mode = subj, None, "interrogative"
        else:                                                # "What do you think about ducks?": 'what' IS what is being asked
            f.unknown, obj = obj.lower_, None
    if obj is not None and dative is None and obj.lower_ in INVERSION and \
            any(c.dep_ == "prep" and c.lower_ in ("about", "of", "on", "regarding") for c in root.children):
        # "Tell me about ducks": spaCy gives 'me' as direct object, but with an 'about X' it is the RECIPIENT; the topic is X
        dative, obj = obj, None
        f.recipient = _anchor(tables, dative, ("subjects", "concepts"))
    if obj is None:
        taken = {t for t in (subj, dative) if t is not None}
        obj = next((t for t in doc if t.pos_ in ("NOUN", "PROPN") and t not in taken and t != root
                    and t.dep_ not in ("npadvmod", "tmod") and t.lower_ not in TIME_WORDS), None)
    if obj is not None:
        f.topic = _anchor(tables, obj, ("concepts", "subjects"), lemma_first=True)      # stories -> story; the quantity has already been read
        for c in obj.children:
            if c.dep_ == "det" and c.lower_ in QUANTITY:
                f.quantity = QUANTITY[c.lower_]
            elif c.dep_ == "nummod":
                try:
                    f.quantity = int(c.text)
                except ValueError:
                    f.quantity = QUANTITY.get(c.lower_, c.text)
            elif c.dep_ == "amod":
                f.qualities.append(c.lemma_.lower())
                q = _anchor(tables, c, ("concepts", "subjects"), lemma_first=True)
                if q.node is not None:
                    f.topic_quality_nodes.append(q.node)
        if f.quantity is None and obj.tag_ == "NNS":      # plural without determiner: "tell me stories"
            f.quantity = "several"

    # about-topic: "about a brave king" (of the topic or of the root). It is a strong anchor for the walk.
    for head in [h for h in (obj, root) if h is not None]:
        prep = next((c for c in head.children if c.dep_ == "prep" and c.lower_ in ("about", "of", "on", "regarding")), None)
        pobj = next((c for c in prep.children if c.dep_ == "pobj"), None) if prep is not None else None
        if pobj is not None:
            f.about_topic = _anchor(tables, pobj, ("subjects", "concepts"), lemma_first=True)
            f.about_qualities = [c.lemma_.lower() for c in pobj.children if c.dep_ == "amod"]
            break

    # polite request: "can/could/would you <verb> …?" asks for a future action
    if f.mode == "interrogative" and subj is not None and subj.lower_ == "you" and \
            any(c.dep_ == "aux" and c.lower_ in REQUEST_AUX for c in root.children):
        f.mode, f.tense, f.tense_reason = "request", "future", "request (can/could/would you…)"

    acomp = (kids.get("acomp") or [None])[0]
    if acomp is not None:
        f.quality = _anchor(tables, acomp, ("concepts", "subjects"), lemma_first=True)

    # the action receives the tense and the negation (design decision: tense moves the action's weight)
    if f.action.node:
        f.displaced_action = shift(f.action.node, f.tense, f.negated, f.form)
    # perspective inversion (lcm.deixis): the asker's 'you' is the system's 'I', and their 'me' is 'you'
    if f.subject is not None and f.subject.node:
        f.displaced_subject = invert(tables, f.subject.node["word"])
    elif f.subject is None and f.topic is not None and f.topic.node and root.lemma_ == "be" and f.unknown in ("who", "what"):
        # "Who/What are YOU?": spaCy leaves 'you' as attribute (topic); perspective inversion applies there too
        f.displaced_subject = invert(tables, f.topic.node["word"])
    if f.recipient is not None and f.recipient.node:
        f.displaced_recipient = invert(tables, f.recipient.node["word"]) or recipient_role(f.recipient.node)

    f.missing = [r.text for r in f.roles().values() if r.node is None and r.text != "(you)"]
    return f
