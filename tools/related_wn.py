# related_wn.py — LCM v0.4: derives `related_wn` links and `opposites` (opposites) from WordNet.
#
# Answers problems A (there was almost no related) and I (there were no opposites).
# It only links to words that ARE already nodes (Subjects or Concepts); the rest is discarded.
#
# Derived link types ("via"):
#   is-a          hypernym (goes up the WordNet tree)              ┐ vertical axis
#   kind-of       hyponym (goes down the tree)                      ┘
#   part          meronym / holonym (part-of, member-of, made-of)
#   derived       derivationally related forms (tell -> teller, story -> storyteller…)
#   similar       similar_tos / also_sees / verb_groups / entailments
#   definition    content words of the sense gloss (weak association, the noisiest)
# All are DETERMINISTIC: ordered by lemma frequency in WordNet, then alphabetically.

import re
from nltk.corpus import wordnet as wn

MAX_SENSES = 3                       # senses per word (the most frequent in WordNet)
CAP_VIA = {"is-a": 3, "kind-of": 3, "part": 3, "derived": 3, "similar": 3, "definition": 4}
CAP_TOTAL = 14

_STOP = set("""a an the of or and to in on at by for with from as is are was were be been being it its this that these those
which who whom whose what when where while into onto over under than then there their they them he she his her you your we our
not no nor but if so such any some one two used use uses using usually especially often something someone somebody person people
thing things way ways kind type form act state having having made make makes making very more most less least can may might will
would should could also about between through during before after above below up down out off again further once each other both
few many much own same only just too""".split())

# Part-of-speech preference according to node type (first level of the tree)
_POS = {"biological": ("n",), "non_biological": ("n",), "action": ("v", "n"), "quality": ("a", "s", "r", "n"),
        "pronoun": ()}


def _senses(word, ntype):
    out = []
    for pos in _POS.get(ntype, ("n", "v", "a", "s", "r")):
        out.extend(wn.synsets(word.replace(" ", "_"), pos)[:2])
    return out[:MAX_SENSES]


def _freq(lemma):
    return -lemma.count(), lemma.name().lower()


def _lemmas(synsets):
    """Lemmas of a list of synsets, ordered by frequency (deterministic)."""
    ls = [l for s in synsets for l in s.lemmas()]
    return [l.name().lower().replace("_", " ") for l in sorted(ls, key=_freq)]


def _morph(tok):
    for pos in ("n", "v", "a"):
        m = wn.morphy(tok, pos)
        if m:
            return m
    return tok


def derive(nodes):
    """nodes: {lowercase_word: node}. Returns {word: (related_wn, opposites)}.

    related_wn = [[word, via], ...]   opuestos = [word, ...]
    """
    out = {}
    for word, node in nodes.items():
        if node.get("type") == "pronoun":
            out[word] = ([], [])
            continue
        senses = _senses(word, node["type"])
        found = []                                   # (via, word)

        def push(via, synsets):
            for w in _lemmas(synsets):
                if w != word and w in nodes:
                    found.append((via, w))

        for s in senses:
            push("is-a", s.hypernyms() + s.instance_hypernyms())
            push("kind-of", s.hyponyms())
            push("part", s.part_meronyms() + s.member_meronyms() + s.substance_meronyms()
                 + s.part_holonyms() + s.member_holonyms() + s.substance_holonyms())
            push("similar", s.similar_tos() + s.also_sees() + s.verb_groups() + s.entailments())
            for l in s.lemmas():
                if l.name().lower().replace("_", " ") == word:
                    for d in sorted(l.derivationally_related_forms(), key=lambda x: x.name()):
                        w = d.name().lower().replace("_", " ")
                        if w != word and w in nodes:
                            found.append(("derived", w))
            # gloss: content words that are nodes, in order of appearance
            for tok in re.findall(r"[a-z]+", s.definition().lower()):
                if len(tok) < 3 or tok in _STOP:
                    continue
                w = _morph(tok)
                if w != word and w in nodes:
                    found.append(("definition", w))

        related, seen, per_via = [], set(), {}
        for via, w in found:
            if w in seen or per_via.get(via, 0) >= CAP_VIA[via] or len(related) >= CAP_TOTAL:
                continue
            seen.add(w)
            per_via[via] = per_via.get(via, 0) + 1
            related.append([w, via])

        # opposites: direct antonyms of the lemma and, for satellite adjectives, those of the head adjective
        opp = []
        for s in senses:
            cands = [l for l in s.lemmas() if l.name().lower().replace("_", " ") == word]
            heads = [s] if s.pos() != "s" else s.similar_tos()
            for l in cands:
                for a in l.antonyms():
                    opp.append(a.name().lower().replace("_", " "))
            if s.pos() == "s":
                for h in heads:
                    for hl in h.lemmas():
                        for a in hl.antonyms():
                            opp.append(a.name().lower().replace("_", " "))
        opposites = []
        for w in opp:
            if w != word and w in nodes and w not in opposites:
                opposites.append(w)
        out[word] = (related, opposites[:4])
    return out
