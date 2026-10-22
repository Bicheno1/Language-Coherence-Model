"""Responder: \"WHAT DO YOU THINK ABOUT DUCKS?\"  ->  \"I think ducks are lovely. They are birds and can swim, fly and walk.\"

How the question is read (everything visible with `trace`):
  * PERSPECTIVE (lcm.deixis): the asker's 'you' is, for the LCM, 'I' (Imposition): the system is the one answering,
    so the answer comes out in the first person. The user's 'me' would be 'you' (Adaptation).
  * MODE (lcm.tense): a question lives in possibility (North); it shifts the position of the verb ('think') upward.
    An exclamation would live in the South. The answer closes that possibility with an assertion.
  * TOPIC: the noun being asked about ('about ducks', 'like ducks'). It is anchored to the ANIMAL, not to the verb 'to duck'.

Where each part of the answer comes from (and what it is NOT):
  * WHAT IT KNOWS (type and abilities): the closest ancestor in the WordNet tree that is a node ('bird') and the verbs that
    the seed endorses for that species or that it inherits from its ancestor (swim, fly, walk). It is knowledge from the tables.
  * ITS DISPOSITION (I like / I don't like): it is NOT data. SentiWordNet is no good for giving opinions on concrete nouns
    (it gives 'dog' a negative sentiment because of the insult slang, 0.0 to 'war' and 'poison', positive to 'disease', and 0.0 to all
    verbs). It is an explicit, editable RULE:
        disposicion=\"amable\"  (default)  ANIMALS are liked unless the tables mark them as strongly negative
                              (sent − sent+ >= 0.5). Only for animals: in the first version it applied to everything and it came out
                              \"I think a war is friendly\". For everything else the neutral rule applies.
        neutral rule          (disposicion=\"neutral\" for everything, and always for what is not an animal) no stance if the
                              net sentiment is < 0.25 in absolute value: \"I don't have a strong opinion about X\"
  * THE ADJECTIVE: it is DRAWN (lcm.sampling: seed, temperature, top-k) among the opinion adjectives of the manual seed with the
    same sign, weighted by closeness to the topic in the plane. It is not a belief: it is a draw over a short list.
  * PEOPLE: about a group of people, or about 'you'/'me', the system does NOT give an opinion (a randomly generated judgment about nationalities,
    religions or the asker is not acceptable). It says it does not give opinions.

Questions about the LCM ITSELF (\"how are you?\", \"are you happy?\"): the 'you' is the system, which for the LCM is 'I' (Imposition), so it
answers in the first person about node I. NOTE: \"I am fine\" is NOT a real state, it is the same 'amable' disposition above applied to
node I (disposicion=\"neutral\" answers that it has no state to report). The state adjectives are a hand-written list.

Question constructions: OPINION (think/feel/believe… about X), LIKING (like/love/hate… X), STATE (\"how are you?\", \"are you happy?\").
\"Who/what are you?\" returns an error: the tables have no description of the system itself. Any other question returns `error`. English only. It needs nltk (WordNet) and lemminflect.
"""
import math
from dataclasses import dataclass, field

from .sampling import Sampler
from .narrate import (_ancestors, _protagonist_verbs, _closeness, _article, STORY_WORDS, narrate, stories)
from .arithmetic import solve_text
from .tense import shift
from .walk import target_of, R_MAX

OPINION_VERBS = {"think", "feel", "believe", "reckon", "say"}
LIKE_VERBS = {"like", "love", "enjoy", "adore", "prefer"}
DISLIKE_VERBS = {"hate", "dislike", "despise"}
LIVING_ONLY = {"friendly", "cute", "happy", "sad", "charming"}       # adjectives not said of a thing ('a war is friendly')
NEGATIVE_THRESHOLD = 0.5             # 'amable' (friendly): only strongly negative things go down badly
NEUTRAL_THRESHOLD = 0.25              # 'neutral': below this there is no stance
MAX_ABILITIES = 3
EMPTY_ANCESTORS = {"entity", "object", "organism", "whole", "physical entity", "abstraction", "thing", "being", "living thing",
                    "causal agent", "artifact", "matter", "unit", "instrumentality", "relation", "attribute", "psychological feature",
                    "event", "act", "group", "communication", "message", "person", "individual", "human", "someone"}
HUMAN_GROUPS = {"people", "folk", "folks", "humanity", "mankind", "human", "humans", "public", "crowd", "family", "community",
                  "society", "nation", "race", "ethnic", "tribe", "population", "citizen", "citizens", "men", "women", "children"}
NO_OPINION = "I don't judge people, and I don't have an opinion about them. Ask me about something else, or ask for a story!"


@dataclass
class ReplySentence:
    text: str
    nodes: list = field(default_factory=list)
    trace: list = field(default_factory=list)


@dataclass
class Reply:
    input_text: str = ""
    kind: str = None                  # opinion | liking | story
    phrases: list = field(default_factory=list)
    target: tuple = (0.0, 0.0)
    seed: int = None
    stance: str = None               # pos | neg | neutral
    error: str = None

    @property
    def text(self):
        return " ".join(f.text for f in self.phrases)

    @property
    def trace(self):
        return [z for f in self.phrases for z in f.trace]

    @property
    def coherence(self):
        ds = [math.hypot(n["x"] - self.target[0], n["y"] - self.target[1]) for f in self.phrases for n in f.nodes]
        return round(1.0 - min(1.0, sum(ds) / len(ds) / R_MAX), 3) if ds else 0.0


# ── pieces ─────────────────────────────────────────────────────────────────────────────────────────────────
def _is_person(node, lemma_text):
    """Is the topic a person, a group of people or a pronoun? (No opinion is given about that.) It looks at the node's category, the
    words for human groups and the WordNet ancestors: the category alone let 'people' through."""
    return (node.get("category") == "person" or node.get("type") == "pronoun" or lemma_text in HUMAN_GROUPS
            or node["word"].lower() in HUMAN_GROUPS or bool({"person", "people", "social group"} & set(_ancestors(lemma_text))))

def _join_words(words):
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _plural(lemma_text):
    from lemminflect import getInflection
    return (getInflection(lemma_text, "NNS") or (lemma_text + "s",))[0]


def _self_reference(frame):
    """Is the question about the LCM itself? The asker's 'you' is 'I' for the LCM (lcm.deixis)."""
    sd = frame.displaced_subject
    return frame.form == "question" and sd is not None and sd["word"] == "i"


def _copula_i(tense):
    """'be' with subject I (morphology): am / was / will be."""
    return {"present": "am", "past": "was", "future": "will be"}[tense]


def _reply_about_self(tables, frame, r, seed, temperature, top_k, disposition):
    yo = frame.displaced_subject
    r.target = target_of(frame) or (yo["x"], yo["y"])
    trace = [f"perspective: '{'you' if frame.subject is None else frame.subject.text}' in your question is 'i' for the LCM "
             f"({yo['category']}, x={yo['x']:+.0f}, y={yo['y']:+.0f}): the question is about me, I answer in the first person"]
    if frame.unknown in ("who", "what") and frame.action is not None and frame.action.lemma_text == "be":
        r.error = "I have no description of myself in the tables (the LCM has no node of its own except the pronoun I): I cannot answer who or what I am"
        return r
    be = _copula_i(frame.tense if frame.tense in ("past", "present", "future") else "present")
    if frame.quality is not None and frame.unknown is None:                  # "Are you happy?" (with 'why' NO: it presupposes the state)
        q = frame.quality.node
        if q is None:
            r.error = f"I do not know '{frame.quality.text}': it is not a node of the tables, I do not know whether I am one"
            return r
        sign = q.get("state") or q.get("opinion")
        r.kind = "state"
        r.stance = sign or "neutral"
        if sign == "pos":
            text, why = f"Yes, I {be} {q['word']}.", "marked positive: the friendly disposition is declared this way"
        elif sign == "neg":
            text, why = f"No, I {be} not {q['word']}.", "marked negative: the friendly disposition denies it"
        else:
            text, why = f"I don't know if I {be} {q['word']}.", "no sign in the tables: I neither affirm nor deny"
        r.phrases.append(ReplySentence(text, [yo, q], trace + [f"quality you are asking about: '{q['word']}' ({why})"]))
        return r
    if frame.unknown == "how" and frame.action is not None and frame.action.lemma_text in ("be", "do", "feel", "go"):   # "How are you?"
        r.kind = "state"
        if disposition == "neutral":
            r.stance = "neutral"
            r.phrases.append(ReplySentence("I don't have a state to report.", [yo], trace + ["'neutral' disposition: no state to report"]))
            return r
        randomize = Sampler(seed, temperature, top_k)
        pool = [n for n in tables["concepts"].nodes if n.get("state") == "pos"]
        adj, p = randomize.choose([(n, _closeness(n, (yo["x"], yo["y"])) * (0.2 + n["sent_pos"])) for n in pool])
        r.seed, r.stance = randomize.seed, "pos"
        verb = "feel" if frame.action.lemma_text == "feel" else be
        r.phrases.append(ReplySentence(f"I {verb} {adj['word']}.", [yo, adj], trace + [
            f"state: {adj['word']}  (DRAW among the positive state adjectives by closeness to node I; T={randomize.temperature:g}, "
            f"seed={randomize.seed}; p={p:.1%}). It is not a real state: it is the friendly disposition applied to I"]))
        return r
    r.error = ("I cannot answer that question about myself" + (f" (with '{frame.unknown}')" if frame.unknown else "")
               + ": only 'how are you?' and 'are you X?' about my state")
    return r


def _question_type(frame):
    lemma_text = frame.action.lemma_text if frame.action else None
    if frame.form != "question" and frame.mode not in ("interrogative", "request"):
        return None
    if lemma_text in OPINION_VERBS and (frame.about_topic or frame.topic):
        return "opinion"
    if (lemma_text in LIKE_VERBS or lemma_text in DISLIKE_VERBS) and frame.topic:
        return "liking"
    return None


def _stance(node, disposition):
    """('pos' | 'neg' | 'neutral', net sentiment of the topic)."""
    net = node["sent_pos"] - node["sent_neg"]
    if disposition == "friendly" and node.get("type") == "biological" and node.get("category") == "animal":
        return ("neg" if net <= -NEGATIVE_THRESHOLD else "pos"), net
    return ("neutral" if abs(net) < NEUTRAL_THRESHOLD else "pos" if net > 0 else "neg"), net


def _adjective(tables, topic, stance, randomize):
    """(adjective_node, probability). Draw among the opinion adjectives of the same sign, weighted by closeness to the topic."""
    alive = topic.get("type") == "biological"
    pool = [n for n in tables["concepts"].nodes if n.get("opinion") == stance and (alive or n["word"] not in LIVING_ONLY)]
    cands = [(n, _closeness(n, (topic["x"], topic["y"])) * (0.2 + abs(n["sent_pos"] - n["sent_neg"]))) for n in pool]
    return randomize.choose(cands)


def _knowledge(tables, node, lemma_text, plural):
    """(sentence | None, nodes, trace): 'They are birds and can swim, fly and walk.' What the tables know about the topic."""
    is_alive = node.get("type") == "biological"
    anc = None
    for a in _ancestors(lemma_text):
        if a in EMPTY_ANCESTORS or a == lemma_text:
            continue
        hit = next((h for _, h in tables.find(a) if h.get("type") != "quality"
                    and (h.get("type") == "biological") == is_alive), None)
        if hit:
            anc = hit
            break
    vs, inherited = _protagonist_verbs(tables, node) if is_alive else ([], None)      # abilities belong to living beings ('danger can run' does not)
    target = (node["x"], node["y"])
    vs = sorted(vs, key=lambda t: (-(t[1] * _closeness(t[0], target)), t[0]["word"]))[:MAX_ABILITIES]
    verbs = [v["word"] for v, _ in vs]
    subject, be_form = ("They", "are") if plural else ("It", "is")
    parts, trace = [], []
    if anc:
        w = anc["word"]
        parts.append(f"{be_form} {_plural(w) if plural else _article(w) + ' ' + w}" if is_alive
                      else f"{be_form} a kind of {w}")
        trace.append(f"type: {w}  (closest ancestor of '{lemma_text}' in the WordNet tree that is a node)")
    if verbs:
        parts.append("can " + _join_words(verbs))
        trace.append(f"abilities: {', '.join(verbs)}  ({'inherited from ' + inherited if inherited else "manual seed of the topic"})")
    if not parts:
        return None, [], ["knowledge: the tables know nothing more about this topic"]
    text = subject + " " + (" and ".join(parts) if len(parts) == 2 and verbs and anc else parts[0]) + "."
    return text, ([anc] if anc else []) + [v for v, _ in vs], trace


# ── API ────────────────────────────────────────────────────────────────────────────────────────────────────
def reply_to_question(tables, frame, seed=None, temperature=1.0, top_k=None, disposition="friendly"):
    r = Reply(input_text=frame.input_text)
    if disposition not in ("friendly", "neutral"):
        raise ValueError("disposition must be 'friendly' or 'neutral'")
    if _self_reference(frame) and _question_type(frame) is None:
        return _reply_about_self(tables, frame, r, seed, temperature, top_k, disposition)
    r.kind = _question_type(frame)
    if r.kind is None:
        r.error = ("I only answer questions (OPINION 'what do you think about X?', LIKING 'do you like X?', STATE 'how are you?') and story requests; this is not a question I can answer" if frame.form == "question" else
                   "I only answer questions and story requests; a statement, a greeting or a thank-you does not have an answer yet")
        return r
    if disposition not in ("friendly", "neutral"):
        raise ValueError("disposition must be 'friendly' or 'neutral'")
    role_obj = (frame.about_topic if r.kind == "opinion" and frame.about_topic else frame.topic)
    if role_obj is None or role_obj.node is None:
        r.error = f"I do not know '{role_obj.text if role_obj else '?'}': it is not a node of the tables, I cannot give an opinion about that"
        return r
    node = role_obj.node
    r.target = target_of(frame) or (node["x"], node["y"])

    # trace of perspective and mode (what the user asked for)
    trace0 = []
    if frame.displaced_subject is not None:
        trace0.append(f"perspective: '{frame.subject.text}' in your question is '{frame.displaced_subject['word']}' for the LCM "
                      f"({frame.displaced_subject['category']}, x={frame.displaced_subject['x']:+.0f}): I answer")
    if frame.form and frame.action and frame.action.node:
        base, displaced = frame.action.node, frame.displaced_action
        trace0.append(f"mode: '{frame.action.lemma_text}' rises from y={shift(base, frame.tense)['y']:+.1f} to y={displaced['y']:+.1f} "
                      f"for being {frame.form} (possibility = North)")

    if role_obj.lemma_text in ("yourself", "you", "your"):
        r.error = "I have no description of myself in the tables: I cannot give an opinion about myself"
        return r
    # topics NOT to give an opinion on
    if _is_person(node, role_obj.lemma_text):
        r.phrases.append(ReplySentence(NO_OPINION, [], trace0 + [f"topic '{role_obj.lemma_text}' is a person or a pronoun ({node.get('category')}): no opinion"]))
        r.stance = "neutral"
        return r

    plural = role_obj.plural
    lemma_text = role_obj.lemma_text
    # the phrase as the user said it: 'ducks', 'a duck', 'the duck', 'war' (without an article none is invented)
    np_ = role_obj.text.lower() if plural else (f"{role_obj.det} {lemma_text}" if role_obj.det else lemma_text)
    randomize = Sampler(seed, temperature, top_k)
    stance, net = _stance(node, disposition)
    r.stance = stance
    trace_p = trace0 + [f"topic: '{node['word']}' ({node['category']}, cell {node['row']},{node['col']}); net sentiment in the "
                        f"tables {net:+.2f}; disposition '{disposition}' -> stance {stance}"]

    adj = None
    if stance != "neutral":
        adj, p = _adjective(tables, node, stance, randomize)
        trace_p.append(f"adjective: {adj['word']}  (DRAW among the '{stance}' adjectives; T={randomize.temperature:g}, "
                       f"seed={randomize.seed}; came out with p={p:.1%})")
        r.seed = randomize.seed
    they = "they" if plural else "it"
    be_form = "are" if plural else "is"

    if r.kind == "opinion":
        if adj:
            first = f"I think {np_} {be_form} {adj['word']}."
        else:
            first = f"I don't have a strong opinion about {np_}."
    else:  # liking
        verb = frame.action.lemma_text
        affirms = (stance == "pos") == (verb in LIKE_VERBS)
        if stance == "neutral":
            first = f"I don't have a strong feeling about {np_}."
        else:
            first = f"Yes, I {verb} {np_}." if affirms else f"No, I don't {verb} {np_}."
            first += f" I think {they} {be_form} {adj['word']}."
    r.phrases.append(ReplySentence(first, [node] + ([adj] if adj else []), trace_p))

    text, nodes, trace_c = _knowledge(tables, node, lemma_text, plural)
    if text:
        r.phrases.append(ReplySentence(text, nodes, trace_c))
    return r


def reply(tables, frame, n_sentences=4, **kw):
    """Single entry point: returns a LIST of Reply. Stories ('tell me a story') -> lcm.narrate; opinion or liking
    questions -> reply_to_question. `kw`: seed, temperature, top_k, disposition (and randomize for stories)."""
    ar = solve_text(frame.input_text)                         # arithmetic exercise -> lcm.arithmetic (if it is not one, returns None and the normal path continues)
    if ar is not None:
        return [Reply(input_text=frame.input_text, kind="arithmetic", phrases=[ReplySentence(ar.text, [], ar.trace)])]
    topic = frame.topic
    is_story = topic is not None and topic.node is not None and (topic.lemma_text in STORY_WORDS or topic.node["word"] in STORY_WORDS)
    if is_story and frame.form != "question":
        if frame.negated:                                   # "Don't tell me a story": the request is cancelled, nothing is told
            return [Reply(input_text=frame.input_text, kind="story", error="the request is negated: I tell no story")]
        out = []
        for rel in stories(tables, frame, n_sentences, **{k: v for k, v in kw.items() if k in ("seed", "temperature", "top_k", "randomize")}):
            x = Reply(input_text=frame.input_text, kind="story", target=rel.target, seed=rel.seed, error=rel.error)
            x.phrases = [ReplySentence(o.text, o.nodes, o.trace) for o in rel.sentences]
            out.append(x)
        return out
    return [reply_to_question(tables, frame, **{k: v for k, v in kw.items() if k in ("seed", "temperature", "top_k", "disposition")})]
