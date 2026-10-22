"""Prepares the PROMPTS for generating, with an AI (e.g. Gemini), the corpus of sentences per word.

Reads data/vocabulary_sheet.xlsx and writes ai_corpus/<batch>/prompts.jsonl: one line per CALL, with the full prompt, and manifest.json.
Three kinds of call per word:
  * 'frases' (sentences): N sentences in total (by default 40 for content words, 60 for function words), in calls of 10, each with two different contexts and
    a quota per category (past, present, future, negative, question, exclamation, imperative, connector).
  * 'libres' (free): 10 sentences WITHOUT quotas (by default): the natural use of the word. Needed because the quotas equalize past, present and future
    and thus erase each word's natural tendency (yesterday, will…), which is evidence that the tense and mode axes need.
  * 'ficha' (card): definition, synonyms and antonyms (the antonyms give the NEGATIVE pins).
Deterministic: same sheet + same seed = same prompts. The model's answer is saved as {\"id\": ..., \"response\": \"<raw text>\"}
in a .jsonl and validated with tools/validate_documents.py.

Usage:  python tools/prepare_ai_batch.py --pilot 60 --batch piloto
        python tools/prepare_ai_batch.py --batch completo
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from string import Template

from openpyxl import load_workbook

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VERSION = "1"

POS_NAMES = {"noun": "noun", "verb": "verb", "adjective": "adjective", "adverb": "adverb", "numeral": "numeral",
              "preposition": "preposition", "determiner": "determiner", "pronoun": "pronoun", "auxiliary/modal": "auxiliary or modal verb",
              "particle": "particle", "conjunction": "conjunction", "negation (particle)": "negation particle"}
FUNCTION_NODES = {"preposition", "determiner", "pronoun", "auxiliary/modal", "particle", "conjunction", "negation (particle)"}
EXTRA_FUNCTIONAL = {
    "preposition": "This is a function word. Show the different kinds of nouns and verbs it combines with, and the different relations it expresses (place, time, direction, cause, manner).",
    "determiner": "This is a function word. Show it before different kinds of nouns (countable, uncountable, singular, plural) and in different positions.",
    "pronoun": "This is a function word. Show it as subject, object and after prepositions where it is possible, with different verbs.",
    "auxiliary/modal": "This is a function word. Show it with many different main verbs and in statements, questions and negatives.",
    "particle": "This is a function word. Show its typical patterns with many different verbs and contexts.",
    "conjunction": "This is a function word. Show it joining different kinds of words and clauses.",
    "negation (particle)": "This is a function word. Show it with different verbs and auxiliaries.",
}
CATEGORIES = [   # (label, proportion, description)
    ("past", 0.20, "a statement in the past tense"),
    ("present", 0.20, "a statement in the present tense"),
    ("future", 0.15, "a statement about the future (will, going to, plans)"),
    ("negative", 0.10, "a negative statement (not, never, no, can't...), in any tense"),
    ("question", 0.125, "a question ending in ?"),
    ("exclamation", 0.075, "an exclamation ending in !"),
    ("imperative", 0.075, "a command or a polite request"),
    ("connector", 0.075, "a longer sentence joining two ideas with a connector such as because, although, then, but, so, if, when, while"),
]
DOMAINS = ["everyday conversation at home", "a children's story", "a news report", "step-by-step instructions", "a science or nature explanation",
            "a personal diary entry", "a workplace email or meeting", "a street or travel scene", "a sports or hobby situation",
            "a short exchange between two people", "a history or school lesson", "a humorous anecdote"]

SENTENCES_PROMPT = Template("""You are generating example sentences for a linguistic database.

TARGET WORD: "$lemma"
PART OF SPEECH: $pos_name
$note

Write exactly $n different English sentences that use the target word with that part of speech and sense. You may use any inflected form of the word (for example "ran" for "run"), but it must clearly work as a $pos_name.

REQUIRED MIX. Give each sentence a "cat" label, using exactly these counts:
$quotas

SETTINGS. Spread the sentences over these two settings, about half each: (1) $dom1; (2) $dom2.

STYLE RULES
- Natural, everyday English that a person would really say or write. One sentence each, 6 to 16 words, ending in . ? or !
- Vary the openings and the structures. Do not start more than two sentences with the same word. No two sentences may share a run of four or more words.
- No lists, definitions, quotations, titles or dialogue tags.
- No proper names of people, places, organizations, brands or titles. No digits.
- Nothing offensive, sexual, graphically violent, discriminatory or political. Keep it neutral and harmless.
- Use only the sense given. If you are not sure you know this word with this part of speech and sense, return an empty "sentences" list instead of guessing. Do not invent facts.
- Do not repeat the same idea in different words.

OUTPUT. Return only valid JSON, with no comments or code fences, exactly in this shape:
{"lemma": "$lemma", "pos": "$pos_name", "sentences": [{"cat": "<label>", "text": "<sentence>"}]}
""")

FREE_PROMPT = Template("""You are generating example sentences for a linguistic database.

TARGET WORD: "$lemma"
PART OF SPEECH: $pos_name
$note

Write exactly $n different English sentences that use the target word with that part of speech and sense. You may use any inflected form of the word, but it must clearly work as a $pos_name.

Do NOT force any tense, mood or sentence type: write whatever is most natural and typical for this word. Give every sentence the label "free" in its "cat" field.

SETTINGS. Spread the sentences over these two settings, about half each: (1) $dom1; (2) $dom2.

STYLE RULES
- Natural, everyday English that a person would really say or write. One sentence each, 6 to 16 words, ending in . ? or !
- Vary the openings and the structures. Do not start more than two sentences with the same word. No two sentences may share a run of four or more words.
- No lists, definitions, quotations, titles or dialogue tags.
- No proper names of people, places, organizations, brands or titles. No digits.
- Nothing offensive, sexual, graphically violent, discriminatory or political. Keep it neutral and harmless.
- Use only the sense given. If you are not sure you know this word with this part of speech and sense, return an empty "sentences" list instead of guessing. Do not invent facts.
- Do not repeat the same idea in different words.

OUTPUT. Return only valid JSON, with no comments or code fences, exactly in this shape:
{"lemma": "$lemma", "pos": "$pos_name", "sentences": [{"cat": "free", "text": "<sentence>"}]}
""")

CARD_PROMPT = Template("""You are writing a dictionary entry for a linguistic database.

TARGET WORD: "$lemma"
PART OF SPEECH: $pos_name
$note

Return only valid JSON, with no comments or code fences, exactly in this shape:
{"lemma": "$lemma", "pos": "$pos_name", "definition": "...", "synonyms": ["..."], "antonyms": ["..."]}

RULES
- definition: one sentence of at most 25 words, in simple common words, explaining this sense without using the target word itself.
- synonyms: 3 to 8 single common English words with the same part of speech and sense (no phrases). Fewer is better than wrong ones.
- antonyms: only true opposites in this sense, same part of speech, single common words. Use an empty list if there is no real opposite. Do not invent.
- No proper names, no offensive words or slurs, nothing outside the JSON.
- If you are not sure about this word with this part of speech and sense, return an empty string and empty lists.
""")


def quotas(n):
    """Split of n sentences by category (largest remainder method). For n = 40: 8, 8, 6, 4, 5, 3, 3, 3."""
    raw_values = [(e, p * n) for e, p, _ in CATEGORIES]
    base = {e: int(math.floor(v)) for e, v in raw_values}
    missing = n - sum(base.values())
    for e, v in sorted(raw_values, key=lambda t: (-(t[1] - math.floor(t[1])), [c[0] for c in CATEGORIES].index(t[0])))[:missing]:
        base[e] += 1
    return base


def split_quotas(quota, k):
    """Splits the quotas among k calls so that each has a similar mix (cyclic split)."""
    parts = [defaultdict(int) for _ in range(k)]
    i = 0
    for e, _, _ in CATEGORIES:
        for _ in range(quota[e]):
            parts[i % k][e] += 1
            i += 1
    return [dict(p) for p in parts]


# Words of the sheet that are NOT generated: nationalities, religions and political groups. They are not base vocabulary and the validator would reject all
# their sentences (group entity). They were found with spaCy over the sheet; if the sheet grows, repeat the search.
SKIP = {"american", "english", "christian", "soviet", "french", "british", "catholic", "german", "russian", "indian", "republican"}


def read_sheet(filepath):
    wb = load_workbook(filepath, read_only=True, data_only=True)
    ws = wb["vocabulary"]
    rows = list(ws.iter_rows(values_only=True))
    cab = {h: i for i, h in enumerate(rows[0])}
    out = []
    for r in rows[1:]:
        if r[cab["word"]] and str(r[cab["word"]]).lower() not in SKIP:
            out.append({"rank": r[cab["rank"]], "word": str(r[cab["word"]]), "class": r[cab["class"]], "pos": r[cab["pos"]],
                        "note": (r[cab["note"]] or "") if "note" in cab else ""})
    return out


def pilot(rows, n):
    """Stratified, deterministic sample: per class, words at regular frequency intervals."""
    grouped = defaultdict(list)
    for f in rows:
        grouped[f["class"]].append(f)
    total = len(rows)
    chosen = []
    for conn_class, fs in sorted(grouped.items()):
        k = max(2, round(n * len(fs) / total))
        stride = max(1, len(fs) // k)
        chosen += fs[::stride][:k]
    return sorted(chosen, key=lambda f: f["rank"])[:max(n, 1)] if len(chosen) > n else sorted(chosen, key=lambda f: f["rank"])


def build(rows, n_content=40, n_function=60, per_call=10, seed=20261004, free_calls=10):
    calls = []
    for f in rows:
        lemma_text, conn_class = f["word"], f["class"]
        pos_n = POS_NAMES.get(conn_class, conn_class)
        note = (EXTRA_FUNCTIONAL.get(conn_class, "") + (" " if conn_class in EXTRA_FUNCTIONAL and f["note"] else "") + (f"Sense hint: {f['note']}" if f["note"] else "")).strip()
        n = n_function if conn_class in FUNCTION_NODES else n_content
        k = math.ceil(n / per_call)
        rng = random.Random(f"{seed}|{lemma_text}|{f['pos']}")
        k_tot = k + (1 if free_calls else 0)
        doms = rng.sample(DOMAINS, min(len(DOMAINS), 2 * k_tot))
        while len(doms) < 2 * k_tot:
            doms += rng.sample(DOMAINS, min(len(DOMAINS), 2 * k_tot - len(doms)))
        cq = split_quotas(quotas(n), k)
        for i in range(k):
            ni = sum(cq[i].values())
            text_lines = "\n".join(f"- {e}: {cq[i][e]} ({d})" for e, _, d in CATEGORIES if cq[i].get(e))
            prompt = SENTENCES_PROMPT.substitute(lemma=lemma_text, pos_name=pos_n, note=note, n=ni, quotas=text_lines, dom1=doms[2 * i], dom2=doms[2 * i + 1])
            calls.append({"id": f"{lemma_text}|{f['pos']}|phrases|{i + 1}", "type": "phrases", "lemma": lemma_text, "pos": f["pos"], "class": conn_class,
                             "n": ni, "quotas": cq[i], "domains": [doms[2 * i], doms[2 * i + 1]], "prompt": prompt})
        if free_calls:
            calls.append({"id": f"{lemma_text}|{f['pos']}|free_calls|1", "type": "free_calls", "lemma": lemma_text, "pos": f["pos"], "class": conn_class, "n": free_calls,
                             "domains": [doms[2 * k], doms[2 * k + 1]],
                             "prompt": FREE_PROMPT.substitute(lemma=lemma_text, pos_name=pos_n, note=note, n=free_calls, dom1=doms[2 * k], dom2=doms[2 * k + 1])})
        calls.append({"id": f"{lemma_text}|{f['pos']}|card|1", "type": "card", "lemma": lemma_text, "pos": f["pos"], "class": conn_class, "n": 0,
                         "prompt": CARD_PROMPT.substitute(lemma=lemma_text, pos_name=pos_n, note=note)})
    return calls


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=os.path.join(ROOT, "data", "vocabulary_sheet.xlsx"))
    ap.add_argument("--batch", default="pilot")
    ap.add_argument("--pilot", type=int, default=0, help="only N words (stratified sample); 0 = all")
    ap.add_argument("--content-phrases", type=int, default=40)
    ap.add_argument("--function-phrases", type=int, default=60)
    ap.add_argument("--per-call", type=int, default=10)
    ap.add_argument("--free-calls", type=int, default=10, help="sentences without per-word quotas (natural use); 0 = none")
    ap.add_argument("--seed", type=int, default=20261004)
    a = ap.parse_args(argv)
    rows = read_sheet(a.sheet)
    if a.pilot:
        rows = pilot(rows, a.pilot)
    calls = build(rows, a.content_phrases, a.function_phrases, a.per_call, a.seed, a.free_calls)
    out = os.path.join(ROOT, "ai_corpus", a.batch)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "prompts.jsonl"), "w", encoding="utf-8") as f:
        for c in calls:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    phrases = sum(c["n"] for c in calls)
    man = {"prompts_version": VERSION, "seed": a.seed, "words": len(rows), "calls": len(calls), "requested_phrases": phrases,
           "content_phrases": a.content_phrases, "function_phrases": a.function_phrases, "per_call": a.per_call, "free_calls": a.free_calls,
           "sha256_prompts": hashlib.sha256(open(os.path.join(out, "prompts.jsonl"), "rb").read()).hexdigest()}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
