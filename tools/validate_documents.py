"""Validates the AI's answers (sentences and per-word cards) and measures how much it hallucinates.

Input: the prompts .jsonl (tools/prepare_ai_batch.py) and a responses .jsonl, one line per call:
    {\"id\": \"<call id>\", \"response\": \"<raw text the model returned>\"}
Output (in --output, by default next to the responses): aceptadas.jsonl, rechazadas.jsonl (with the reason), fichas.jsonl, informe.json.

What it checks on each SENTENCE (in this order; the first failure rejects it):
  invalid_json · different_lemma · invalid_category · no_final_punctuation · longitud (5 to 20 words) · digitos · banned_word ·
  entidad (spaCy: person, group, place, organization, product…: covers nationalities and religions without listing them) · proper_name ·
  missing_word (the target word does not appear) · different_pos (it appears, but with ANOTHER part of speech: the cheapest indicator of
  sense HALLUCINATION) · inconsistent_category (a 'question' without '?', an 'exclamation' without '!', or a '?' in another) · duplicada ·
  near_duplicate (within the same word: they share a sequence of 5 or more words, or Jaccard of 4-grams >= 0.5).
And on each CARD: schema, definition <= 30 words without the target word, one-word synonyms and antonyms, no banned words or duplicates.

It is an automatic JUDGE: spaCy is sometimes wrong (~3 % of tags), which is why the batch's acceptance criterion is a percentage, not 100 %.

Usage:  python tools/validate_documents.py --prompts ai_corpus/piloto/prompts.jsonl --replies ai_corpus/piloto/replies.jsonl
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CATEGORIES = {"past", "present", "future", "negative", "question", "exclamation", "imperative", "connector", "free"}
POS_OK = {"noun": {"NOUN"}, "verb": {"VERB", "AUX"}, "adjective": {"ADJ"}, "adverb": {"ADV"}, "numeral": {"NUM"},
          "preposition": {"ADP"}, "determiner": {"DET"}, "pronoun": {"PRON"}, "auxiliary/modal": {"AUX", "VERB"},
          "particle": {"PART", "ADP", "ADV"}, "conjunction": {"CCONJ", "SCONJ"}, "negation (particle)": {"PART", "ADV"}}
BANNED_ENTITIES = {"PERSON", "NORP", "GPE", "ORG", "FAC", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE"}
MIN_PAL, MAX_PAL, JACCARD, MIN_SHARED = 5, 20, 0.5, 2     # MIN_SHARED = shared 4-grams (2 = a sequence of 5 words)
_NLP = None


def nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def load_banned(filepath=None):
    filepath = filepath or os.path.join(ROOT, "data", "banned_words.txt")
    try:
        return {l.strip().lower() for l in open(filepath, encoding="utf-8") if l.strip() and not l.startswith("#")}
    except OSError:
        return set()


def parse_json(raw_text):
    """The JSON of the raw answer, tolerating code blocks and surrounding text. None if there is no valid JSON."""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw_text or "").strip(), flags=re.M).strip()
    for cand in (s, s[s.find("{"): s.rfind("}") + 1] if "{" in s else ""):
        try:
            d = json.loads(cand)
            return d if isinstance(d, dict) else None
        except (ValueError, TypeError):
            continue
    return None


def _norm(t):
    return re.sub(r"[^a-z' ]", "", t.lower()).strip()


def _shingles(t, k=4):
    p = _norm(t).split()
    return {tuple(p[i:i + k]) for i in range(max(1, len(p) - k + 1))}


def validate_sentences(api_call, data, banned_words, seen_global, seen_lemma):
    """-> (accepted, rejected) of a 'phrases' call. `data` is already the JSON. Updates the duplicate sets."""
    lemma_text, conn_class = api_call["lemma"].lower(), api_call["class"]
    base = {"id": api_call["id"], "lemma": api_call["lemma"], "pos": api_call["pos"], "class": conn_class}
    accepted, rejected_items = [], []

    def no(item, reason):
        rejected_items.append({**base, "cat": (item or {}).get("cat") if isinstance(item, dict) else None,
                     "text": (item or {}).get("text") if isinstance(item, dict) else str(item), "reason": reason})
    if str(data.get("lemma", "")).lower() != lemma_text:
        for it in data.get("sentences") or []:
            no(it, "different_lemma")
        return accepted, rejected_items
    items = [it for it in (data.get("sentences") or [])]
    texts = [str(it.get("text", "")).strip() if isinstance(it, dict) else "" for it in items]
    docs = list(nlp().pipe(texts)) if texts else []
    for it, text, doc in zip(items, texts, docs):
        if not isinstance(it, dict) or it.get("cat") not in CATEGORIES:
            no(it, "invalid_category"); continue
        if not text or text[-1] not in ".?!":
            no(it, "no_final_punctuation"); continue
        n = sum(1 for t in doc if t.is_alpha)
        if not MIN_PAL <= n <= MAX_PAL:
            no(it, "length"); continue
        if any(ch.isdigit() for ch in text):
            no(it, "digits"); continue
        if any(t.lower_ in banned_words for t in doc):
            no(it, "banned_word"); continue
        if any(e.label_ in BANNED_ENTITIES for e in doc.ents):
            no(it, "entity"); continue
        if any(t.pos_ == "PROPN" for t in doc):
            no(it, "proper_name"); continue
        cand = [t for t in doc if t.lemma_.lower() == lemma_text or t.lower_ == lemma_text]
        if not cand:
            no(it, "missing_word"); continue
        if not any(t.pos_ in POS_OK.get(conn_class, {t.pos_}) for t in cand):
            no(it, "different_pos"); continue
        if it["cat"] != "free" and ((it["cat"] == "question") != text.endswith("?") or (it["cat"] == "exclamation" and not text.endswith("!"))):
            no(it, "inconsistent_category"); continue
        k = _norm(text)
        if k in seen_global:
            no(it, "duplicate"); continue
        sh = _shingles(text)
        if any(len(sh & o) >= MIN_SHARED or len(sh & o) / max(1, len(sh | o)) >= JACCARD for o in seen_lemma[lemma_text]):
            no(it, "near_duplicate"); continue
        seen_global.add(k)
        seen_lemma[lemma_text].append(sh)
        accepted.append({**base, "cat": it["cat"], "text": text})
    return accepted, rejected_items


def validate_card(api_call, data, banned_words):
    """-> (clean_card | None, reasons[]). Bad list elements are cleaned out and counted as reasons."""
    lemma_text = api_call["lemma"].lower()
    why = []
    if str(data.get("lemma", "")).lower() != lemma_text:
        return None, ["different_lemma"]
    d = str(data.get("definition") or "").strip()
    if d and (len(d.split()) > 30 or re.search(rf"\b{re.escape(lemma_text)}\b", d.lower())):
        why.append("invalid_definition"); d = ""

    def clean(items, name):
        out = []
        for w in (items if isinstance(items, list) else []):
            w = str(w).strip().lower()
            if not re.fullmatch(r"[a-z]+(-[a-z]+)?", w):
                why.append(f"{name}_is_not_a_word"); continue
            if w == lemma_text or w in out:
                why.append(f"{name}_repeated"); continue
            if w in banned_words:
                why.append(f"{name}_banned"); continue
            out.append(w)
        return out[:8]
    sin, ant = clean(data.get("synonyms"), "synonym"), clean(data.get("antonyms"), "antonym")
    if not d and not sin and not ant:
        return None, why + ["empty_card (the model declared it was unsure)"]
    return {"id": api_call["id"], "lemma": api_call["lemma"], "pos": api_call["pos"], "class": api_call["class"], "definition": d,
            "synonyms": sin, "antonyms": ant}, why


def validate_batch(calls, replies, banned_words=None):
    """calls: list of dicts from prompts.jsonl · replies: {id: raw_text}. -> (accepted, rejected, cards, report)."""
    banned_words = load_banned() if banned_words is None else banned_words
    seen_global, seen_lemma = set(), defaultdict(list)
    accepted, rejected_items, cards = [], [], []
    reasons, no_reply, bad_json, empty_rows, card_reasons = Counter(), [], [], 0, Counter()
    requested = defaultdict(int)
    for c in calls:
        if c["type"] in ("phrases", "free_calls"):
            requested[(c["lemma"], c["pos"])] += c["n"]
        raw_text = replies.get(c["id"])
        if raw_text is None:
            no_reply.append(c["id"]); continue
        data = parse_json(raw_text)
        if data is None:
            bad_json.append(c["id"]); continue
        if c["type"] in ("phrases", "free_calls"):
            if not data.get("sentences"):
                empty_rows += 1
            a, r = validate_sentences(c, data, banned_words, seen_global, seen_lemma)
            accepted += a; rejected_items += r
            reasons.update(x["reason"] for x in r)
        else:
            f, m = validate_card(c, data, banned_words)
            card_reasons.update(m)
            if f:
                cards.append(f)
    grouped = defaultdict(int)
    by_cat = defaultdict(Counter)
    openings = defaultdict(Counter)
    for a in accepted:
        grouped[(a["lemma"], a["pos"])] += 1
        by_cat[(a["lemma"], a["pos"])][a["cat"]] += 1
        openings[(a["lemma"], a["pos"])][a["text"].split()[0].lower()] += 1
    total_ev = len(accepted) + len(rejected_items)
    to_replenish = sorted(f"{l}|{p}: {grouped[(l, p)]}/{n}" for (l, p), n in requested.items() if grouped[(l, p)] < 0.8 * n)
    report = {
        "calls": len(calls), "no_reply": len(no_reply), "invalid_json": len(bad_json), "calls_with_empty_list": empty_rows,
        "evaluated_phrases": total_ev, "accepted": len(accepted), "rejected": len(rejected_items),
        "pct_accepted": round(100 * len(accepted) / total_ev, 1) if total_ev else None,
        "rejections_by_reason": dict(reasons.most_common()),
        "pct_different_pos": round(100 * reasons["different_pos"] / total_ev, 1) if total_ev else None,
        "accepted_cards": len(cards), "card_issues": dict(card_reasons.most_common()),
        "words_below_80pct": to_replenish,
        "most_repeated_opening_pct_max": round(100 * max((c.most_common(1)[0][1] / sum(c.values()) for c in openings.values()), default=0), 1),
        "ids_no_reply": no_reply[:20], "ids_invalid_json": bad_json[:20],
    }
    return accepted, rejected_items, cards, report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)
    calls = [json.loads(l) for l in open(a.prompts, encoding="utf-8") if l.strip()]
    resp = {}
    for l in open(a.replies, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            resp[d["id"]] = d["response"]
    accepted, rejected_items, cards, inf = validate_batch(calls, resp)
    out_dir = a.output or os.path.dirname(os.path.abspath(a.replies))
    for name, rows in (("accepted.jsonl", accepted), ("rejected.jsonl", rejected_items), ("cards.jsonl", cards)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump(inf, open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(inf, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
