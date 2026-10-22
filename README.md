# Language Coherence Model (LCM)

An **associative** language model built on four-axis tables. Every word, concept, subject or connector is a node
with four integer weights (0–10) whose centroid fixes its cell in a 10 × 10 table. Nodes are linked through
`related` and `synonyms`. It derives from **Unique Host / CCM**: the geometry, the tree and the language rules are
kept; every internal process (chemistry, vitality, somatic engine) is removed.

**Version 0.5 — status:** tables, links, segmentation, decomposer and recomposer driven by value rules (stories,
opinions, likes and the system's own state, in English) + digit-by-digit arithmetic.

> **Design principle:** a minimal core of dynamic conditional rules by axis, with no templates or patches. Only
> `lcm/recompose.py` meets it; `narrate.py` and the fixed phrases in `reply.py` are templates scheduled for removal.

## The four axes

| Pole | Label | Weight | What goes there |
|---|---|---|---|
| Up | Context (We, Possibility) | `V` | future, possibility, condition, purpose, "we/us" |
| Down | World (That, logical) | `I` | past, negation, danger, closure, result |
| Right | Imposition (I, Sensory) | `Lv` | the sensory, the concrete, the speaker imposing |
| Left | Adaptation (You, structural) | `Gv` | the relational, the structural, the social |

`x = (Lv − Gv) / 2`, `y = (V − I) / 2`. Time runs vertically (past South, future North); the present is the
horizontal band (y ≈ 0).

## Quick start

```bash
pip install -r requirements.txt
python tests/test_tables.py                 # language tests, ~1-2 min
python -m lcm stats                         # summary of the three tables
python -m lcm lookup happy                  # node, position, neighbors, synonyms, related
python -m lcm cell subjects 1 5              # what is in a cell
python -m lcm parse "Can you tell me a story about a brave king?"   # roles of the sentence (needs spaCy)
python -m lcm recompose "TELL ME A STORY" --trace   # rule-based recomposer: which rule produced each word
python -m lcm decompose "TELL ME A STORY"           # roles by value (axis, tense, perspective, forms, related)
python -m lcm reply "WHAT DO YOU THINK ABOUT DUCKS?" # opinion; also "Do you like dogs?" and "How are you?"
python -m lcm story "TELL ME A STORY"         # the earlier template (being retired)
python -m lcm story "tell me a story about a pig" --trace   # and why each word was chosen
python -m lcm walk "tell me a story" 12      # node walk guided by geometry
python -m lcm path story jump                # link path between two words
```

Segmentation needs `pip install spacy && python -m spacy download en_core_web_sm` (English only for now).

```python
from lcm import load
t = load()
t.find("happy")                 # [("concepts", {...})]
t["subjects"].near("dog", 1)     # nodes in neighboring cells
t["connectors"].get("so")       # 'so (result)': a connector with two roles is two nodes
t.links("story")                # links ordered by weight: manual > concept > is-a > … > definition
t.opposites("happy"), t.mirror("fire", radius=1)   # lexical antonym / geometric mirror

from lcm.parse import parse; from lcm.walk import walk
walk(t, parse("tell me a story", t)).words          # node walk
from lcm.narrate import narrate; narrate(t, parse("tell me a story", t)).text   # the story
```

Regenerate the tables from the Unique Host data (needs the NLTK corpora):

```bash
python tools/setup_nltk.py
python tools/build_tables.py --write        # rewrites data/LCM_tables.xlsx and .json
python tests/test_tables.py --slow          # also checks that the regeneration is identical
```

## How it fits together

```
source_data/db/ (lexicon, concepts, per-axis lists)                     ─┐
tools/connectors_data.py (connectors)                                   ─┼─> tools/build_tables.py ─> data/LCM_tables.{json,xlsx}
SentiWordNet + WordNet (valence and lineage)                            ─┘                               │
                                                                                                         v
                                                                                       lcm/ (geometry, tables, parse, walk, …)
```

## Folders

| Folder | Contents |
|---|---|
| `docs/` | `ARITHMETIC.md` (arithmetic module) |
| `data/` | the three tables: `LCM_tables.xlsx` (for review, with formulas) and `LCM_tables.json` (for the code); arithmetic and punctuation data; the base-vocabulary annotation sheet |
| `lcm/` | geometry, loading, neighbors, synonyms, related, opposites (`tables`), verb tense (`tense`), segmentation (`parse`), walk (`walk`), text realization (`narrate`, template), decomposer (`decompose`), recomposer (`recompose`), responder (`reply`), perspective (`deixis`), randomness (`sampling`), provisional function words (`function_words`), arithmetic (`arithmetic`), command line |
| `tools/` | `build_tables.py`, `connectors_data.py`, `related_wn.py` (links from WordNet), `related_manual.py` (narrative seed), `build_vocabulary_sheet.py` (annotation sheet), `prepare_ai_batch.py` and `validate_documents.py` (AI-generated corpus), `enrich_with_oanc.py` (optional OANC links), `setup_nltk.py` |
| `source_data/` | the Unique Host data modules needed to regenerate the tables |
| `ai_corpus/` | pilot prompts for generating the association-network corpus with an AI (`pilot/prompts.jsonl`); responses and full batches are generated, not versioned |
| `tests/` | language tests (`test_tables.py`) and arithmetic tests (`test_arithmetic.py`) |
| `experiments/` | association and closure meter on real text (`exp_geometry.py`), connector scoring (`exp_connectors.py`), pure generator (`gen_pure.py`) and their JSON results |

## Arithmetic

`python -m lcm reply "what is 123 times 45"` -> `123 times 45 is 5535.` (type *arithmetic*). Operations are stored as related
entries (`data/LCM_arithmetic.json`); precedence is derived from those relations; calculation is done digit by digit
with fact tables. See `docs/ARITHMETIC.md`; tests: `python tests/test_arithmetic.py`.

## Status of the optional parts

The model itself only needs `lcm/` and the files in `data/` (`LCM_tables.json`, `LCM_arithmetic.json`, `connector_punctuation.json`); the Excel files are for review and are not read by the code.

**Experimental.** The base data needed to train or calibrate the model at scale is not available. The AI-corpus pipeline (`tools/prepare_ai_batch.py`, `tools/validate_documents.py`, `ai_corpus/`, `data/banned_words.txt`, `data/vocabulary_sheet.xlsx`) was an attempt to build it: only a prompt pilot exists, with no responses, and the model does not depend on it. `tools/enrich_with_oanc.py` expects bases that are not included, and `experiments/` reproduces the measurements of the paper on the Brown corpus.

## License

Licensed under the PolyForm Noncommercial License 1.0.0 (see `LICENSE` and `NOTICE`).
Some source data originates from the Unique Host project (`source_data/`).
