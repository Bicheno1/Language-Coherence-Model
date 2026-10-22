# db_axis_lexicon.py
# -----------------------------------------------------------------------------
# Lexicon classified by CARDINAL AXIS (North/South/East/West).
#
# This is a layer SEPARATE from the thematic lexicon (db_lexicon.py: animal,
# person, object, etc.). This layer classifies words by their GEOMETRIC
# FUNCTION in the model, not by their topic.
#
#   NORTH (Viability)    -> future / possibility / "will do"    | subject: we, us
#   SOUTH (Inviability)  -> past / impossibility / "done it"    | subject: that, it (impersonal)
#   EAST  (Local / I)    -> raw perception from ONE sense, no comparison between
#                            parts (direct physical property: how it feels,
#                            looks, sounds, tastes, smells — as-is, no relating)
#   WEST  (Global / You) -> relation between parts / judgment against a
#                            standard (structure, proportion, aesthetic or
#                            moral evaluation)
#
# Quick test for classifying a new quality word (East vs West):
#   Is it a single-sense raw reading, nothing compared?         -> EAST
#   Does it require comparing parts to each other or to an
#   ideal/standard?                                             -> WEST
#
# All of this is a seed / first pass, meant to be reviewed and expanded via
# synonyms (WordNet synsets, thesaurus, or manual list).
# -----------------------------------------------------------------------------


# =============================================================================
# VERTICAL AXIS — NORTH / SOUTH (tense/modality, sentence subject)
# =============================================================================

AXIS_NORTH = {
    "subjects": [
        "we", "us", "our", "ours", "ourselves",
    ],
    # "will do" — future / possibility / intention
    "modal_verbs": [
        "will", "would", "can", "could", "may", "might", "shall",
        "going to", "about to", "plan to", "intend to", "hope to",
        "aim to", "expect to", "look forward to",
    ],
    # concepts that only exist projected forward
    "concepts": [
        "hope", "expectation", "project", "goal", "plan", "dream",
        "ambition", "opportunity", "promise", "potential", "future",
        "prospect", "aspiration", "possibility", "vision", "intention",
    ],
    # interrogatives — a question opens toward possibility, not a closed fact
    "interrogatives": [
        "what", "who", "when", "where", "why", "how", "which",
        "whose", "what if",
    ],
    # affirmatives — clean, resolved agreement (no hedging, no negation)
    "affirmatives": [
        "yes", "sure", "of course", "why not", "absolutely",
        "definitely", "certainly", "i don't mind", "no problem",
        "gladly", "sounds good", "agreed",
    ],
}

AXIS_SOUTH = {
    "subjects": [
        "that", "it", "those", "them", "itself",
    ],
    # "done it" — past / impossibility / negation
    "modal_verbs": [
        "was", "were", "did", "had", "couldn't", "cannot", "can't",
        "won't", "wouldn't", "shouldn't", "didn't", "wasn't", "isn't",
        "never", "no longer", "impossible", "unable to", "failed to",
        "ended", "finished", "gave up",
    ],
    # negative auxiliary contractions — fixed forms, don't depend on the base
    # verb, so they live here instead of in the generated per-verb lexicon
    "negative_auxiliaries": [
        "isn't", "aren't", "wasn't", "weren't",
        "don't", "doesn't", "didn't",
        "haven't", "hasn't", "hadn't",
        "won't", "wouldn't",
        "can't", "cannot", "couldn't",
        "shouldn't", "mustn't", "mightn't", "shan't",
        "not",
    ],
    "concepts": [
        "failure", "loss", "end", "impossibility", "negation", "regret",
        "past", "history", "ruin", "defeat", "resignation", "dead end",
        "limit", "collapse", "downfall",
    ],
    # exclamations — a declaration slammed shut, no room left to negotiate
    "exclamations": [
        "no!", "never!", "stop!", "enough!", "how dare you!",
        "unbelievable!", "get out!", "absolutely not!", "that's it!",
    ],
}


# =============================================================================
# HORIZONTAL AXIS — EAST / WEST (qualities: raw perception vs. relation/judgment)
# =============================================================================

AXIS_EAST = {
    "subjects": [
        "I", "me", "my", "mine", "myself",
    ],
    "touch": [
        "hard", "soft", "rough", "smooth", "heavy", "light", "hot",
        "cold", "warm", "icy", "wet", "dry", "sticky", "sharp",
        "pointed", "slippery", "firm", "flexible", "rigid", "coarse",
    ],
    "sight": [
        "red", "blue", "green", "yellow", "black", "white", "gray",
        "dark", "bright", "opaque", "transparent", "big", "large",
        "small", "tiny", "tall", "short", "wide", "narrow", "round",
        "square", "curved", "straight", "thin", "thick",
    ],
    "hearing": [
        "loud", "quiet", "silent", "high-pitched", "low-pitched",
        "noisy", "shrill", "muffled", "deafening",
    ],
    "taste": [
        "sweet", "salty", "bitter", "sour", "spicy", "bland",
        "tasteless", "stale", "savory",
    ],
    "smell": [
        "fragrant", "aromatic", "smelly", "stinky", "odorless",
        "pungent", "musty",
    ],
    "perceived_motion_time": [
        "fast", "slow", "quick", "swift", "sluggish", "abrupt", "brisk",
        "sudden",
    ],
}

AXIS_WEST = {
    "subjects": [
        "you", "your", "yours", "yourself",
    ],
    "aesthetic": [
        "beautiful", "pretty", "ugly", "elegant", "harmonious",
        "symmetrical", "asymmetrical", "balanced", "unbalanced",
        "proportionate", "disproportionate", "attractive",
        "sophisticated", "vulgar", "refined", "graceful",
    ],
    "moral_value": [
        "good", "bad", "fair", "unfair", "right", "wrong", "honest",
        "dishonest", "noble", "cruel", "generous", "selfish", "loyal",
        "treacherous", "decent", "corrupt", "virtuous",
    ],
    "general_evaluation": [
        "interesting", "boring", "admirable", "mediocre", "excellent",
        "terrible", "valuable", "useless", "impressive",
        "disappointing", "outstanding", "worthy", "unworthy",
        "remarkable",
    ],
}


# =============================================================================
# DYNAMIC SUBJECTS — no fixed axis position
# =============================================================================
# Unlike fixed-value nouns (dog, grandfather, uncle), these "descriptive
# subjects" behave like pronouns: a speaker can address someone through them,
# but their axis position is not hardcoded here. It depends on the relationship
# between the speakers, and once that places one of them near a quadrant, it
# gets named the way that quadrant names things (We/You/I/That).
#
# They live in the lexicon as a distinct category so they are neither silently
# excluded like plain nouns nor silently fixed like real pronouns.
DYNAMIC_SUBJECTS = [
    "friend", "enemy", "villain", "hero", "rival", "ally", "stranger",
    "traitor", "companion", "leader", "follower", "mentor", "student",
    "protector", "threat", "target", "guardian", "opponent", "partner",
    "outsider", "intruder", "captor", "hostage", "witness", "accomplice",
]


# =============================================================================
# NEGATION AS AN OPERATOR — inversion vs. dampening
# =============================================================================
# Negation ("not"/"don't"/"never"/etc.) does not always push toward SOUTH by
# itself. It is an OPERATOR that acts on whatever follows it, and it behaves
# in two different ways depending on what that is:
#
#   1. INVERSION — negation applied to a verb/word that is ALREADY negative
#      in itself (e.g. "mind", "dislike", "refuse", "doubt") flips the sign:
#      South x South = North.
#         "I don't mind"    -> don't (South) + mind (already negative)
#                            -> clean affirmation, resolved -> NORTH
#
#   2. PLAIN PUSH — negation applied to a neutral/positive verb pushes
#      straight to South, no inversion:
#         "I don't want"    -> don't (South) + want (neutral/positive)
#                            -> clean negation -> SOUTH
#
#   3. DAMPENING (a different operator, don't confuse it with negation) —
#      a possibility modal (might/could/may) placed before a negation does
#      NOT invert the sign, it REDUCES THE MAGNITUDE of the vector. The
#      result sits near the center: neither a clean affirmation nor a clean
#      negation, just low-confidence doubt.
#         "I might not mind" -> might (dampens magnitude, doesn't flip sign)
#                             -> stays close to center -> DOUBT, not NORTH
#
# Practical rule:
#   - Verbs tagged as "inherently negative" (mind, dislike, refuse, doubt,
#     resent, deny, reject...) need their own small flag in the lexicon so
#     the negation-inversion rule can find them. TODO: tag these explicitly
#     in db_axis_lexicon_generated.py (currently not marked).
#   - Possibility modals (might/could/may/perhaps/maybe) should NEVER be
#     treated as negation-equivalent — they modify magnitude, not sign.
INHERENTLY_NEGATIVE_VERBS = [
    "mind", "dislike", "refuse", "doubt", "resent", "deny", "reject",
    "hate", "avoid", "distrust", "regret", "resist", "oppose",
]

POSSIBILITY_DAMPENERS = [
    "might", "could", "may", "perhaps", "maybe", "possibly",
    "presumably", "conceivably",
]


# =============================================================================
# INTEGRATION NOTES
# =============================================================================
#
# 1. This file does NOT replace db_lexicon.py (the thematic lexicon built
#    from WordNet lexnames). It is a cross-cutting layer: any word from
#    db_lexicon.py's "quality" category should end up tagged as AXIS_EAST
#    or AXIS_WEST using the criterion above.
#
# 2. Open questions:
#    - Can a word belong to more than one axis depending on context?
#      (e.g. "heavy" physically = East, but "a heavy situation" = West?)
#    - Do we auto-expand each list via WordNet's similar_tos(), or keep a
#      hand-picked "core" and only expand the core (safer, less noise)?
#
# 3. Suggested next step: run the "quality" words from db_lexicon.py through
#    the quick test one by one (single-sense, nothing compared -> East /
#    comparison of parts or standard -> West), instead of writing this list
#    from scratch, so nothing in the real lexicon gets duplicated or missed.

