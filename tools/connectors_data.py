# connectors_data.py — LCM: Connectors table (English)
#
# Source of the lists: Halliday & Hasan, PDTB, Quirk et al., Huddleston & Pullum.
# Axis criteria (see README):
#   North  (Context, V)       = future, possibility, condition, purpose, "we/us" (join, include)
#   South  (World, I)         = past, negation, closure, result, exception
#   East–West (Lv / Gv)       = present: the horizontal band (V = I  ->  y ≈ 0)
#       East  (Imposition, Lv) = the speaker imposes: emphasis, stance, the concrete
#       West  (Adaptation, Gv) = structure: joins, compares, explains, relates
#
# Weights (V, I, Lv, Gv), integers 0-10. They are design assumptions, editable.

REL = {
    # EXPANSION
    "additive_we":        (9, 1, 5, 5),   # join/include ("we/us"): up and centered, like 'us'
    "additive":           (3, 1, 2, 8),   # add elements of the same level: structural, present (#5, West)
    "alternative":        (5, 3, 3, 7),   # opens a choice (#5, West)
    "substitution":       (3, 6, 3, 7),   # choose one and discard the other
    "negative_alt":       (0, 9, 4, 6),   # nor / neither: "not" and "not"
    "exception":          (1, 8, 3, 6),   # excludes: partial negation
    "exemplification":    (4, 2, 7, 4),   # points at a concrete case (present, sensory)
    "clarification":      (4, 3, 2, 8),   # rephrases for the listener (structural)
    "clarif_speaker":     (2, 6, 6, 3),   # "I mean / what I mean is": the speaker corrects (#15, South)
    "emphasis":           (6, 1, 9, 1),   # the speaker imposes, asserts (East)
    "hedge":              (8, 2, 3, 5),   # softens: possibility (North, #16)
    "topic_shift":        (5, 3, 2, 8),   # reframes the topic (structural)
    # COMPARISON
    "contrast_soft":      (3, 5, 2, 8),   # contrasts without closing (present, West)
    "contrast_hard":      (1, 8, 3, 6),   # denies the expectation: discourse (#30, South)
    "concession":         (4, 4, 2, 8),   # V/I balance: present band (#9, West)
    "manner":             (8, 2, 3, 6),   # "as if": hypothetical (#22, North)
    "comparison":         (4, 3, 2, 9),   # compares parts (West, #19)
    # CONTINGENCY
    "cause":              (3, 5, 2, 8),   # looks back at what already happened (#7, West)
    "result":             (2, 7, 3, 6),   # closed consequence (#21, South)
    "condition":          (9, 1, 3, 6),   # opens a possibility (#8, North)
    "condition_soft":     (8, 2, 3, 7),
    "condition_neg":      (4, 6, 3, 7),   # unless = "if not": mixes North and South
    "hypothetical":       (9, 1, 4, 5),
    "purpose":            (9, 1, 4, 6),   # where it is heading (#18, North)
    "purpose_neg":        (2, 7, 3, 7),   # lest / for fear that: NEGATED purpose -> South (deliberately differs from #18, North)
    # TEMPORAL — time runs vertically: past South, future North, present in the band
    "simultaneity":       (2, 2, 2, 8),   # present, structural
    "simultaneity_now":   (2, 2, 7, 3),   # present, sensory ("meanwhile")
    "precedence":         (1, 6, 2, 7),   # before = past
    "succession_sub":     (5, 3, 2, 8),   # after, subordinator: slightly forward
    "succession_adv":     (7, 2, 3, 7),   # then / next / later: forward (relative future)
    "seq_open":           (6, 2, 2, 8),   # opens a series
    "seq_mid":            (3, 3, 2, 8),
    "seq_close":          (2, 6, 2, 8),   # closes the series
    "summary":            (1, 7, 2, 8),   # discourse closing
    # CLAUSES
    "relative":           (4, 3, 2, 8),   # joins a noun with its clause (#23/#24, West)
    "relative_adv":       (4, 3, 3, 7),
    "compl_question":     (8, 2, 3, 6),   # whether / if: indirect question (#29, North)
    "compl_that":         (7, 2, 2, 8),   # that: reporta (#4)
    "stance_neutral":     (5, 2, 8, 2),   # the speaker comments on the sentence (East)
    "stance_pos":         (8, 1, 8, 2),
    "stance_neg":         (1, 8, 8, 2),
}

# Axis of each construction, inherited from Unique Host
CONS_AXIS = {4: "North", 5: "West", 6: "South", 7: "West", 8: "North", 9: "West", 10: "West", 11: "West",
             12: "West", 14: "North", 15: "South", 16: "North", 18: "North", 19: "West", 21: "South", 22: "North",
             23: "West", 24: "West", 27: "East", 28: "North", 29: "North", 30: "South"}

# Design decision #2 (v0.4): rule #10 TEMPORAL and #18 PURPOSE are SPLIT by relation, instead of having a single axis.
#   #10: simultaneous -> present band (West); earlier (before, until) -> South; later (then, later) -> North.
#   #18: affirmed purpose (so that, in order to) -> North; NEGATED purpose (lest, for fear that) -> South.
# Key: (construction, relation). If it exists, it takes precedence over CONS_AXIS.
CONS_AXIS_SPLIT = {(10, "precedence"): "South", (18, "purpose_neg"): "South"}

GROUP_OF = {
    "additive_we": "expansion", "additive": "expansion", "alternative": "expansion", "substitution": "expansion",
    "negative_alt": "expansion", "exception": "expansion", "exemplification": "expansion", "clarification": "expansion",
    "clarif_speaker": "expansion", "emphasis": "expansion", "hedge": "expansion", "topic_shift": "expansion",
    "contrast_soft": "comparison", "contrast_hard": "comparison", "concession": "comparison", "manner": "comparison",
    "comparison": "comparison", "cause": "contingency", "result": "contingency", "condition": "contingency",
    "condition_soft": "contingency", "condition_neg": "contingency", "hypothetical": "contingency",
    "purpose": "contingency", "purpose_neg": "contingency", "simultaneity": "temporal", "simultaneity_now": "temporal",
    "precedence": "temporal", "succession_sub": "temporal", "succession_adv": "temporal", "seq_open": "temporal",
    "seq_mid": "temporal", "seq_close": "temporal", "summary": "temporal", "relative": "clauses",
    "relative_adv": "clauses", "compl_question": "clauses", "compl_that": "clauses",
    "stance_neutral": "clauses", "stance_pos": "clauses", "stance_neg": "clauses",
}

REL_LABEL = {   # readable name of the relation
    "additive_we": "additive (we/us)", "additive": "additive", "alternative": "alternative", "substitution": "alternative (substitution)",
    "negative_alt": "negative alternative", "exception": "exception", "exemplification": "exemplification",
    "clarification": "clarification", "clarif_speaker": "clarification (speaker)", "emphasis": "emphasis", "hedge": "hedge / nuance",
    "topic_shift": "topic shift", "contrast_soft": "contrast", "contrast_hard": "contrast (discourse)", "concession": "concession",
    "manner": "manner", "comparison": "comparison", "cause": "cause", "result": "result", "condition": "condition",
    "condition_soft": "condition", "condition_neg": "condition (negative)", "hypothetical": "hypothetical", "purpose": "purpose",
    "purpose_neg": "purpose (negative)", "simultaneity": "simultaneity", "simultaneity_now": "simultaneity (now)",
    "precedence": "precedence", "succession_sub": "succession", "succession_adv": "succession (adverb)",
    "seq_open": "sequence (opens)", "seq_mid": "sequence", "seq_close": "sequence (closes)", "summary": "summary / conclusion",
    "relative": "relative", "relative_adv": "relative (adverb)", "compl_question": "complementizer (question)",
    "compl_that": "complementizer (that)", "stance_neutral": "stance adjunct", "stance_pos": "stance adjunct (+)",
    "stance_neg": "stance adjunct (−)",
}

# (word or phrase, relation, class, construction(s), ambiguous, note)
# class: CC coordinating, CORR correlative, SC subordinating, CA conjunctive adverb, TP transition phrase,
#        REL relative, COMP complementizer, PREP prepositional, STANCE stance adjunct
N = []
def add(rel, cls, words, cons=(), amb=False, note=""):
    for w in words.split("|"):
        N.append((w.strip(), rel, cls, tuple(cons), amb, note))

# EXPANSION ------------------------------------------------------------------
add("additive", "CC", "and|also|too|plus", cons=(5,), note="#5 SIMPLE_CONNECTOR")
add("additive_we", "CA", "moreover|furthermore|besides|additionally|likewise|similarly")
add("additive_we", "TP", "in addition|as well as")
add("additive_we", "CORR", "both…and")
add("alternative", "CC", "or", cons=(5,))
add("alternative", "CORR", "either…or")
add("alternative", "CA", "alternatively")
add("substitution", "CA", "instead|rather|or else")
add("negative_alt", "CC", "nor")
add("negative_alt", "CORR", "neither…nor")
add("exception", "SC", "except|except that|apart from|other than|but for|save")
add("exemplification", "TP", "for example|for instance|such as|including|e.g.|in particular|specifically|namely")
add("clarification", "TP", "in other words|that is|i.e.|to put it another way", cons=(15, 11))
add("clarif_speaker", "TP", "I mean|what I mean is", cons=(15,), note="#15 METALINGUISTIC")
add("emphasis", "CA", "indeed|in fact|actually|especially|certainly")
add("emphasis", "TP", "as a matter of fact|above all|of course", note="of course is also in AXIS_NORTH.affirmatives")
add("hedge", "TP", "in a way|sort of|kind of|somewhat|a bit|more or less", cons=(16,), note="#16 NUANCE")
add("topic_shift", "TP", "by the way|incidentally|anyway|anyhow|as for|speaking of|regarding|with regard to")
# COMPARISON ------------------------------------------------------------------
add("contrast_soft", "CC", "but", cons=(5,), note="#5 SIMPLE_CONNECTOR")
add("contrast_soft", "TP", "on the other hand|in contrast|by contrast")
add("contrast_soft", "SC", "whereas")
add("contrast_soft", "SC", "while (contrast)", cons=(), amb=True, note="while: temporal / contraste")
add("contrast_hard", "CA", "however|nevertheless|nonetheless|conversely|otherwise", cons=(30,), note="#30 DISCOURSE_CONNECTOR (otherwise is also a condition)")
add("contrast_hard", "CA", "still (contrast)|yet (contrast)", cons=(30,), amb=True, note="still / yet: also time adverbs")
add("contrast_hard", "TP", "all the same", cons=(30,))
add("concession", "SC", "although|even though|albeit|despite|in spite of", cons=(9,), note="#9 CONCESSIVE")
add("concession", "SC", "though (concessive)", cons=(9,), amb=True, note="though: #9 West, but #30 South if it opens the sentence")
add("contrast_hard", "CA", "though (sentence-initial)", cons=(30,), amb=True, note="#30 DISCOURSE_CONNECTOR")
add("condition_soft", "SC", "even if", cons=(8,))
add("manner", "SC", "as if|as though|like (manner)|the way", cons=(22,), note="#22 MANNER")
add("comparison", "SC", "than|as…as|the more…the more|in the same way", cons=(19,), note="#19 COMPARATIVE")
# CONTINGENCY ------------------------------------------------------------------
add("cause", "SC", "because|now that|given that", cons=(7,), note="#7 CAUSAL")
add("cause", "SC", "since (causal)|as (causal)|for (causal)", cons=(7,), amb=True, note="since/as/for: causal, temporal or preposition")
add("cause", "PREP", "because of|due to|owing to|thanks to", cons=(7,))
add("result", "CA", "therefore|thus|hence|consequently|accordingly")
add("result", "TP", "as a result|so…that|such…that", cons=(21,))
add("result", "CC", "so (result)", cons=(21,), amb=True, note="so: result (#21, South) or purpose (#18, North)")
add("condition", "SC", "if", cons=(8,), note="#8 CONDITIONAL")
add("condition_soft", "SC", "provided (that)|providing|as long as|so long as|in case|on condition that|supposing|assuming", cons=(8,))
add("condition_neg", "SC", "unless")
add("hypothetical", "SC", "if only", cons=(14,), note="#14 HYPOTHETICAL")
add("purpose", "SC", "so that|in order to|in order that", cons=(18,), note="#18 PURPOSE")
add("purpose_neg", "SC", "lest|for fear that", cons=(18,))
# TEMPORAL ----------------------------------------------------------------------
add("simultaneity", "SC", "when|whenever", cons=(10,), note="#10 TEMPORAL")
add("simultaneity", "SC", "while (temporal)|as (temporal)", cons=(10,), amb=True)
add("simultaneity_now", "CA", "meanwhile|at the same time")
add("precedence", "SC", "before|until|till|by the time", cons=(10,))
add("precedence", "CA", "previously|earlier")
add("succession_sub", "SC", "after|once|as soon as", cons=(10,))
add("succession_sub", "SC", "since (temporal)", cons=(10,), amb=True)
add("succession_adv", "CA", "then|next|afterwards|later|subsequently")
add("seq_open", "CA", "first|to begin with")
add("seq_mid", "CA", "second|third")
add("seq_close", "CA", "finally|lastly")
add("summary", "TP", "in conclusion|to sum up|in short|in summary|overall|all in all|to conclude")
# CLAUSES -----------------------------------------------------------------------
add("relative", "REL", "who|whom|whose|which|that (relative)", cons=(23, 24), note="#23 restrictiva / #24 no restrictiva (coma)")
add("relative_adv", "REL", "where|when (relative)|why", cons=(23,))
add("compl_question", "COMP", "whether|if (complementizer)", cons=(29,), note="#29 INDIRECT_Q")
add("compl_that", "COMP", "that (complementizer)", cons=(4,), amb=True, note="that: relativo / completivo / demostrativo")
add("stance_neutral", "STANCE", "frankly|honestly|clearly|obviously|apparently|surprisingly")
add("stance_pos", "STANCE", "fortunately")
add("stance_neg", "STANCE", "unfortunately|sadly")

CONNECTORS = N

# PUNCTUATION (own weights: there is no common semantic relation) ---------------------
# (mark, name, V, I, Lv, Gv, constructions, note)
PUNCT = [
    (",",  "comma",            2, 2, 2, 8, (24, 12), "separates clauses: present band, structural"),
    (".",  "period",           1, 7, 3, 6, (),       "closes the sentence: closure = South"),
    ("?",  "question mark",   9, 1, 4, 5, (28,),    "opens possibility: up and centered (#28 DIRECT_Q)"),
    ("!",  "exclamation",     3, 5, 9, 2, (27, 30), "#27 EXCLAMATIVE is East; AXIS_SOUTH.exclamations is South: compromise"),
    (";",  "semicolon",    2, 2, 3, 8, (),       "joins clauses without a conjunction: present, structural"),
    (":",  "colon",      3, 2, 2, 8, (11,),    "introduces an explanation (#11 EXPLANATORY)"),
    ("—",  "dash",            3, 3, 7, 3, (),       "abrupt break: Imposition"),
    ("-",  "hyphen",           2, 2, 3, 7, (),       "joins compounds (high-pitched)"),
    ("( )", "parentheses",     3, 3, 3, 7, (),       "aside, clarification"),
    ('" "', "quotation marks",       4, 3, 1, 9, (),       "another person's direct speech (You); clashes with marker_parser (emotion)"),
    ("'",  "apostrophe",       2, 2, 3, 7, (),       "contraction and possessive; today the tokenizer deletes it"),
    ("…",  "ellipsis", 5, 3, 3, 6, (),    "trailing-off phrase, doubt (like POSSIBILITY_DAMPENERS)"),
    ("/",  "slash",           5, 2, 2, 8, (),       "alternative (and/or)"),
    ("&",  "ampersand",       6, 1, 2, 8, (),       "equivalent to and"),
]
