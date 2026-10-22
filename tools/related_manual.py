# related_manual.py — LCM v0.4: MANUAL seed of `related` links (narrative associations).
#
# WordNet gives taxonomic relations (is-a, part-of, derivational) but NOT usage associations
# ("a story has characters", "characters jump"). Those are written here, by hand.
# It is a small seed for the guiding scenario "tell me a story" (decision #5: derive from WordNet and
# complete the critical parts by hand). Each line:  word -> words it opens.
# The links are DIRECTED; build_tables filters out those pointing at words that are not nodes.

MANUAL = {
    # the guiding example: story -> person -> jump
    "story":      ["person", "event", "adventure", "tell", "listen", "begin", "end", "character", "journey", "hero"],
    "tell":       ["story", "listen", "person", "speak", "say", "hear"],
    "listen":     ["hear", "story", "person", "child"],
    "person":     ["jump", "run", "walk", "talk", "laugh", "cry", "play", "child", "friend", "family", "hope", "fear"],
    "child":      ["play", "jump", "run", "laugh", "story", "family", "mother", "father", "dog"],
    "man":        ["walk", "work", "run", "story", "family"],
    "woman":      ["walk", "work", "run", "story", "family"],
    "hero":       ["adventure", "journey", "fear", "hope", "jump", "fight", "story"],
    "king":       ["queen", "castle", "story", "war", "power"],
    "queen":      ["king", "castle", "story"],
    "dragon":     ["fire", "fear", "fly", "hero", "cave", "story"],
    "dog":        ["jump", "run", "bark", "play", "friend", "child", "walk"],
    "cat":        ["jump", "run", "sleep", "play", "mouse"],
    "bird":       ["fly", "sing", "tree"],
    "animal":     ["jump", "run", "eat", "sleep", "forest"],
    "jump":       ["run", "fall", "land", "fly", "climb", "play", "person", "dog", "fear"],
    "run":        ["jump", "walk", "fall", "fear", "escape", "person", "dog"],
    "walk":       ["run", "path", "forest", "journey", "person"],
    "fall":       ["jump", "fear", "pain", "hole"],
    "fly":        ["jump", "bird", "sky", "dragon", "fall"],
    "climb":      ["jump", "mountain", "tree", "fall"],
    "play":       ["jump", "run", "child", "laugh", "game", "dog"],
    "laugh":      ["happy", "play", "friend", "joy"],
    "cry":        ["sad", "pain", "fear", "loss"],
    "adventure":  ["journey", "hero", "fear", "hope", "danger", "forest", "story"],
    "journey":    ["walk", "path", "adventure", "begin", "end", "road"],
    "forest":     ["tree", "animal", "dark", "path", "danger", "adventure"],
    "castle":     ["king", "queen", "tower", "story"],
    "night":      ["dark", "fear", "sleep", "dream", "moon"],
    "dream":      ["sleep", "night", "hope", "fear", "story"],
    "fear":       ["danger", "run", "dark", "hope", "cry"],
    "hope":       ["future", "dream", "happy", "journey"],
    "danger":     ["fear", "run", "escape", "fire", "dark"],
    "friend":     ["help", "laugh", "play", "family"],
    "family":     ["mother", "father", "child", "home", "friend"],
    "home":       ["family", "house", "night", "safe"],
    "begin":      ["end", "start", "story", "journey"],
    "end":        ["begin", "story", "journey", "loss"],
    "magic":      ["wonder", "story", "hero", "dragon", "fear"],
    "fire":       ["danger", "light", "dragon", "burn"],
}


# ── v0.4b: from links to SENTENCES ───────────────────────────────────────────────────────────────────────
# For the walk to produce text, two kinds of association are needed that WordNet does not give:
#   1. WHO CAN DO WHAT  (pig -> jump): species -> typical actions without an object.
#   2. HOW IT IS DONE   (jump -> high): verb -> typical adverb of manner.
# Also the intermediate step of the story: story -> animal / child (so an animal can be a protagonist).
# All of this is a hand-written seed and can be edited; only links whose target is a node survive.

_SPECIES = {
    "pig":      ["jump", "run", "play", "sleep", "dance", "roll", "dig"],
    "rabbit":   ["hop", "jump", "run", "sleep", "dig", "play"],
    "mouse":    ["run", "hide", "climb", "jump", "sleep", "play"],
    "fox":      ["run", "jump", "hide", "climb", "sleep", "hunt"],
    "lamb":     ["jump", "run", "play", "sleep", "dance"],
    "sheep":    ["run", "sleep", "jump", "walk"],
    "horse":    ["run", "jump", "gallop", "walk", "sleep"],
    "frog":     ["jump", "hop", "swim", "sing", "sleep"],
    "bear":     ["walk", "run", "sleep", "climb", "swim", "dance"],
    "wolf":     ["run", "hunt", "howl", "sleep", "jump"],
    "duck":     ["swim", "fly", "walk", "dance", "sing"],
    "cow":      ["walk", "sleep", "run", "dance", "jump"],
    "goat":     ["jump", "climb", "run", "play"],
    "monkey":   ["climb", "jump", "swing", "play", "dance", "laugh"],
    "squirrel": ["climb", "jump", "run", "hide", "play"],
    "bird":     ["fly", "sing", "sleep", "jump", "dance"],
    "cat":      ["jump", "run", "sleep", "play", "climb", "hide"],
    "dog":      ["jump", "run", "bark", "play", "sleep", "dig", "swim"],
    "puppy":    ["jump", "run", "play", "bark", "sleep"],
    "lion":     ["run", "sleep", "jump", "hunt", "walk"],
    "elephant": ["walk", "swim", "dance", "sleep"],
    "turtle":   ["walk", "swim", "sleep", "hide"],
    "child":    ["play", "jump", "run", "laugh", "sing", "dance", "sleep", "climb", "swim", "cry"],
    "boy":      ["play", "jump", "run", "laugh", "sing", "dance", "sleep", "climb", "swim", "cry"],
    "girl":     ["play", "jump", "run", "laugh", "sing", "dance", "sleep", "climb", "swim", "cry"],
    "king":     ["walk", "sleep", "laugh", "cry", "dance", "sing"],
    "queen":    ["walk", "sleep", "laugh", "cry", "dance", "sing"],
    "hero":     ["jump", "run", "climb", "fight", "swim", "walk", "cry"],
    # generic types: inherited by species that have no verbs of their own (lcm/narrate.py: inheritance via WordNet)
    "animal":   ["walk", "run", "jump", "sleep", "play", "hide"],
    "fish":     ["swim", "hide", "play", "sleep"],
    "insect":   ["walk", "fly", "hide", "jump", "sleep"],
}
for _w, _vs in _SPECIES.items():
    MANUAL[_w] = list(dict.fromkeys(MANUAL.get(_w, []) + _vs))

# the story can have an animal or a child as protagonist
MANUAL["story"] = list(dict.fromkeys(MANUAL["story"] + ["animal", "child", "king", "dog", "pig", "rabbit", "mouse", "fox", "frog", "bear"]))
MANUAL["animal"] = list(dict.fromkeys(MANUAL.get("animal", []) + ["pig", "rabbit", "mouse", "fox", "lamb", "horse", "frog", "bear",
                                                                  "wolf", "duck", "goat", "monkey", "squirrel", "bird", "cat", "dog"]))

_MANNER = {   # verb -> typical adverbs of manner (the first is the most typical)
    "jump":   ["high", "far", "happily"],       "hop":    ["high", "happily"],      "run":    ["fast", "far", "away"],
    "walk":   ["slowly", "far", "home"],        "fly":    ["high", "far", "away"],  "swim":   ["far", "fast", "slowly"],
    "climb":  ["high", "slowly", "carefully"],  "fall":   ["hard", "down"],         "dance":  ["happily", "slowly"],
    "play":   ["happily", "together"],          "sing":   ["loudly", "happily", "softly"], "laugh": ["loudly", "happily"],
    "cry":    ["loudly", "softly"],             "sleep":  ["soundly", "peacefully"], "bark":  ["loudly"],
    "hide":   ["quietly", "well"],              "dig":    ["deep", "hard"],         "roll":   ["slowly", "away"],
    "hunt":   ["quietly", "far"],               "howl":   ["loudly"],               "gallop": ["fast", "far"],
    "swing":  ["high", "happily"],              "fight":  ["bravely", "hard"],
}
for _v, _advs in _MANNER.items():
    MANUAL[_v] = list(dict.fromkeys(MANUAL.get(_v, []) + _advs))

# Verbs the seed APPROVES as story actions without an object ("it jumped", "it slept"). Only these can be the
# verb of a generated sentence: an automatic filter by WordNet frame lets through 'home' and 'war' (odd verbs).
NARRATIVE_VERBS = set(_MANNER) | {v for vs in _SPECIES.values() for v in vs} | {"land", "fall", "escape", "talk", "smile"}

# ── v0.4e: giving opinions ("what do you think about ducks?") ────────────────────────────────────────────
# Adjectives the system can EVALUATE something with. Hand-written list: SentiWordNet sentiment is NOT suitable for
# picking them from the tables (it gives 'thorough', 'righteous', 'sustainable' or 'prestigious' as positive, and 'dog' has negative
# sentiment because of its insult slang). Only those that are quality nodes survive. build_tables marks them (`opinion` field).
OPINION_ADJECTIVES = {
    "pos": ["nice", "wonderful", "lovely", "beautiful", "pleasant", "happy", "gorgeous", "charming", "delightful", "great", "good",
            "fine", "excellent", "brilliant", "perfect", "cute", "funny", "friendly", "fun", "amazing", "awesome", "fascinating",
            "interesting"],
    "neg": ["nasty", "awful", "terrible", "scary", "dirty", "dangerous", "creepy", "filthy", "painful", "sad", "ugly", "boring",
            "horrible", "annoying", "noisy", "dull"],
}

# ── v0.4i: questions about the LCM's own state ("how are you?", "are you happy?") ────────────────────────
# Adjectives the system can answer about its state with ("I am fine"). Hand-written list; build_tables marks them (field
# `state`). NOTE: it is not a real state: it is the 'amable' (friendly) disposition (lcm/reply.py) applied to node I. Only quality nodes survive.
STATE_ADJECTIVES = {
    "pos": ["fine", "good", "great", "happy", "wonderful", "excellent", "well", "pleasant", "glad", "calm", "ready", "cheerful", "awesome"],
    "neg": ["tired", "sad", "bored", "angry", "sick", "upset", "worried", "lonely", "afraid", "confused"],
}

