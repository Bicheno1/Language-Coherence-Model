"""Randomness: a weighted draw with seed, temperature and top-k. It is NOT a language model.

It is used when a request is too GENERIC to choose a concrete node (\"tell me a story about an animal\":
animal is a type with hundreds of descendants). Then the tables say WHO the candidates are and with what
score (closeness to the target, etc.), and this module DRAWS one. The knobs are named like those of LLM
sampling, but here it is just a draw over a list that the tables already built:

    temperature 0   -> always the highest score (deterministic)
    temperature 1   -> probability proportional to the score
    temperature > 1 -> flatter: unlikely ones win more often
    temperature < 1 -> sharper: the best almost always wins
    top_k           -> only draws among the k highest scores
    seed            -> same seed = same result (if none is given, one is generated and reported, so it can be repeated)

Standard library only.
"""
import random
import secrets


class Sampler:
    def __init__(self, seed=None, temperature=1.0, top_k=None):
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        self.seed = int(seed) if seed is not None else secrets.randbits(32)
        self.temperature = float(temperature)
        self.top_k = top_k
        self.rng = random.Random(self.seed)
        self.used = False            # was anything actually drawn at random? (if not, the seed had no influence)

    def weights(self, candidates):
        """[(item, score)] -> [(item, probability)], ordered from highest to lowest probability (ties broken by the given order)."""
        order = sorted(range(len(candidates)), key=lambda i: (-candidates[i][1], i))
        if self.top_k:
            order = order[:max(1, int(self.top_k))]
        if self.temperature == 0:
            return [(candidates[order[0]][0], 1.0)]
        w = [max(candidates[i][1], 0.0) ** (1.0 / self.temperature) for i in order]
        total = sum(w)
        if total <= 0:                                # all zero: uniform
            w, total = [1.0] * len(order), float(len(order))
        return [(candidates[i][0], wi / total) for i, wi in zip(order, w)]

    def choose(self, candidates):
        """(item, probability with which it came out). `candidates` = [(item, score)], score >= 0."""
        if not candidates:
            raise ValueError("no candidates to draw from")
        self.used = True
        ps = self.weights(candidates)
        if len(ps) == 1:
            return ps[0]
        item = self.rng.choices([x for x, _ in ps], weights=[p for _, p in ps], k=1)[0]
        return item, dict((id(x), p) for x, p in ps)[id(item)]
