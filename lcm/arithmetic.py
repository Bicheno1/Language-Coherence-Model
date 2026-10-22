"""LCM arithmetic (optional, it does not touch the tables or the axes).

How it works:
  1. Operations are entries related to each other (data/LCM_arithmetic.json): add = base; subtract = inverse of add;
     multiply = repeats add; divide = inverse of multiply; power = repeats multiply; root = inverse of power.
     The precedence LEVEL is derived from that chain (level = how many times the base is repeated): it is not written by hand.
  2. Signs follow the logic of the axes: up adds / down subtracts (vertical) and right is positive / left is negative
     (horizontal); the effective step is the product of the two (\"plus times minus = minus\").
  3. Calculation goes DIGIT by DIGIT with fact tables (0-9 + 0-9, 0-9 - 0-9, 0-9 x 0-9) built once by COUNTING, plus
     carry, borrow and shifts. Cost: it depends on the number of digits, not on the value (99999 x 9999 = ~107 lookups).
Limits: integers only (no fractions or decimals as input); inexact roots are given with 2 decimals; chained powers
are read left to right (a convention, it does not come from the relations); no parentheses (the functions replace them).
Reading number words ('twenty seven') uses ordinary Python integers; the CALCULATION does not.
"""
import json
import os
import re
from dataclasses import dataclass, field

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "LCM_arithmetic.json")
CFG = json.load(open(_DATA, encoding="utf-8"))
LIM = CFG["limits"]
REL = CFG["operations"]


class TooLong(Exception):
    pass


class ArithError(Exception):
    """The exercise is arithmetic but has no answer (division by zero, fractions, root of a negative...). It is answered with text."""


class Unsupported(Exception):
    pass


class _Count:
    look = 0


C = _Count()


def tick(n=1):
    C.look += n
    if C.look > LIM["steps"]:
        raise TooLong()


# ── fact tables: built ONCE, by counting ───────────────────────────────────────────────────────────────
ADD, SUB, MUL = {}, {}, {}
BUILD_STEPS = 0


def _build():
    global BUILD_STEPS
    for a in range(10):
        for b in range(10):
            s = a
            for _ in range(b):
                s += 1; BUILD_STEPS += 1
            ADD[(a, b)] = divmod(s, 10)                 # (carry, digit)
            d, borrow = a, 0
            for _ in range(b):                          # count down; if it goes below 0 it wraps around and borrows
                if d == 0:
                    d, borrow = 9, 1
                else:
                    d -= 1
                BUILD_STEPS += 1
            SUB[(a, b)] = (borrow, d)                   # (borrow, digit)
            p = 0
            for _ in range(b):                          # a x b = add a, b times
                for _ in range(a):
                    p += 1; BUILD_STEPS += 1
            MUL[(a, b)] = divmod(p, 10)


_build()


# ── magnitudes: digit lists, least significant first ─────────────────────────────────────────────────────
def trim(A):
    A = list(A)
    while len(A) > 1 and A[-1] == 0:
        A.pop()
    return A


def cmp(A, B):
    A, B = trim(A), trim(B)
    tick()
    if len(A) != len(B):
        return 1 if len(A) > len(B) else -1
    for x, y in zip(reversed(A), reversed(B)):
        tick()
        if x != y:
            return 1 if x > y else -1
    return 0


def add_mag(A, B):
    out, carry = [], 0
    for i in range(max(len(A), len(B))):
        x = A[i] if i < len(A) else 0
        y = B[i] if i < len(B) else 0
        tick(); c1, d = ADD[(x, y)]
        c2 = 0
        if carry:
            tick(); c2, d = ADD[(d, carry)]
        out.append(d); carry = c1 or c2
    if carry:
        out.append(carry)
    return trim(out)


def sub_mag(A, B):                                      # exige A >= B
    out, borrow = [], 0
    for i in range(len(A)):
        x = A[i]
        y = B[i] if i < len(B) else 0
        tick(); b1, d = SUB[(x, y)]
        b2 = 0
        if borrow:
            tick(); b2, d = SUB[(d, borrow)]
        out.append(d); borrow = b1 or b2
    return trim(out)


def mul_mag(A, B):
    A, B = trim(A), trim(B)
    if A == [0] or B == [0]:
        return [0]
    total = [0]
    for j, db in enumerate(B):
        row, carry = [], 0
        for da in A:
            tick(); t, o = MUL[(da, db)]
            c2 = 0
            if carry:
                tick(); c2, o = ADD[(o, carry)]
            tick(); _, carry = ADD[(t, c2)]
            row.append(o)
        if carry:
            row.append(carry)
        tick(j); row = [0] * j + row                    # shift = one step per position
        total = add_mag(total, row)
        if len(total) > LIM["digits"]:
            raise TooLong()
    return total


def divmod_mag(A, B):                                   # long division; B != 0
    A, B = trim(A), trim(B)
    Q, R = [], [0]
    for digit in reversed(A):
        R = trim([digit] + R); tick()                   # bring down the next digit
        d = 0
        for t in range(1, 10):                          # the largest d such that B x d <= R
            tick()
            if cmp(mul_mag(B, [t]), R) <= 0:
                d = t
            else:
                break
        Q.append(d)
        if d:
            R = sub_mag(R, mul_mag(B, [d]))
    return trim(list(reversed(Q))), trim(R)


def pow_mag(A, n):
    r = [1]
    for _ in range(n):
        r = mul_mag(r, A)
    return r


def root_floor(X, n):                                   # the largest r with r^n <= X, digit by digit (10 trials per digit)
    X = trim(X)
    m = (len(X) + n - 1) // n
    r = [0] * m
    for pos in range(m - 1, -1, -1):
        for d in range(9, -1, -1):
            cand = list(r); cand[pos] = d
            if cmp(pow_mag(cand, n), X) <= 0:
                r = cand
                break
    return trim(r)


# ── signed numbers: (sign, digits) ───────────────────────────────────────────────────────────────────────
def mk(s, sign=1):
    return (sign, trim([int(c) for c in reversed(s)]))


def is_zero(a):
    return a[1] == [0]


def norm(a):
    return (1, a[1]) if is_zero(a) else a


def neg(a):                                             # bottom x left
    return norm((-a[0], a[1]))


def show(a):
    return ("-" if a[0] < 0 else "") + "".join(str(d) for d in reversed(a[1]))


def n_add(a, b):
    if a[0] == b[0]:
        return norm((a[0], add_mag(a[1], b[1])))
    c = cmp(a[1], b[1])
    if c == 0:
        return (1, [0])
    return norm((a[0], sub_mag(a[1], b[1]))) if c > 0 else norm((b[0], sub_mag(b[1], a[1])))


def n_sub(a, b):
    return n_add(a, neg(b))


def n_mul(a, b):
    return norm((a[0] * b[0], mul_mag(a[1], b[1])))     # plus times minus = minus


def n_divmod(a, b):
    if is_zero(b):
        raise ArithError("zero")
    q, r = divmod_mag(a[1], b[1])
    return norm((a[0] * b[0], q)), norm((a[0], r))


def n_pow(a, k):                                        # k: number of times the multiplication repeats
    if k > LIM["exponent"]:
        raise TooLong()
    r = mk("1")
    for _ in range(k):
        r = n_mul(r, a)
        if len(r[1]) > LIM["digits"]:
            raise TooLong()
    return r


def n_root(a, n):
    if a[0] < 0 and n % 2 == 0:
        raise ArithError("negative-root")
    r = root_floor(a[1], n)
    exact = cmp(pow_mag(r, n), a[1]) == 0
    sign = a[0] if not is_zero(a) else 1
    return (sign, r), exact


def root_decimals(a, n, places):                        # a x (10^n)^places, integer root, then shift the decimal point
    shifted = [0] * (n * places) + trim(a[1]); tick(n * places)
    r = root_floor(shifted, n)
    r = r + [0] * (places + 1 - len(r)) if len(r) < places + 1 else r
    ip, dp = r[places:], r[:places]
    return ("-" if a[0] < 0 else "") + "".join(str(d) for d in reversed(ip)) + "." + "".join(str(d) for d in reversed(dp))


# ── operations as related entries: the level comes from the chain ────────────────────────────────────────
def level(op):
    r = REL[op]
    if r.get("base"):
        return 1
    if "inverse_of" in r:
        return level(r["inverse_of"])
    return level(r["repeats"]) + 1


LEVELS = {op: level(op) for op in REL}
WORD_OPS = sorted([(w, op) for op, r in REL.items() for w in r.get("words", [])], key=lambda x: -len(x[0]))
SYMBOL_OPS = {s: op for op, r in REL.items() for s in r.get("symbols", [])}
NUM = CFG["numbers"]
_WORDS = set(NUM["units"]) | set(NUM["tens"]) | set(NUM["scales"]) | set(NUM["sign"])


def words_to_int(tokens):
    total = cur = 0
    for w in tokens:
        if w in NUM["units"]:
            cur += NUM["units"][w]
        elif w in NUM["tens"]:
            cur += NUM["tens"][w]
        elif w == "hundred":
            cur = (cur or 1) * 100
        elif w in NUM["scales"]:
            total += (cur or 1) * NUM["scales"][w]; cur = 0
        else:
            return None
    return total + cur


def parse_number(s):
    """'twenty seven' / '123' / '1,234' / 'negative five' / '-5' -> (sign, digits); None if it is not a number; Unsupported if it is a decimal."""
    s = s.strip().lower().replace(",", "")
    sign = 1
    if s.startswith("-"):
        sign, s = -1, s[1:].strip()
    else:
        for w in NUM["sign"]:
            if s.startswith(w + " "):
                sign, s = -1, s[len(w) + 1:].strip()
                break
    if re.fullmatch(r"\d+", s):
        return norm(mk(s, sign))
    if re.fullmatch(r"\d*\.\d+", s):
        raise Unsupported("decimals")
    toks = s.replace("-", " ").split()
    if not toks or any(t not in _WORDS for t in toks):
        return None
    v = words_to_int(toks)
    return None if v is None else norm(mk(str(v), sign))


def _numeral(n):
    return show(n)


def _op_word(op):
    return REL[op]["words"][0] if REL[op].get("words") else op


# ── text -> exercise ────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Answer:
    text: str
    trace: list = field(default_factory=list)
    steps: int = 0


def _normalize(text):
    t = text.lower().strip().rstrip("?.!")
    t = t.replace("×", "*").replace("÷", "/")
    for p in CFG["opening_phrases"]:
        if t.startswith(p + " "):
            t = t[len(p) + 1:]
            break
    for imp in CFG["imperatives"]:
        if re.match(imp["pattern"], t):
            t = re.sub(imp["pattern"], imp["template"], t)
            break
    t = re.sub(r"\b(" + "|".join(NUM["tens"]) + r")-(" + "|".join(k for k, v in NUM["units"].items() if 0 < v < 10) + r")\b", r"\1 \2", t)
    for w, op in WORD_OPS:
        t = re.sub(r"\b" + re.escape(w) + r"\b", f" @{op}@ ", t)
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d|-)", " @multiply@ ", t)
    t = re.sub(r"\s*\^\s*", " @power@ ", t)
    t = re.sub(r"\s*\*\s*", " @multiply@ ", t)
    t = re.sub(r"\s*/\s*", " @divide@ ", t)
    t = re.sub(r"\s*\+\s*", " @add@ ", t)
    t = re.sub(r"(?<=\S)\s+-\s+(?=\S)", " @subtract@ ", t)
    t = re.sub(r"(?<=\d)-(?=-?\d)", " @subtract@ ", t)
    return t


def _resolve_calls(t, depth=0):                         # f(a, b): resolved from the inside out
    funcs = CFG["functions"]
    for _ in range(30):
        m = re.search(r"([a-z_0-9]+)\(([^()]*)\)", t)
        if not m or m.group(1) not in funcs:
            break
        name = m.group(1)
        args = [_value(_normalize(a)) for a in m.group(2).split(",")]
        params, body = funcs[name]["parameters"], funcs[name]["body"]
        for p, v in zip(params, args):
            body = re.sub(rf"\b{p}\b", ("negative " if v[0] < 0 else "") + "".join(str(d) for d in reversed(v[1])), body)
        res = _value(_normalize(_resolve_calls(body, depth + 1)))
        t = t[:m.start()] + ("negative " if res[0] < 0 else "") + "".join(str(d) for d in reversed(res[1])) + t[m.end():]
    return t


def _operand(s):
    """One operand: number, 'the square root of N', 'N squared'... -> (value, description) or None."""
    s = s.strip()
    for pre, spec in CFG["unary"]["prefixes"].items():
        if s.startswith(pre + " "):
            inner = _operand(s[len(pre) + 1:])
            if inner is None:
                return None
            return _apply_unary(inner[0], spec), f"{pre} {inner[1]}"
    for suf, spec in CFG["unary"]["suffixes"].items():
        if s.endswith(" " + suf):
            inner = _operand(s[:-len(suf) - 1])
            if inner is None:
                return None
            return _apply_unary(inner[0], spec), f"{inner[1]} {suf}"
    n = parse_number(s)
    return None if n is None else (n, _numeral(n))


def _apply_unary(a, spec):
    if spec["op"] == "power":
        return n_pow(a, spec["n"])
    r, exact = n_root(a, spec["n"])
    if exact:
        return r
    raise _Inexact(a, spec["n"])


class _Inexact(Exception):
    def __init__(self, a, n):
        self.a, self.n = a, n


def _value(t):
    """Evaluates an already normalized text -> signed number (for the functions)."""
    pieces = [p.strip() for p in re.split(r"(@\w+@)", t) if p.strip()]
    items, ops, expect, pending_neg = [], [], True, False
    for p in pieces:
        if p.startswith("@"):
            op = p[1:-1]
            if expect:
                if op == "subtract":
                    pending_neg = not pending_neg; continue
                raise ArithError("syntax")
            ops.append(op); expect = True
        else:
            o = _operand(p)
            if o is None or not expect:
                raise ArithError("syntax")
            v = o[0]
            items.append(neg(v) if pending_neg else v); pending_neg = False; expect = False
    if expect or not items:
        raise ArithError("syntax")
    return _reduce(items, ops)[0]


def _reduce(vals, ops):
    """Reduces the highest level first, from left to right. Returns (value, remainder of the last division or None)."""
    vals, ops, rem = list(vals), list(ops), None
    while ops:
        top = max(LEVELS[o] for o in ops)
        i = next(k for k, o in enumerate(ops) if LEVELS[o] == top)
        a, b, op = vals[i], vals[i + 1], ops[i]
        if op == "add":
            r = n_add(a, b)
        elif op == "subtract":
            r = n_sub(a, b)
        elif op == "multiply":
            r = n_mul(a, b)
        elif op == "divide":
            q, rr = n_divmod(a, b)
            if not is_zero(rr) and len(ops) > 1:
                raise ArithError(f"fraction:{show(a)}:{show(b)}")
            r, rem = q, (None if is_zero(rr) else rr)
        elif op == "power":
            if b[0] < 0:
                raise ArithError("negative-exponent")
            r = n_pow(a, int(show(b)))
        else:
            raise ArithError("syntax")
        vals[i:i + 2] = [r]; ops.pop(i)
    return vals[0], rem


def solve_text(text):
    """Returns Answer if the text is an arithmetic exercise; None if it is not (the LCM follows its normal path)."""
    C.look = 0
    try:
        norm_in = _normalize(text)
        had_call = bool(re.search(r"\b(" + "|".join(CFG["functions"]) + r")\(", norm_in))
        t = _resolve_calls(norm_in)
        pieces = [p.strip() for p in re.split(r"(@\w+@)", t) if p.strip()]
        has_unary = any(re.search(r"(^|\s)(" + "|".join(map(re.escape, list(CFG["unary"]["prefixes"]) + list(CFG["unary"]["suffixes"]))) + r")(\s|$)", p) for p in pieces)
        if had_call and len(pieces) == 1 and not has_unary:
            v = parse_number(pieces[0])
            if v is not None:
                return Answer(f"{norm_in.strip()} is {show(v)}.", [f"function defined as a relation between operations · {C.look} lookups"], C.look)
        if not any(p.startswith("@") for p in pieces) and not has_unary:
            return None
        items, ops, descr, expect, pending_neg, dec = [], [], [], True, False, False
        for p in pieces:
            if p.startswith("@"):
                op = p[1:-1]
                if expect:
                    if op == "subtract":
                        pending_neg = not pending_neg; descr.append("minus"); continue
                    return None
                ops.append(op); descr.append(_op_word(op)); expect = True
            else:
                if not expect:
                    return None
                try:
                    o = _operand(p)
                except Unsupported:
                    dec = True; o = None
                    items.append(None); descr.append(p); expect = False; continue
                except _Inexact as e:
                    items.append(("inexact", e)); descr.append(p); expect = False; continue
                if o is None:
                    return None
                v, d = o
                if pending_neg:
                    v = neg(v); d = "negative " + d; descr.pop(); pending_neg = False
                items.append(v); descr.append(d); expect = False
        if expect or not items:
            return None
        echo = " ".join(descr)
        if dec:
            return Answer("I only work with whole numbers for now, so I can't do " + echo + ".", ["decimals: not supported"])
        if any(isinstance(i, tuple) and i[0] == "inexact" for i in items):
            if len(items) == 1:
                _, e = items[0]
                if e.n == 2 or e.n == 3:
                    return Answer(f"{echo} is about {root_decimals(e.a, e.n, CFG['root_decimals'])}.", [f"inexact root: {C.look} lookups in the fact table"], C.look)
            return Answer("That root is not a whole number, and I can't use it in a longer calculation yet.", ["inexact root inside a chain"])
        value, rem = _reduce(items, ops)
        text_out = f"{echo} is {show(value)}" + (f" with a remainder of {show(rem)}" if rem else "") + "."
        return Answer(text_out, [f"{len(ops) + 1 if ops else 1} operands · {C.look} lookups in the fact table · derived levels: {LEVELS}"], C.look)
    except ArithError as e:
        k = str(e)
        if k == "zero":
            return Answer("I can't divide by zero: subtracting nothing never reaches the end.", ["division by zero"])
        if k == "negative-root":
            return Answer("A negative number has no real square root.", ["even root of a negative"])
        if k == "negative-exponent":
            return Answer("I can't use a negative exponent yet: it needs fractions.", ["negative exponent"])
        if k.startswith("fraction"):
            _, a, b = k.split(":")
            return Answer(f"{a} divided by {b} is not a whole number, and I don't have fractions yet.", ["fraction: not supported"])
        return None
    except TooLong:
        return Answer("That one is too long for me to count.", [f"step limit ({LIM['steps']}) or size limit"])
