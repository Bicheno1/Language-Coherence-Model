"""LCM arithmetic tests.   python tests/test_arithmetic.py      (also compatible with pytest)"""
import os, random, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lcm import load
from lcm import arithmetic as A
from lcm.parse import parse
from lcm.reply import reply

random.seed(7)


def val(n):
    return int(A.show(n))


def mkn(i):
    return A.mk(str(abs(i)), 1 if i >= 0 else -1) if i != 0 else A.mk("0")


def say(q):
    r = A.solve_text(q)
    return r.text if r else None


def test_fact_tables_match_ordinary_arithmetic():
    for a in range(10):
        for b in range(10):
            assert A.ADD[(a, b)] == divmod(a + b, 10), (a, b)
            assert A.MUL[(a, b)] == divmod(a * b, 10), (a, b)
            assert A.SUB[(a, b)] == ((1, a + 10 - b) if a < b else (0, a - b)), (a, b)
    assert A.BUILD_STEPS > 0                                # the tables are built by counting, not copied


def test_precedence_levels_come_from_the_relations():
    L = A.LEVELS
    assert L["add"] == L["subtract"] < L["multiply"] == L["divide"] < L["power"] == L["root"]
    assert (L["add"], L["multiply"], L["power"]) == (1, 2, 3)


def test_signed_add_sub_mul_match_python():
    for _ in range(3000):
        a, b = random.randint(-99999, 99999), random.randint(-99999, 99999)
        x, y = mkn(a), mkn(b)
        assert val(A.n_add(x, y)) == a + b, (a, b)
        assert val(A.n_sub(x, y)) == a - b, (a, b)
        assert val(A.n_mul(x, y)) == a * b, (a, b)


def test_division_with_remainder():
    for _ in range(2000):
        a, b = random.randint(-99999, 99999), random.randint(-999, 999)
        if b == 0:
            continue
        q, r = A.n_divmod(mkn(a), mkn(b))
        q, r = val(q), val(r)
        assert a == q * b + r and abs(r) < abs(b), (a, b, q, r)
        assert q == int(a / b) if abs(a) < 2 ** 40 else True, (a, b, q)       # truncates toward zero


def test_more_by_less_sign_rule():
    assert say("what is 3 times 4") == "3 times 4 is 12."
    assert say("what is negative 3 times 4") == "-3 times 4 is -12."
    assert say("what is 3 times negative 4") == "3 times -4 is -12."
    assert say("what is negative 3 times negative 4") == "-3 times -4 is 12."
    assert say("what is 5 minus negative 3") == "5 minus -3 is 8."


def test_precedence_examples():
    assert say("what is 2 plus 3 times 4").endswith("is 14.")
    assert say("what is 2 times 3 plus 4").endswith("is 10.")
    assert say("what is 10 minus 2 times 3").endswith("is 4.")
    assert say("what is 12 divided by 4 times 3").endswith("is 9.")
    assert say("what is 2 plus 3 times 4 minus 5").endswith("is 9.")
    assert say("what is 2 times 3 to the power of 2").endswith("is 18.")
    assert say("what is 3 plus 2 to the power of 3 times 2").endswith("is 19.")


def test_random_expressions_match_python():
    names = {"+": "plus", "-": "minus", "*": "times"}
    for _ in range(1500):
        k = random.randint(2, 4)
        nums = [random.randint(0, 99) for _ in range(k)]
        ops = [random.choice("+-*") for _ in range(k - 1)]
        py = "".join(f"{n}{o}" for n, o in zip(nums, ops + [""]))
        txt = " ".join(str(n) + (f" {names[o]}" if o else "") for n, o in zip(nums, ops + [None]))
        assert say(txt).endswith(f"is {eval(py)}."), (txt, say(txt))


def test_powers_and_roots():
    assert say("what is 2 to the power of 10").endswith("is 1024.")
    assert say("what is 2 to the power of 100").endswith("is 1267650600228229401496703205376.")
    assert say("what is the square root of 144").endswith("is 12.")
    assert say("what is the cube root of 27").endswith("is 3.")
    assert say("what is the cube root of negative 27").endswith("is -3.")
    assert say("what is the square root of 2") == "the square root of 2 is about 1.41."
    assert say("what is the square root of 150") == "the square root of 150 is about 12.24."
    assert say("12 squared") == "12 squared is 144."
    assert say("what is 3 cubed") == "3 cubed is 27."


def test_functions_are_relations_between_operations():
    assert say("double(7)") == "double(7) is 14."
    assert say("square(9)") == "square(9) is 81."
    assert say("what is cube(double(2))") == "cube(double(2)) is 64."
    assert say("perimeter(3, 5)") == "perimeter(3, 5) is 16."
    assert say("what is hypot2(3, 4)") == "hypot2(3, 4) is 25."
    assert say("what is 3 plus double(4)").endswith("is 11.")


def test_language_forms():
    assert say("what is twenty seven plus fifteen").endswith("is 42.")
    assert say("what is twenty-seven plus fifteen").endswith("is 42.")
    assert say("what is one hundred twenty three plus two thousand").endswith("is 2123.")
    assert say("123*45").endswith("is 5535.")
    assert say("99999 x 9999").endswith("is 999890001.")
    assert say("10 - 3 - 2").endswith("is 5.")
    assert say("-5 + 8").endswith("is 3.")
    assert say("multiply 6 by 7") == "6 times 7 is 42."
    assert say("add 5 to 3") == "3 plus 5 is 8."
    assert say("subtract 2 from 10") == "10 minus 2 is 8."
    assert say("divide 20 by 5") == "20 divided by 5 is 4."
    assert say("what is 20 divided by 6") == "20 divided by 6 is 3 with a remainder of 2."


def test_what_has_no_answer_is_still_answered_in_text():
    assert "divide by zero" in say("what is 5 divided by 0")
    assert "fractions" in say("what is 7 divided by 2 plus 1")
    assert "whole numbers" in say("what is 3.5 plus 2")
    assert "negative exponent" in say("what is 2 to the power of negative 1")
    assert "no real square root" in say("what is the square root of negative 4")
    assert "too long" in say("what is 9 to the power of 100000")


def test_it_does_not_hijack_normal_inputs():
    for q in ["tell me a story about a cat", "what is a bridge?", "what do you think about ducks?", "do you like cats?", "how are you?",
              "tell me a story about a cat and a dog", "what is 5", "I jumped over the fence", "it was 1990 and 1991", "describe a desert",
              "what is the capital of France?", "tell me about 7 over there"]:
        assert say(q) is None, q


def test_cost_depends_on_digits_not_value():
    assert A.solve_text("99999 x 9999").steps < 200            # counting one by one would take ~10^9 steps
    assert A.solve_text("123456 plus 654321").steps <= 12
    assert A.solve_text("what is 123 times 45").steps < 40


def test_responder_integration_and_no_regression():
    T = load()
    r = reply(T, parse("what is 123 times 45", T))[0]
    assert r.kind == "arithmetic" and r.text == "123 times 45 is 5535." and not r.error
    s = reply(T, parse("tell me a story about a cat", T))[0]
    assert s.kind == "story" and s.text.startswith("There was a cat")
    o = reply(T, parse("what do you think about ducks?", T), seed=1)[0]
    assert o.kind == "opinion"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"  OK    {name}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} arithmetic tests pass")
    sys.exit(1 if failed else 0)
