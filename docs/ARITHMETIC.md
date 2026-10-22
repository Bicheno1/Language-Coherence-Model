# LCM arithmetic

Module `lcm/arithmetic.py` + data `data/LCM_arithmetic.json` + tests `tests/test_arithmetic.py` (14). It is wired into `reply()`:
if the text is an arithmetic exercise it is answered by this module; otherwise the LCM follows its normal path (stories, opinions, likes).

## How it works
1. **Operations are entries related to each other** (`data/LCM_arithmetic.json`): `add` is the base; `subtract` is the inverse of `add`; `multiply` repeats `add`;
   `divide` is the inverse of `multiply`; `power` repeats `multiply`; `root` is the inverse of `power`. The **precedence level comes from that chain** (how many times the base is repeated):
   1 for add and subtract, 2 for multiply and divide, 3 for power and root. It is not written by hand.
2. **Signs follow the logic of the axes:** up adds and down subtracts (vertical); right is positive and left is negative (horizontal). The effective step is the product of the two
   ("plus times minus = minus"): *5 minus negative 3* = 8.
3. **Calculation goes digit by digit** with three fact tables (0-9 + 0-9, 0-9 − 0-9, 0-9 × 0-9) built once by **counting** (2,925 steps), plus carry, borrow,
   shifts and long division. The cost depends on the number of digits, not on the value.
4. **Functions** = entries related to operations and to other functions (`double`, `square`, `cube`, `sum2`, `perimeter`, `hypot2`, in the JSON). Composing them replaces parentheses.

## What it understands
`what is 123 times 45` · `2 plus 3 times 4` · `twenty-seven plus fifteen` · `123*45` · `99999 x 9999` · `10 - 3 - 2` · `-5 + 8` · `multiply 6 by 7` · `add 5 to 3` ·
`subtract 2 from 10` · `20 divided by 6` (with remainder) · `2 to the power of 100` · `the square root of 144` · `the cube root of 27` · `12 squared` · `double(7)` · `cube(double(2))`.

## Measured
| Exercise | Counting one by one | Fact-table lookups |
|---|---|---|
| 123456 + 654321 | 654,321 steps | 6 |
| 123 × 45 | 5,535 | 24 |
| 4821 × 3976 | 19,168,296 | 77 |
| 99999 × 9999 | 999,890,001 | 107 |

No differences from Python in: 3,000 signed operations (+, −, ×), 2,000 divisions with remainder and 1,500 random expressions of 2 to 4 numbers.

## Limits (unresolved)
- **Integers only.** There are no fractions or decimals as input: `7 divided by 2 plus 1` and `3.5 plus 2` are answered with a text saying so. Inexact roots are given with 2 decimals.
- **Chained powers** (2^3^2): read from left to right; this is a convention, it does not come from the relations. **There are no parentheses** in the language.
- **The relations were written by a person** (definitions); they are not learned from the corpus. Only the **format** of the related system is used.
- **Reading number words** (`twenty seven`) uses ordinary Python integers; the calculation does not.
- **Safety limits** (in the JSON): 3,000,000 lookups, exponent ≤ 500, 3,000 digits. Past those limits the answer is "too long for me to count".
- In the output of `python -m lcm reply`, the coherence of an arithmetic answer shows as 0.0 (there are no plane nodes in it); this is cosmetic only.
- It was not tested with the project's `parse.py` beyond what `reply()` does (the module reads `frame.entrada`, the original text).

## Open items
- Fractions and decimals (define what a fraction is as a relation).
- Order and parentheses in the language; chained powers.
- Have `parse` mark the "calculation" form instead of the module reading the raw text.
