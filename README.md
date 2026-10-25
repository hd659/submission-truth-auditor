# submission-truth-auditor

Pre-submission checks for coursework, in two parts:

1. **Numeric cross-verifier** (`numeric_diff`): every number in the report (Markdown / text / PDF) is compared with the deterministic output of your scripts. It catches the classic "the report says 127 rows but the pipeline now prints 118" drift.
2. **Viva challenger** (`viva_challenger`): profiles each Python / C++ function for complexity invariants and generates the adversarial questions a tutor is likely to ask in a live-coding Viva or rediscuss.

Pure Python 3.9+, standard library only. `pypdf` is optional and only needed for PDF reports.

## Layout

```
truth_auditor/
  numeric_diff.py     claim extraction + rounding-aware cross-verification + CLI
  code_profile.py     per-function profiles: Python via ast, C++ via stripped lexing
  viva_challenger.py  profile -> ranked Viva questions + CLI
tests/
  test_numeric_diff.py
  test_viva.py
```

## 1. Numeric cross-verifier

```bash
python -m truth_auditor.numeric_diff report.pdf --truth outputs.txt
python -m truth_auditor.numeric_diff report.md  --run "bash pipeline.sh" --json
```

- Truth sources: `key=value`, `key: value`, TAB-separated, `wc -l`-style `127 file.csv`, or a JSON object (flattened with `.`).
- Report claims are labelled using the surrounding words, `key: value` lines, and table rows. Dates, clock times, versions and identifiers such as `FIT1043` or `A2` are masked out before extraction.
- Tolerance depends on the precision the report states: `0.85` accepts anything in `[0.845, 0.855]`, while `0.850` is stricter. `3.4%` also matches a fractional truth of `0.0341`.
- Exit code `0` means clean, `1` means a MISMATCH or MISSING finding, and `2` means a usage or I/O error. That makes it usable as a final gate before upload.

```
[MISMATCH] rows_after_cleaning   expected=118  report=L4 '127' delta=9
[MATCH   ] accuracy              expected=0.8512 report=L5 '0.85'
summary: 4 match, 1 mismatch, 0 missing
```

## 2. Viva challenger

```bash
python -m truth_auditor.viva_challenger solution.py list_adt.cpp --top 10
python -m truth_auditor.viva_challenger solution.py --json > viva.json
```

| Signal detected | Question category | Example probe |
|---|---|---|
| nested loops / comprehensions | `complexity` | justify O(n^k), construct the worst-case input |
| `x in list`, `.index`, `.pop(0)`, `sorted`, `erase`, `find` inside a loop | `hidden-cost` | an O(n) op inside a loop adds a factor of n |
| two or more self-calls per frame | `recursion` | recursion tree, memoisation |
| recursion on a slice `xs[1:]` | `hidden-cost` | O(k) copy per level, pass indices instead |
| `(lo + hi) // 2` shape | `invariant`, `overflow` | loop invariant, `lo + (hi - lo) / 2` |
| `while cond` | `termination` | which variant strictly decreases |
| `return` / `break` inside a loop | `best-vs-worst` | best case vs worst case |
| `len(x) - 1`, `size() - 1` | `edge-case` | empty input, `size_t` underflow |
| mutating a parameter / non-const `&` | `side-effect` | aliasing as seen by the caller |
| container passed by value (C++) | `copy-cost` | O(n) copy per call, use `const&` |
| `new` without `delete` (C++) | `memory` | ownership, RAII, Valgrind |

Every function also gets a baseline question asking for its time and space complexity, with the expected heuristic bound. Your report and your spoken answer should both agree with it. Questions are ranked by severity (`***` high, `*` low).

```
 1. [***] invariant     bsearch:L4  State the loop invariant of the binary search in `bsearch`. Why does it terminate when lo == hi?
 2. [***] hidden-cost   dedup:L13   Line 16: `.erase(...)` sits inside a loop. What is its own cost, and what does that do to the overall bound?
```

### Limits

- The C++ analysis is lexical. Comments and string literals are stripped and braces are matched, but templates, macros and out-of-class method definitions with unusual signatures may be missed.
- `big_o` is a heuristic for rehearsal and does not prove anything. Amortised structures and loops whose bounds depend on the outer index (for example `j < i`) still need your own argument.

## Tests

```bash
python -m pytest -q
```

## Suggested pre-submission routine

1. Re-run the pipeline from a clean shell and pipe its output to `outputs.txt`.
2. `numeric_diff report.pdf --truth outputs.txt` should exit with `0`.
3. `viva_challenger src/*.py src/*.cpp --top 15`, then rehearse an answer to each question aloud.
