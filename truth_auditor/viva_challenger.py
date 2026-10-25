"""Adversarial Viva / Rediscuss question generator for Python and C++ submissions.

Turns `code_profile.FuncProfile` invariants into the questions a tutor will ask:
justify the bound, expose hidden costs, prove termination, defend edge cases.

CLI:  python -m truth_auditor.viva_challenger solution.py lib.cpp [--json] [--top 10]
"""
from __future__ import annotations

import argparse, json, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .code_profile import FuncProfile, analyze_cpp, analyze_python

__all__ = ["Question", "FuncProfile", "analyze_python", "analyze_cpp", "big_o", "challenge", "generate", "lang_of", "main"]


@dataclass(frozen=True)
class Question:
    severity: int  # 3 high, 2 medium, 1 low
    category: str
    func: str
    line: int
    prompt: str
    probe: str  # what the tutor is fishing for


def big_o(p: FuncProfile) -> str:
    """Heuristic time bound a tutor would expect you to defend."""
    if p.self_calls >= 2:
        return "O(2^n) unless memoised"
    if p.binary_search and p.loop_depth <= 1 and not p.hidden:
        return "O(log n)"
    d = p.loop_depth + (1 if p.hidden else 0) + (1 if p.self_calls == 1 and p.slices else 0)
    return "O(1)" if d == 0 else "O(n)" if d == 1 else f"O(n^{d})"


def challenge(p: FuncProfile) -> List[Question]:
    q, f, L = [], p.name, p.line
    add = lambda sev, cat, prompt, probe: q.append(Question(sev, cat, f, L, prompt, probe))  # noqa: E731
    bound = big_o(p)
    if p.loop_depth >= 2:
        add(3, "complexity", f"`{f}` nests loops {p.loop_depth} deep. Justify {bound} and name the input that hits the worst case.",
            "worst-case input construction; whether inner bound depends on outer index")
    for ln, expr in p.hidden:
        add(3, "hidden-cost", f"Line {ln}: `{expr}` sits inside a loop. What is its own cost, and what does that do to the overall bound?",
            "O(n) membership/search/shift inside loop -> extra factor of n; fix with set/dict/deque")
    if p.self_calls >= 2:
        add(3, "recursion", f"`{f}` calls itself {p.self_calls} times per frame. Draw the recursion tree: how many calls for n=30?",
            "exponential fan-out, overlapping subproblems, memoisation / DP")
    elif p.self_calls == 1:
        add(2, "recursion", f"What is the base case of `{f}`, and what is the maximum recursion depth before a stack overflow?",
            "termination argument, auxiliary O(depth) stack space")
    if p.slices and p.self_calls:
        add(3, "hidden-cost", f"`{f}` recurses on a slice (`{p.slices[0][1]}`). How much memory and time does copying cost per level?",
            "slicing copies O(k); pass indices instead")
    if p.binary_search:
        add(3, "invariant", f"State the loop invariant of the binary search in `{f}`. Why does it terminate when lo == hi?",
            "invariant on [lo, hi], strictly shrinking interval, off-by-one on mid +/- 1")
    if p.overflow:
        add(2, "overflow", f"In `{f}`, can `(lo + hi) / 2` overflow for int? Rewrite it safely.",
            "lo + (hi - lo) / 2; signed overflow is UB in C++")
    for cond in p.while_conds[:2]:
        add(2, "termination", f"Prove `while ({cond})` in `{f}` terminates. Which variable strictly decreases?",
            "variant function / measure argument")
    if p.early_exit and p.loop_depth:
        add(2, "best-vs-worst", f"`{f}` can exit early. Give its best case and worst case separately - are they different Big-O?",
            "best/worst/average distinction; early-exit does not change worst case")
    if p.edge_minus_one:
        add(2, "edge-case", f"`{f}` uses `size - 1`. What happens on an empty input?",
            "len(x)-1 == -1 indexes last element in Python; size()-1 underflows size_t in C++")
    for name in p.mutates:
        add(2, "side-effect", f"`{f}` mutates its argument `{name}`. What does the caller observe afterwards? Is that intended?",
            "aliasing, pass-by-reference semantics, defensive copy cost")
    for prm in p.by_value:
        add(2, "copy-cost", f"`{f}` takes `{prm}` by value. What does each call cost? Would `const&` change the bound?",
            "O(n) copy per call; const reference")
    if p.leak:
        add(3, "memory", f"`{f}` calls `new` without `delete`. Who owns that memory, and what does Valgrind report?",
            "ownership, RAII, unique_ptr / destructor")
    add(1, "complexity", f"Without running it: what are the time and space complexity of `{f}`? (expected answer ~ {bound})",
        "examiner baseline question - answer must match your report")
    return q


def analyze_source(src: str, lang: str) -> List[FuncProfile]:
    return analyze_python(src) if lang == "python" else analyze_cpp(src)


def lang_of(path: str | Path) -> Optional[str]:
    ext = Path(path).suffix.lower()
    return "python" if ext == ".py" else "cpp" if ext in {".cpp", ".cc", ".cxx", ".hpp", ".h"} else None


def generate(src: str, lang: str, top: Optional[int] = None) -> List[Question]:
    qs = [x for p in analyze_source(src, lang) for x in challenge(p)]
    qs.sort(key=lambda x: (-x.severity, x.line, x.category))
    return qs[:top] if top else qs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="viva_challenger", description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    result = {}
    for fp in a.files:
        lang = lang_of(fp)
        try:
            if lang is None:
                raise ValueError("unsupported extension")
            result[fp] = generate(Path(fp).read_text(encoding="utf-8"), lang, a.top)
        except (OSError, SyntaxError, ValueError) as e:
            print(f"error: {fp}: {e}", file=sys.stderr)
            return 2
    if a.json:
        print(json.dumps({k: [asdict(x) for x in v] for k, v in result.items()}, indent=2))
    else:
        for fp, qs in result.items():
            print(f"== {fp} ({len(qs)} questions)")
            for i, x in enumerate(qs, 1):
                print(f"{i:>2}. [{'*' * x.severity:<3}] {x.category:<13} {x.func}:L{x.line}  {x.prompt}\n      probe: {x.probe}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
