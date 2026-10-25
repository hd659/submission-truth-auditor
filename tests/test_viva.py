import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truth_auditor.viva_challenger import (  # noqa: E402
    analyze_cpp,
    analyze_python,
    big_o,
    generate,
    lang_of,
    main,
)

PY = '''
def has_dup(xs):
    seen = []
    for x in xs:
        if x in seen:
            return True
        seen.append(x)
    return False

def bsearch(a, t):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == t:
            return mid
        if a[mid] < t:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

def rsum(xs):
    return 0 if not xs else xs[0] + rsum(xs[1:])

def pairs(xs):
    return [(a, b) for a in xs for b in xs]

def drain(q, sink):
    while q:
        sink.append(q.pop(0))

class Stack:
    def size(self):
        return 0
'''

CPP = r'''
#include <vector>
// int ghost(std::vector<int> v) { return 0; }
int bsearch(std::vector<int> a, int t) {
    int lo = 0, hi = a.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (a[mid] == t) return mid;
        if (a[mid] < t) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}
void dedup(std::vector<int>& v) {
    for (size_t i = 0; i < v.size(); i++)
        for (size_t j = i + 1; j < v.size(); j++)
            if (v[i] == v[j]) v.erase(v.begin() + j);
}
int* mk(int n) { const char* s = "delete"; int* p = new int[n]; return p; }
long fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
int sum(const std::vector<int>& v) { int s = 0; for (int x : v) s += x; return s; }
'''


@pytest.fixture(scope="module")
def py():
    return {p.name: p for p in analyze_python(PY)}


@pytest.fixture(scope="module")
def cpp():
    return {p.name: p for p in analyze_cpp(CPP)}


# --------------------------------------------------------------- python AST


def test_python_hidden_membership_and_early_exit(py):
    p = py["has_dup"]
    assert p.loop_depth == 1 and p.early_exit
    assert any("in seen" in e for _, e in p.hidden)
    assert big_o(p) == "O(n^2)"


def test_python_binary_search_shape(py):
    p = py["bsearch"]
    assert p.binary_search and p.edge_minus_one and p.while_conds == ["lo <= hi"]
    assert big_o(p) == "O(log n)"


def test_python_recursion_and_slice_copy(py):
    assert py["fib"].self_calls == 2 and big_o(py["fib"]).startswith("O(2^n)")
    r = py["rsum"]
    assert r.self_calls == 1 and r.slices and big_o(r) == "O(n)"


def test_python_comprehension_depth_and_mutation(py):
    assert py["pairs"].loop_depth == 2 and big_o(py["pairs"]) == "O(n^2)"
    d = py["drain"]
    assert set(d.mutates) == {"q", "sink"}
    assert any("pop(0)" in e for _, e in d.hidden)
    assert py["size"].loop_depth == 0 and big_o(py["size"]) == "O(1)"


# --------------------------------------------------------------- C++ lexical


def test_cpp_ignores_commented_functions_and_keywords(cpp):
    assert set(cpp) == {"bsearch", "dedup", "mk", "fib", "sum"}


def test_cpp_binary_search_overflow_and_copy(cpp):
    p = cpp["bsearch"]
    assert p.binary_search and p.overflow and p.edge_minus_one and p.early_exit
    assert p.by_value == ["std::vector<int> a"]
    assert p.line == 4


def test_cpp_nested_loops_hidden_erase_and_mutation(cpp):
    p = cpp["dedup"]
    assert p.loop_depth == 2 and p.mutates == ["v"]
    assert p.hidden and p.hidden[0][0] == 16
    assert big_o(p) == "O(n^3)"


def test_cpp_leak_ignores_string_literal_and_const_ref(cpp):
    assert cpp["mk"].leak  # "delete" inside a string literal must not count
    assert cpp["fib"].self_calls == 2
    s = cpp["sum"]
    assert not s.by_value and not s.mutates and s.loop_depth == 1


# ---------------------------------------------------------------- questions


def test_generate_sorted_by_severity_and_covers_categories():
    qs = generate(CPP, "cpp")
    sev = [q.severity for q in qs]
    assert sev == sorted(sev, reverse=True)
    cats = {q.category for q in qs}
    assert {"invariant", "overflow", "hidden-cost", "recursion", "memory", "copy-cost",
            "side-effect", "edge-case", "termination", "best-vs-worst", "complexity"} <= cats
    assert len(generate(CPP, "cpp", top=3)) == 3


def test_every_function_gets_baseline_question():
    qs = generate(PY, "python")
    base = {q.func for q in qs if q.category == "complexity" and q.severity == 1}
    assert base == {p.name for p in analyze_python(PY)}
    assert any("pop(0)" in q.prompt for q in qs)


def test_lang_of():
    assert lang_of("a.py") == "python" and lang_of("x/b.hpp") == "cpp" and lang_of("r.pdf") is None


# ---------------------------------------------------------------------- CLI


def test_cli_text_json_and_errors(tmp_path, capsys):
    f = tmp_path / "s.cpp"
    f.write_text(CPP)
    assert main([str(f), "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert "(2 questions)" in out and "probe:" in out

    assert main([str(f), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[str(f)][0]["severity"] == 3

    bad = tmp_path / "bad.py"
    bad.write_text("def (:\n")
    assert main([str(bad)]) == 2
    assert main([str(tmp_path / "r.pdf")]) == 2
