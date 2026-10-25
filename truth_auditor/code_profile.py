"""Per-function complexity profiles: Python via `ast`, C++ via comment/string-stripped lexing.

Signals: loop nesting, hidden O(n) ops inside loops, recursion fan-out, slicing copies,
binary-search shape / midpoint overflow, early exits, `len-1` edges, argument mutation,
by-value containers, `new` without `delete`.
"""
from __future__ import annotations

import ast, re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FuncProfile:
    name: str
    lang: str
    line: int
    loop_depth: int = 0
    self_calls: int = 0
    hidden: List[Tuple[int, str]] = field(default_factory=list)
    slices: List[Tuple[int, str]] = field(default_factory=list)
    mutates: List[str] = field(default_factory=list)
    by_value: List[str] = field(default_factory=list)
    while_conds: List[str] = field(default_factory=list)
    binary_search: bool = False
    overflow: bool = False
    edge_minus_one: bool = False
    early_exit: bool = False
    leak: bool = False


# ------------------------------------------------------------------ Python AST
_MUTATORS = {"append", "extend", "insert", "pop", "remove", "sort", "reverse", "clear", "update", "add", "discard", "setdefault"}
_LINEAR_ATTR = {"index", "count", "remove", "sort"}
_LINEAR_CALL = {"sorted", "min", "max", "sum", "list", "reversed"}


def _short(node: ast.AST) -> str:
    s = ast.unparse(node)
    return s if len(s) <= 48 else s[:45] + "..."


class _PyFunc(ast.NodeVisitor):
    def __init__(self, fn):
        self.fn, self.depth = fn, 0
        self.p = FuncProfile(fn.name, "python", fn.lineno)
        self.params = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs} - {"self", "cls"}

    def _enter(self, node, n=1):
        self.depth += n
        self.p.loop_depth = max(self.p.loop_depth, self.depth)
        self.generic_visit(node)
        self.depth -= n

    def visit_For(self, node): self._enter(node)
    visit_AsyncFor = visit_For

    def visit_While(self, node):
        self.p.while_conds.append(_short(node.test))
        self._enter(node)

    def _comp(self, node): self._enter(node, len(node.generators))
    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp

    def visit_FunctionDef(self, node):
        if node is self.fn:
            self.generic_visit(node)
    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = lambda self, node: None  # noqa: E731

    def _exit(self, node):
        self.p.early_exit |= self.depth > 0
        self.generic_visit(node)
    visit_Break = visit_Return = _exit

    def visit_Call(self, node):
        f, p = node.func, self.p
        if (isinstance(f, ast.Name) and f.id == self.fn.name) or (
            isinstance(f, ast.Attribute) and f.attr == self.fn.name and isinstance(f.value, ast.Name) and f.value.id == "self"):
            p.self_calls += 1
        if isinstance(f, ast.Attribute):
            first0 = bool(node.args) and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0
            if self.depth and (f.attr in _LINEAR_ATTR or (f.attr in ("pop", "insert") and first0)):
                p.hidden.append((node.lineno, _short(node)))
            if f.attr in _MUTATORS and isinstance(f.value, ast.Name) and f.value.id in self.params and f.value.id not in p.mutates:
                p.mutates.append(f.value.id)
        elif isinstance(f, ast.Name) and self.depth and f.id in _LINEAR_CALL and node.args:
            p.hidden.append((node.lineno, _short(node)))
        self.generic_visit(node)

    def visit_Compare(self, node):
        if self.depth and any(isinstance(o, (ast.In, ast.NotIn)) for o in node.ops) and \
                not any(isinstance(c, (ast.Set, ast.Dict, ast.Constant)) for c in node.comparators):
            self.p.hidden.append((node.lineno, _short(node)))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Slice):
            self.p.slices.append((node.lineno, _short(node)))
        if isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and \
                node.value.id in self.params and node.value.id not in self.p.mutates:
            self.p.mutates.append(node.value.id)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, (ast.FloorDiv, ast.RShift)) and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Add):
            self.p.binary_search = True
        if isinstance(node.op, ast.Sub) and isinstance(node.left, ast.Call) and getattr(node.left.func, "id", "") == "len":
            self.p.edge_minus_one = True
        self.generic_visit(node)


def analyze_python(src: str) -> List[FuncProfile]:
    tree = ast.parse(src)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            v = _PyFunc(n)
            v.visit(n)
            out.append(v.p)
    return sorted(out, key=lambda p: p.line)


# ------------------------------------------------------------------ C++ lexical
_CPP_FN = re.compile(r"^[ \t]*[\w:<>,*&\s]*?[\w>*&]\s+[*&]*(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{}()]*)\)\s*(?:const\s*)?(?:noexcept\s*)?\{", re.M)
_CPP_KW = {"if", "for", "while", "switch", "return", "catch", "else", "do"}
_CONTAINER = re.compile(r"\b(?:std::)?(?:vector|string|map|set|unordered_map|unordered_set|deque|list)\b")


def _strip_cpp(src: str) -> str:
    blank = lambda m: re.sub(r"[^\n]", " ", m.group(0))  # noqa: E731
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', blank, src, flags=re.S)


def _match(code: str, i: int, o: str, c: str) -> int:
    depth = 0
    for j in range(i, len(code)):
        depth += (code[j] == o) - (code[j] == c)
        if depth == 0:
            return j
    return len(code) - 1


def _stmt_end(code: str, i: int) -> int:
    while i < len(code) and code[i].isspace():
        i += 1
    if code.startswith("{", i):
        return _match(code, i, "{", "}")
    m = re.match(r"(?:for|while)\s*\(", code[i:])
    if m:
        return _stmt_end(code, _match(code, i + m.end() - 1, "(", ")") + 1)
    j = code.find(";", i)
    return j if j >= 0 else len(code) - 1


def _cpp_profile(code: str, m: re.Match) -> FuncProfile:
    name, start = m["name"], m.end() - 1
    end = _match(code, start, "{", "}")
    body = code[start:end + 1]
    line_of = lambda pos: code.count("\n", 0, start + pos) + 1  # noqa: E731
    p = FuncProfile(name, "cpp", code.count("\n", 0, m.start("name")) + 1)
    for prm in filter(None, (s.strip() for s in m["params"].split(","))):
        if _CONTAINER.search(prm) and "&" not in prm and "*" not in prm:
            p.by_value.append(prm)
        elif "&" in prm and "const" not in prm:
            p.mutates.append(re.findall(r"\w+", prm)[-1])
    spans = [(lm.start(), _stmt_end(body, _match(body, lm.end() - 1, "(", ")") + 1))
             for lm in re.finditer(r"\b(?:for|while)\s*\(", body)]
    depth_at = lambda pos: sum(s < pos <= e for s, e in spans)  # noqa: E731
    p.loop_depth = max((depth_at(s + 1) for s, _ in spans), default=0)
    p.while_conds = [body[w.end():_match(body, w.end() - 1, "(", ")")].strip() for w in re.finditer(r"\bwhile\s*\(", body)]
    p.self_calls = len(re.findall(rf"(?<![\w.]){re.escape(name)}\s*\(", body))
    for h in re.finditer(r"\.\s*(?:erase|insert|find|count)\s*\(|\b(?:std::)?(?:find|sort|count)\s*\(", body):
        if depth_at(h.start()):
            p.hidden.append((line_of(h.start()), h.group(0).strip(" (") + "(...)"))
    p.early_exit = any(depth_at(x.start()) for x in re.finditer(r"\b(?:break|return)\b", body))
    p.overflow = bool(re.search(r"\(\s*\w+\s*\+\s*\w+\s*\)\s*(?:/\s*2|>>\s*1)", body))
    p.binary_search = p.overflow or bool(re.search(r"\+\s*\(\s*\w+\s*-\s*\w+\s*\)\s*/\s*2", body))
    p.edge_minus_one = bool(re.search(r"\.\s*(?:size|length)\s*\(\s*\)\s*-\s*1", body))
    p.leak = bool(re.search(r"\bnew\b", body)) and not re.search(r"\bdelete\b", body)
    return p


def analyze_cpp(src: str) -> List[FuncProfile]:
    code = _strip_cpp(src)
    return [_cpp_profile(code, m) for m in _CPP_FN.finditer(code) if m["name"] not in _CPP_KW]
