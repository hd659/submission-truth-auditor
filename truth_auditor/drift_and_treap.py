"""Temporal drift, carbon amortization and Treap invariant auditors.

Three deterministic checks for claims that commonly appear in reports or Viva:

1. drift_audit(records, window)    -> per-window recall + degradation flags
   (FIT1061-style "model degrades after deployment" claims, C4 model cards).
2. carbon_amortization(...)        -> training CO2 amortized over inferences,
   plus the break-even inference count where serving emissions == training.
3. verify_treap(root, heap="max")  -> BST-order / heap-priority / size-field
   violations with node paths (FIT1008 live-coding Viva).

All results are plain dataclasses so they can be cross-checked against
report numbers with `within_tolerance` (same rounding-aware rule as numeric_diff).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 1. Temporal concept-drift recall degradation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowStat:
    start: float
    end: float
    n: int
    positives: int
    recall: Optional[float]   # None when the window has no positives
    degraded: bool


@dataclass
class DriftReport:
    baseline_recall: Optional[float]
    windows: List[WindowStat]
    max_drop: float = 0.0
    first_degraded: Optional[int] = None

    @property
    def drifted(self) -> bool:
        return self.first_degraded is not None


def _recall(pairs: Sequence[Tuple[int, int]]) -> Tuple[int, Optional[float]]:
    pos = [p for t, p in pairs if t == 1]
    return len(pos), (sum(pos) / len(pos) if pos else None)


def drift_audit(records: Iterable[Tuple[float, int, int]], window: float,
                max_drop: float = 0.10, baseline_windows: int = 1) -> DriftReport:
    """records: (timestamp, y_true, y_pred) with binary labels.

    Records are sorted by timestamp and bucketed into fixed-width windows
    [t0 + k*window, t0 + (k+1)*window). Baseline recall is pooled over the
    first `baseline_windows` non-empty windows; a later window is degraded when
    baseline - recall > max_drop (absolute drop, not relative).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    rows = sorted(records, key=lambda r: r[0])
    if not rows:
        return DriftReport(None, [])
    for _, t, p in rows:
        if t not in (0, 1) or p not in (0, 1):
            raise ValueError("labels must be binary 0/1")
    t0 = rows[0][0]
    buckets: dict = {}
    for ts, t, p in rows:
        buckets.setdefault(int((ts - t0) // window), []).append((t, p))
    keys = sorted(buckets)
    base_pairs = [pp for k in keys[:baseline_windows] for pp in buckets[k]]
    _, base = _recall(base_pairs)
    report = DriftReport(base, [])
    for i, k in enumerate(keys):
        pos, rec = _recall(buckets[k])
        drop = (base - rec) if (base is not None and rec is not None) else 0.0
        bad = i >= baseline_windows and drop > max_drop + 1e-12
        report.windows.append(WindowStat(t0 + k * window, t0 + (k + 1) * window,
                                         len(buckets[k]), pos, rec, bad))
        if i >= baseline_windows:
            report.max_drop = max(report.max_drop, drop)
        if bad and report.first_degraded is None:
            report.first_degraded = i
    return report


# --------------------------------------------------------------------------- #
# 2. Carbon amortization
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CarbonReport:
    training_kg: float
    per_inference_g: float
    amortized_g_per_inference: float   # (training + serving) / n_inferences
    total_kg: float
    break_even_inferences: Optional[int]  # serving CO2 >= training CO2


def carbon_amortization(train_kwh: float, grid_g_per_kwh: float,
                        n_inferences: int, infer_wh: float = 0.0,
                        pue: float = 1.0) -> CarbonReport:
    """CO2 = energy * PUE * grid intensity. Units: kWh, gCO2e/kWh, Wh/inference."""
    if min(train_kwh, grid_g_per_kwh, infer_wh) < 0 or pue < 1.0:
        raise ValueError("energies/intensity must be >= 0 and PUE >= 1")
    if n_inferences <= 0:
        raise ValueError("n_inferences must be positive")
    train_g = train_kwh * pue * grid_g_per_kwh
    per_inf_g = (infer_wh / 1000.0) * pue * grid_g_per_kwh
    total_g = train_g + per_inf_g * n_inferences
    be = math.ceil(train_g / per_inf_g) if per_inf_g > 0 else None
    return CarbonReport(train_g / 1000.0, per_inf_g, total_g / n_inferences,
                        total_g / 1000.0, be)


def within_tolerance(claimed: float, actual: float, decimals: Optional[int] = None) -> bool:
    """Accept a claim if it equals `actual` rounded to the claim's precision."""
    if decimals is None:
        s = repr(float(claimed))
        decimals = 0 if s.endswith(".0") else len(s.split(".")[1])
    return abs(claimed - actual) <= 0.5 * 10 ** (-decimals) + 1e-12


# --------------------------------------------------------------------------- #
# 3. Treap invariant verifier
# --------------------------------------------------------------------------- #


@dataclass
class TreapNode:
    key: object
    priority: float
    left: Optional["TreapNode"] = None
    right: Optional["TreapNode"] = None
    size: Optional[int] = None   # optional augmented subtree size


@dataclass
class TreapAudit:
    violations: List[str] = field(default_factory=list)
    size: int = 0
    height: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations


def verify_treap(root: Optional[TreapNode], heap: str = "max") -> TreapAudit:
    """Check strict BST order on keys, heap order on priorities, size fields.

    Iterative (explicit stack) so degenerate chains do not hit recursion limits.
    Paths are reported as 'root', 'root.L', 'root.L.R', ...
    """
    if heap not in ("max", "min"):
        raise ValueError("heap must be 'max' or 'min'")
    better = (lambda a, b: a >= b) if heap == "max" else (lambda a, b: a <= b)
    audit = TreapAudit()
    if root is None:
        return audit
    sizes: dict = {}
    # (node, path, lo, hi, depth, expanded)
    stack = [(root, "root", None, None, 1, False)]
    seen = set()
    while stack:
        node, path, lo, hi, depth, expanded = stack.pop()
        if expanded:
            n = 1 + sizes.get(id(node.left), 0) + sizes.get(id(node.right), 0)
            sizes[id(node)] = n
            if node.size is not None and node.size != n:
                audit.violations.append(f"{path}: size field {node.size} != actual {n}")
            continue
        if id(node) in seen:
            audit.violations.append(f"{path}: cycle/shared node detected")
            continue
        seen.add(id(node))
        audit.height = max(audit.height, depth)
        if (lo is not None and not node.key > lo) or (hi is not None and not node.key < hi):
            audit.violations.append(f"{path}: key {node.key!r} outside BST range ({lo!r}, {hi!r})")
        stack.append((node, path, lo, hi, depth, True))
        for child, tag, clo, chi in ((node.left, "L", lo, node.key),
                                     (node.right, "R", node.key, hi)):
            if child is None:
                continue
            if not better(node.priority, child.priority):
                audit.violations.append(
                    f"{path}.{tag}: priority {child.priority} breaks {heap}-heap "
                    f"under parent {node.priority}")
            stack.append((child, f"{path}.{tag}", clo, chi, depth + 1, False))
    audit.size = sizes.get(id(root), 0)
    return audit


def treap_insert(root: Optional[TreapNode], key, priority: float) -> TreapNode:
    """Reference max-heap Treap insert with rotations (maintains size fields)."""
    def sz(n): return n.size if n else 0

    def fix(n):
        n.size = 1 + sz(n.left) + sz(n.right)
        return n

    def ins(n):
        if n is None:
            return TreapNode(key, priority, size=1)
        if key < n.key:
            n.left = ins(n.left)
            if n.left.priority > n.priority:          # rotate right
                l = n.left; n.left = l.right; l.right = fix(n); n = l
        elif key > n.key:
            n.right = ins(n.right)
            if n.right.priority > n.priority:         # rotate left
                r = n.right; n.right = r.left; r.left = fix(n); n = r
        return fix(n)
    return ins(root)
