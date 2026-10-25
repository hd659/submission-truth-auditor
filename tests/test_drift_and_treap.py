import random

import pytest

from truth_auditor.drift_and_treap import (
    TreapNode, carbon_amortization, drift_audit, treap_insert, verify_treap,
    within_tolerance,
)


# ----------------------------- drift ------------------------------------- #

def _window(t0, n_pos, hits, n_neg=2):
    rows = [(t0 + i * 0.01, 1, 1 if i < hits else 0) for i in range(n_pos)]
    rows += [(t0 + 0.5 + i * 0.01, 0, 0) for i in range(n_neg)]
    return rows


def test_drift_detects_first_degraded_window():
    recs = _window(0, 10, 9) + _window(1, 10, 9) + _window(2, 10, 6) + _window(3, 10, 5)
    random.Random(0).shuffle(recs)               # order must not matter
    rep = drift_audit(recs, window=1.0, max_drop=0.10)
    assert rep.baseline_recall == pytest.approx(0.9)
    assert [w.recall for w in rep.windows] == pytest.approx([0.9, 0.9, 0.6, 0.5])
    assert rep.first_degraded == 2 and rep.drifted
    assert rep.max_drop == pytest.approx(0.4)
    assert [w.n for w in rep.windows] == [12, 12, 12, 12]


def test_drift_boundary_drop_equal_threshold_not_flagged():
    recs = _window(0, 10, 9) + _window(1, 10, 8)   # drop exactly 0.10
    rep = drift_audit(recs, window=1.0, max_drop=0.10)
    assert not rep.drifted


def test_drift_window_without_positives_has_none_recall():
    recs = _window(0, 4, 4) + [(1.2, 0, 1), (1.3, 0, 0)]
    rep = drift_audit(recs, window=1.0)
    assert rep.windows[1].recall is None and not rep.windows[1].degraded


def test_drift_pooled_baseline_and_validation():
    recs = _window(0, 10, 10) + _window(1, 10, 8) + _window(2, 10, 7)
    rep = drift_audit(recs, window=1.0, baseline_windows=2, max_drop=0.15)
    assert rep.baseline_recall == pytest.approx(0.9)
    assert rep.first_degraded == 2 and rep.max_drop == pytest.approx(0.2)
    assert not drift_audit(recs, 1.0, max_drop=0.25, baseline_windows=2).drifted
    assert drift_audit([], 1.0).windows == []
    with pytest.raises(ValueError):
        drift_audit(recs, window=0)
    with pytest.raises(ValueError):
        drift_audit([(0, 2, 1)], window=1)


# ----------------------------- carbon ------------------------------------ #

def test_carbon_amortization_numbers():
    r = carbon_amortization(train_kwh=100, grid_g_per_kwh=500, n_inferences=1_000_000,
                            infer_wh=0.2, pue=1.5)
    assert r.training_kg == pytest.approx(75.0)          # 100*1.5*500 g
    assert r.per_inference_g == pytest.approx(0.15)       # 0.0002*1.5*500
    assert r.total_kg == pytest.approx(225.0)
    assert r.amortized_g_per_inference == pytest.approx(0.225)
    assert r.break_even_inferences == 500_000


def test_carbon_no_serving_cost_and_validation():
    r = carbon_amortization(10, 400, 1000)
    assert r.break_even_inferences is None
    assert r.amortized_g_per_inference == pytest.approx(4.0)
    for bad in [dict(train_kwh=-1, grid_g_per_kwh=1, n_inferences=1),
                dict(train_kwh=1, grid_g_per_kwh=1, n_inferences=0),
                dict(train_kwh=1, grid_g_per_kwh=1, n_inferences=1, pue=0.9)]:
        with pytest.raises(ValueError):
            carbon_amortization(**bad)


def test_within_tolerance_rounding_aware():
    assert within_tolerance(0.23, 0.225)
    assert within_tolerance(225, 225.4)
    assert not within_tolerance(127, 118)
    assert not within_tolerance(0.22, 0.226)


# ----------------------------- treap ------------------------------------- #

def _build(n, seed=1):
    rng = random.Random(seed)
    root = None
    for k in rng.sample(range(10 * n), n):
        root = treap_insert(root, k, rng.random())
    return root


def test_reference_insert_produces_valid_treap():
    root = _build(300)
    a = verify_treap(root)
    assert a.ok, a.violations[:3]
    assert a.size == 300 == root.size
    assert a.height < 60          # expected O(log n); 300 nodes ~ 2*ln(300)*~2


def test_detects_bst_violation_deep_in_subtree():
    # 50 -> L 30 -> R 60 : 60 > 30 locally but violates ancestor bound < 50
    root = TreapNode(50, 9, left=TreapNode(30, 5, right=TreapNode(60, 1)))
    a = verify_treap(root)
    assert not a.ok
    assert any(v.startswith("root.L.R: key 60") for v in a.violations)


def test_detects_heap_violation_and_min_heap_mode():
    root = TreapNode(5, 1, left=TreapNode(2, 7), right=TreapNode(8, 3))
    max_a = verify_treap(root, heap="max")
    assert len([v for v in max_a.violations if "max-heap" in v]) == 2
    assert verify_treap(root, heap="min").ok
    with pytest.raises(ValueError):
        verify_treap(root, heap="mid")


def test_detects_stale_size_field_and_duplicates():
    root = _build(20)
    root.size += 1
    a = verify_treap(root)
    assert a.violations == [f"root: size field {root.size} != actual 20"]
    dup = TreapNode(5, 9, right=TreapNode(5, 1))
    assert any("outside BST range" in v for v in verify_treap(dup).violations)


def test_degenerate_chain_and_cycle_no_recursion_error():
    root = None
    for k in range(5000):                       # increasing keys and priorities
        root = TreapNode(k, k, left=root)
    # chain built as left children with increasing keys upward -> valid BST
    a = verify_treap(root)
    assert a.ok and a.height == 5000 and a.size == 5000
    loop = TreapNode(1, 5)
    loop.right = TreapNode(2, 4, left=None)
    loop.right.right = loop
    assert any("cycle" in v for v in verify_treap(loop).violations)
    assert verify_treap(None).ok
