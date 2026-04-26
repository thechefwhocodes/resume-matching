"""Regression tests for eval/reranker_eval.aggregate_rater_scores.

Locks the alignment invariant that previously broke cohens_kappa silently:
when EXTRACT_MODEL != RERANK_MODEL, the primary and judge score lists must
have the same length so cohens_kappa(...) does not hit its
`len(rater_a) != len(rater_b)` short-circuit.
"""

from __future__ import annotations

import pytest

from eval.reranker_eval import aggregate_rater_scores


def _scores(*pairs: tuple[str, float]) -> dict[str, dict]:
    return {jid: {"fit_score": s} for jid, s in pairs}


def test_single_model_returns_all_primary_no_judge():
    """No judge supplied -> primary only, judge list empty."""
    primary = _scores(("a", 80), ("b", 60), ("c", 40))
    p, j = aggregate_rater_scores(
        ids=["a", "b", "c"],
        primary=primary,
        judge=None,
    )
    assert p == [80, 60, 40]
    assert j == []


def test_cross_model_lists_are_aligned():
    """When judge is supplied, len(primary) == len(judge) per id intersection."""
    primary = _scores(("a", 90), ("b", 70), ("c", 50))
    judge = _scores(("a", 88), ("b", 65), ("c", 45))
    p, j = aggregate_rater_scores(
        ids=["a", "b", "c"],
        primary=primary,
        judge=judge,
    )
    assert len(p) == len(j) == 3


def test_cross_model_skips_ids_missing_from_either_rater():
    """Only ids scored by BOTH raters appear in the aligned outputs."""
    primary = _scores(("a", 90), ("b", 70), ("c", 50))
    judge = _scores(("a", 88), ("c", 42))
    p, j = aggregate_rater_scores(
        ids=["a", "b", "c"],
        primary=primary,
        judge=judge,
    )
    assert p == [90, 50]
    assert j == [88, 42]
    assert len(p) == len(j)


def test_cross_model_no_double_count_regression():
    """Regression: previously the unconditional extend ran BEFORE the
    cross-model loop, so primary scores were counted twice. Lock that the
    primary list has at most one entry per id intersection."""
    primary = _scores(("a", 90), ("b", 70), ("c", 50))
    judge = _scores(("a", 88), ("b", 65), ("c", 45))
    p, _j = aggregate_rater_scores(
        ids=["a", "b", "c"],
        primary=primary,
        judge=judge,
    )
    assert len(p) == 3
    assert sorted(p) == [50, 70, 90]


@pytest.mark.parametrize(
    "ids,primary_pairs,judge_pairs,expected_len",
    [
        (["x"], [("x", 1.0)], [("x", 2.0)], 1),
        (["x", "y"], [("x", 1.0)], [("y", 2.0)], 0),
        ([], [("x", 1.0)], [("x", 2.0)], 0),
    ],
)
def test_cross_model_edge_cases(ids, primary_pairs, judge_pairs, expected_len):
    p, j = aggregate_rater_scores(
        ids=ids,
        primary=_scores(*primary_pairs),
        judge=_scores(*judge_pairs),
    )
    assert len(p) == len(j) == expected_len
