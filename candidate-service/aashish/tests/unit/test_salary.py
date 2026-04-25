from __future__ import annotations

from src.utils.salary import format_salary_range


def test_min_and_max():
    assert format_salary_range(80_000, 120_000, "$") == "$80K - $120K"


def test_only_min():
    assert format_salary_range(150_000, None, "$") == "$150K+"


def test_only_max():
    assert format_salary_range(None, 200_000, "$") == "Up to $200K"


def test_neither():
    assert format_salary_range(None, None, "$") == ""


def test_equal_min_max():
    assert format_salary_range(100_000, 100_000, "$") == "$100K"


def test_default_currency_when_none():
    assert format_salary_range(80_000, 120_000, None) == "$80K - $120K"
