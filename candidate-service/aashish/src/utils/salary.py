"""Format numeric salary min/max into the human-readable string the UI expects.

The UI's JobCard renders `match.salary_range` directly (no formatting on its end),
so we standardise the display here. Examples:
    (80_000, 120_000, "$") -> "$80K - $120K"
    (180_000, 240_000, "$") -> "$180K - $240K"
    (None, None, "$") -> ""
"""

from __future__ import annotations


def _to_k(amount: int | float | None) -> str | None:
    if amount is None:
        return None
    return f"{int(round(amount / 1000))}K"


def format_salary_range(
    salary_min: int | float | None,
    salary_max: int | float | None,
    currency: str | None = "$",
) -> str:
    sym = currency or "$"
    lo = _to_k(salary_min)
    hi = _to_k(salary_max)
    if lo and hi:
        if lo == hi:
            return f"{sym}{lo}"
        return f"{sym}{lo} - {sym}{hi}"
    if lo:
        return f"{sym}{lo}+"
    if hi:
        return f"Up to {sym}{hi}"
    return ""
