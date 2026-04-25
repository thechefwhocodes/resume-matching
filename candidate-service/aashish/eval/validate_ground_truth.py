"""Validate ground-truth files: every job_id must exist in `data/jobs.json`.

Usage:
    make eval-validate-gt
    # or
    python -m eval.validate_ground_truth
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GT_DIR = Path(__file__).resolve().parent / "ground_truth"
PROFILES_FILE = GT_DIR / "profiles_handlabeled.json"


def _load_jobs_index() -> set[str]:
    from src.config import get_settings  # noqa: PLC0415

    jobs_path = Path(get_settings().jobs_json_path)
    if not jobs_path.exists():
        raise FileNotFoundError(f"jobs.json not found at {jobs_path}")
    with jobs_path.open() as f:
        records = json.load(f)
    return {r["job_id"] for r in records}


def _validate_picks(jobs_idx: set[str]) -> tuple[int, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    pick_files = sorted(
        p for p in GT_DIR.glob("*.json") if p.name != "profiles_handlabeled.json"
    )
    if not pick_files:
        errors.append(f"No ground-truth pick files found in {GT_DIR}")
        return 0, errors, warnings

    for pf in pick_files:
        try:
            data = json.loads(pf.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"{pf.name}: invalid JSON ({e})")
            continue
        ids = data.get("expected_job_ids")
        if not isinstance(ids, list):
            errors.append(f"{pf.name}: expected_job_ids must be a list")
            continue
        if len(ids) == 0:
            warnings.append(
                f"{pf.name}: expected_job_ids is empty (placeholder). "
                "Eval suites that need it will be skipped for this resume."
            )
            continue
        if len(ids) != 5:
            warnings.append(
                f"{pf.name}: expected_job_ids has {len(ids)} entries (expected 5)."
            )
        for jid in ids:
            if jid not in jobs_idx:
                errors.append(f"{pf.name}: job_id {jid!r} not in jobs.json")
    return len(pick_files), errors, warnings


def _validate_profiles() -> tuple[int, list[str]]:
    if not PROFILES_FILE.exists():
        return 0, [f"missing {PROFILES_FILE}"]
    try:
        rows = json.loads(PROFILES_FILE.read_text())
    except json.JSONDecodeError as e:
        return 0, [f"profiles_handlabeled.json: invalid JSON ({e})"]
    if not isinstance(rows, list):
        return 0, ["profiles_handlabeled.json: top-level must be a list"]

    from src.extractor.prompts import ResumeProfile  # noqa: PLC0415

    errors: list[str] = []
    for i, row in enumerate(rows):
        if "profile" not in row:
            errors.append(f"profiles_handlabeled[{i}]: missing 'profile' field")
            continue
        try:
            ResumeProfile.model_validate(row["profile"])
        except Exception as e:
            errors.append(f"profiles_handlabeled[{i}]: invalid ResumeProfile ({e})")
    return len(rows), errors


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    jobs_idx = _load_jobs_index()
    picks_count, picks_errs, picks_warnings = _validate_picks(jobs_idx)
    profiles_count, profile_errs = _validate_profiles()

    print(f"Ground truth picks: {picks_count} files")
    print(f"Hand-labeled profiles: {profiles_count} entries")

    if picks_warnings:
        print("\nWarnings:")
        for w in picks_warnings:
            print(f"  - {w}")

    errs = picks_errs + profile_errs
    if errs:
        print("\nErrors:")
        for e in errs:
            print(f"  - {e}")
        return 1

    print("\nGround-truth validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
