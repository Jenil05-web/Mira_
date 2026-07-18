#!/usr/bin/env python3
"""
eval/run_eval.py
================
MIRA regression test runner.

Usage:
    python eval/run_eval.py                   # run all cases
    python eval/run_eval.py TC-006 TC-007     # run specific case IDs
    python eval/run_eval.py --fail-fast        # stop on first failure

Exit code: 0 = all passed, 1 = one or more failures.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import yaml

# ── Make sure project root is on sys.path ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.engine import get_engine, MIRAEngineProd  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {YELLOW}→{RESET} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Assertion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rows(state: dict) -> list:
    try:
        return json.loads(state.get("sql_result", "{}")).get("rows", [])
    except Exception:
        return []


def _sql_error(state: dict) -> str:
    try:
        return json.loads(state.get("sql_result", "{}")).get("error", "")
    except Exception:
        return ""


def run_assertions(asserts: dict, state: dict, case_id: str) -> list[str]:
    """
    Run each assertion against state.
    Returns a list of failure messages (empty = all pass).
    """
    failures = []

    def check(condition, msg):
        if not condition:
            failures.append(msg)

    for key, expected in asserts.items():

        if key == "data_status":
            actual = state.get("data_status", "")
            check(actual == expected,
                  f"data_status: expected '{expected}', got '{actual}'")

        elif key == "data_status_in":
            actual = state.get("data_status", "")
            check(actual in expected,
                  f"data_status: expected one of {expected}, got '{actual}'")

        elif key == "safety_flags_any":
            flags = state.get("safety_flags", [])
            check(expected in flags,
                  f"safety_flags_any: '{expected}' not found in {flags}")

        elif key == "safety_flags_empty":
            flags = state.get("safety_flags", [])
            if expected:  # assert it IS empty
                check(not flags,
                      f"safety_flags should be empty, got {flags}")

        elif key == "sql_error_contains":
            # Check both the direct sql_error state field (set on security rejections)
            # and the error key inside the sql_result JSON (set on DB-level errors).
            err_direct = state.get("sql_error", "")
            err_json   = _sql_error(state)
            combined   = err_direct + err_json
            check(expected.lower() in combined.lower(),
                  f"sql_error_contains: '{expected}' not in sql_error='{err_direct}' "
                  f"or sql_result.error='{err_json}'")

        elif key == "patient_count_gte":
            count = len(_rows(state))
            check(count >= expected,
                  f"patient_count_gte: expected >= {expected}, got {count}")

        elif key == "patient_count_exact":
            count = len(_rows(state))
            check(count == expected,
                  f"patient_count_exact: expected {expected}, got {count}")

        elif key == "approved":
            actual = state.get("approved")
            check(actual == expected,
                  f"approved: expected {expected}, got {actual}")

        elif key == "has_clinical_reasoning":
            cr = state.get("clinical_reasoning", "")
            if expected:
                check(bool(cr and cr.strip()),
                      "clinical_reasoning is empty or missing")
            else:
                check(not cr,
                      "clinical_reasoning expected empty but has content")

        else:
            failures.append(f"Unknown assertion key: '{key}'")

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Individual case runners
# ─────────────────────────────────────────────────────────────────────────────

def run_case(engine: MIRAEngineProd, case: dict) -> tuple[bool, list[str]]:
    """
    Run a single test case. Returns (passed: bool, detail_lines: list[str]).
    """
    question    = case["question"]
    hospital_id = case.get("hospital_id", "demo")
    asserts     = case.get("asserts", {})
    action      = case.get("action_after_pause")
    feedback    = case.get("feedback", "")

    cfg = engine.new_thread()

    # ── Step 1: run the query ────────────────────────────────────────────
    try:
        paused = engine.run_until_review(
            question, cfg,
            user_id="eval_runner",
            hospital_id=hospital_id,
            session_id=f"eval-{uuid.uuid4().hex[:8]}",
        )
    except Exception as exc:
        return False, [f"run_until_review raised exception: {exc}"]

    # ── Step 2: optional human decision ──────────────────────────────────
    if action == "approve":
        try:
            state = engine.submit_human_decision(
                cfg, "approve",
                user_id="eval_runner",
                hospital_id=hospital_id,
                clinical_question=question,
                paused_state=paused,
            )
        except Exception as exc:
            return False, [f"submit_human_decision(approve) raised: {exc}"]

    elif action == "reject":
        original_reasoning = paused.get("clinical_reasoning", "")
        try:
            state = engine.submit_human_decision(
                cfg, "reject",
                feedback=feedback,
                user_id="eval_runner",
                hospital_id=hospital_id,
                clinical_question=question,
                paused_state=paused,
            )
        except Exception as exc:
            return False, [f"submit_human_decision(reject) raised: {exc}"]

        # Special check for TC-010: revised report must differ from original
        revised_reasoning = state.get("clinical_reasoning", "")
        if "has_clinical_reasoning" in asserts and feedback:
            if revised_reasoning == original_reasoning:
                return False, [
                    "Revision path: final report is IDENTICAL to original — "
                    "feedback was not incorporated."
                ]

    else:
        state = paused

    # ── Step 3: run assertions ────────────────────────────────────────────
    failures = run_assertions(asserts, state, case["id"])
    return (len(failures) == 0), failures


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MIRA regression eval runner")
    parser.add_argument("case_ids", nargs="*", help="Case IDs to run (default: all)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on first failure")
    args = parser.parse_args()

    cases_path = Path(__file__).parent / "test_cases.yaml"
    if not cases_path.exists():
        print(f"{RED}Error:{RESET} test_cases.yaml not found at {cases_path}")
        sys.exit(1)

    with open(cases_path) as f:
        all_cases = yaml.safe_load(f)["cases"]

    # Filter by ID if requested
    if args.case_ids:
        cases = [c for c in all_cases if c["id"] in args.case_ids]
        if not cases:
            print(f"{RED}No matching cases found for IDs: {args.case_ids}{RESET}")
            sys.exit(1)
    else:
        cases = all_cases

    print(f"\n{BOLD}MIRA Regression Eval — {len(cases)} test(s){RESET}")
    print("─" * 60)

    print("Initialising engine (this may take a few seconds)...")
    try:
        engine = get_engine()
    except Exception as exc:
        print(f"{RED}Failed to initialise engine: {exc}{RESET}")
        sys.exit(1)

    print("Engine ready.\n")

    passed_ids, failed_ids = [], []
    total_time = 0.0

    for case in cases:
        cid   = case["id"]
        label = case["label"]
        print(f"{BOLD}[{cid}]{RESET} {label}")
        info(f"hospital={case.get('hospital_id','demo')}  "
             f"action={case.get('action_after_pause','—')}")

        t0 = time.monotonic()
        try:
            passed, details = run_case(engine, case)
        except Exception as exc:
            passed, details = False, [f"Unexpected exception: {exc}"]
        elapsed = time.monotonic() - t0
        total_time += elapsed

        if passed:
            ok(f"PASS  ({elapsed:.1f}s)")
            passed_ids.append(cid)
        else:
            for d in details:
                fail(d)
            print(f"  {RED}FAIL{RESET}  ({elapsed:.1f}s)")
            failed_ids.append(cid)
            if args.fail_fast:
                print(f"\n{YELLOW}--fail-fast: stopping.{RESET}")
                break

        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("─" * 60)
    total = len(passed_ids) + len(failed_ids)
    print(f"{BOLD}Results: {GREEN}{len(passed_ids)} passed{RESET}  "
          f"{RED}{len(failed_ids)} failed{RESET}  "
          f"/ {total} run  ({total_time:.1f}s total){RESET}")

    if failed_ids:
        print(f"\nFailed cases: {', '.join(failed_ids)}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}All tests passed.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
