#!/usr/bin/env python3
"""
Standalone CLI to manually trigger a full benchmark sync for all Neural Gateway models.

Usage:
    python scripts/run_benchmark_sync.py

This script:
  1. Fetches live scores from Artificial Analysis, LiveBench, HuggingFace.
  2. Normalizes all scores population-wide (no hardcoded ceilings).
  3. Matches scores to every model in the Neural Gateway SQLite registry.
  4. Unlocks all matched models (eligible_for_auto_route = True).
  5. Prints a final summary report.

No LLM is used at any stage. All scoring is purely mathematical.
"""

import sys
import os
import logging

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.core.database import init_db, get_all_models
from app.core.benchmark_sync import run_benchmark_sync


def main() -> None:
    print("\n" + "=" * 60)
    print("  Neural Gateway Automated Benchmark Sync")
    print("  Zero hardcoding. Zero LLM judges. Pure leaderboard data.")
    print("=" * 60 + "\n")

    init_db()

    before = get_all_models()
    locked_before = sum(
        1 for m in before
        if not m.get("evidence", {}).get("eligible_for_auto_route", False)
    )
    unlocked_before = len(before) - locked_before

    print(f"Registry state BEFORE sync:")
    print(f"  Total models : {len(before)}")
    print(f"  Unlocked     : {unlocked_before}")
    print(f"  Locked       : {locked_before}\n")

    summary = run_benchmark_sync()

    after = get_all_models()
    locked_after = sum(
        1 for m in after
        if not m.get("evidence", {}).get("eligible_for_auto_route", False)
    )
    unlocked_after = len(after) - locked_after

    print("\n" + "=" * 60)
    print("  SYNC COMPLETE")
    print("=" * 60)
    print(f"  Status         : {summary.get('status', 'unknown')}")
    print(f"  Sources active : {summary.get('sources_active', 0)}/3")
    print(f"  Total models   : {summary.get('total_models', 0)}")
    print(f"  Newly unlocked : {summary.get('unlocked', 0)}")
    print(f"  No data found  : {summary.get('skipped_no_data', 0)}")
    print(f"\n  Registry state AFTER sync:")
    print(f"  Unlocked : {unlocked_after}")
    print(f"  Locked   : {locked_after}")
    net_gain = unlocked_after - unlocked_before
    print(f"  Net gain : +{net_gain} models now routing-eligible")
    print("=" * 60 + "\n")

    if summary.get("status") == "no_sources":
        print("WARNING: All 3 leaderboard sources were unreachable.")
        print("Check your internet connection and try again.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
