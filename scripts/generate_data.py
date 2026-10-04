#!/usr/bin/env python
"""Generate ALL synthetic data: bank JSON, intent dataset, evaluation sets.

    python scripts/generate_data.py [--anchor YYYY-MM-DD] [--seed 42]
"""
import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from banking_ai.datagen import bank_data, eval_data, intent_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic banking data")
    parser.add_argument("--anchor", default=date.today().isoformat(), help="the bank's 'today' (default: today's date)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    anchor = date.fromisoformat(args.anchor)
    print("bank data:", bank_data.write_all(ROOT / "data" / "bank", anchor, args.seed))
    print("intent rows:", intent_data.write_intent_dataset(ROOT / "data" / "ml" / "intent_dataset.csv", seed=7))
    print("evaluation sets:", eval_data.write_eval_files(ROOT / "data" / "evaluation"))


if __name__ == "__main__":
    main()
