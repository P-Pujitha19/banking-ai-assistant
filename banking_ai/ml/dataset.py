"""Dataset loading and stratified train/validation/test splitting."""
from __future__ import annotations

import csv
from pathlib import Path

from sklearn.model_selection import train_test_split


def load_intent_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Read ``text,intent`` rows from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/generate_data.py")
    texts, labels = [], []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            texts.append(row["text"])
            labels.append(row["intent"])
    return texts, labels


def split_dataset(texts: list[str], labels: list[str], val_size: float = 0.15, test_size: float = 0.15, seed: int = 42):
    """Stratified 70/15/15 split. Returns ``(train, val, test)`` as ``(texts, labels)`` tuples."""
    x_train, x_rest, y_train, y_rest = train_test_split(
        texts, labels, test_size=val_size + test_size, stratify=labels, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=test_size / (val_size + test_size), stratify=y_rest, random_state=seed)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
