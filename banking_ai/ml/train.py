"""Train, evaluate and save the TensorFlow intent classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from banking_ai.config.settings import get_settings
from banking_ai.ml.dataset import load_intent_dataset, split_dataset
from banking_ai.ml.model import build_intent_model
from banking_ai.ml.tokenizer import Vocabulary

MAX_LEN = 24
MODEL_FILE, VOCAB_FILE, LABELS_FILE, CONFIG_FILE, METRICS_FILE = (
    "intent_model.keras", "vocab.json", "labels.json", "config.json", "metrics.json")


def train_intent_model(dataset_path: Path, output_dir: Path, epochs: int = 60, batch_size: int = 32,
                       seed: int = 42, verbose: int = 2) -> dict[str, Any]:
    """Full training pipeline: split -> vocab -> encode -> fit -> evaluate -> save. Returns the metrics dict."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    texts, labels = load_intent_dataset(dataset_path)
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = split_dataset(texts, labels, seed=seed)

    classes = sorted(set(labels))
    label_to_id = {name: i for i, name in enumerate(classes)}
    vocab = Vocabulary.build(x_tr, max_size=4000)  # built on TRAIN only -> no leakage into val/test

    def encode(xs: list[str], ys: list[str]):
        return vocab.encode_batch(xs, MAX_LEN), np.asarray([label_to_id[y] for y in ys], dtype=np.int32)

    xtr, ytr = encode(x_tr, y_tr)
    xva, yva = encode(x_va, y_va)
    xte, yte = encode(x_te, y_te)

    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=ytr)
    model = build_intent_model(len(vocab), len(classes), MAX_LEN)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    history = model.fit(xtr, ytr, validation_data=(xva, yva), epochs=epochs, batch_size=batch_size,
                        class_weight={i: float(w) for i, w in enumerate(weights)}, callbacks=callbacks, verbose=verbose)

    probs = model.predict(xte, verbose=0)
    pred = probs.argmax(axis=1)
    metrics: dict[str, Any] = {
        "backend": "tensorflow", "tensorflow_version": tf.__version__,
        "train_size": len(x_tr), "val_size": len(x_va), "test_size": len(x_te),
        "epochs_run": len(history.history["loss"]),
        "val_accuracy_best": float(max(history.history["val_accuracy"])),
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_macro_f1": float(f1_score(yte, pred, average="macro")),
        "mean_confidence_correct": float(probs.max(axis=1)[pred == yte].mean()) if (pred == yte).any() else 0.0,
        "mean_confidence_wrong": float(probs.max(axis=1)[pred != yte].mean()) if (pred != yte).any() else None,
        "classes": classes,
        "per_class": classification_report(yte, pred, labels=list(range(len(classes))), target_names=classes,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(yte, pred, labels=list(range(len(classes)))).tolist(),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "note": "Metrics are on a held-out split of SYNTHETIC template data and are optimistic; "
                "use scripts/evaluate.py (hand-written queries) for a realistic estimate.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / MODEL_FILE))
    vocab.save(output_dir / VOCAB_FILE)
    (output_dir / LABELS_FILE).write_text(json.dumps(classes), encoding="utf-8")
    (output_dir / CONFIG_FILE).write_text(json.dumps({"max_len": MAX_LEN}), encoding="utf-8")
    (output_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train the TensorFlow intent classifier")
    parser.add_argument("--dataset", type=Path, default=settings.intent_dataset_path)
    parser.add_argument("--output-dir", type=Path, default=settings.intent_model_dir)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train_intent_model(args.dataset, args.output_dir, args.epochs, args.batch_size, args.seed)
    print(f"\nSaved model to {args.output_dir}")
    print(f"Held-out test accuracy: {metrics['test_accuracy']:.4f} | macro-F1: {metrics['test_macro_f1']:.4f} "
          f"| epochs run: {metrics['epochs_run']}")
    print(metrics["note"])


if __name__ == "__main__":
    main()
