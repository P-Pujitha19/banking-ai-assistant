"""Inference wrapper around the trained intent model (TensorFlow, with sklearn fallback)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from banking_ai.config.logging_config import get_logger
from banking_ai.config.settings import Settings
from banking_ai.ml.baseline import SklearnIntentBackend
from banking_ai.ml.tokenizer import Vocabulary

logger = get_logger("ml.intent")


class ProbBackend(Protocol):
    name: str
    classes: list[str]

    def predict_proba(self, text: str) -> np.ndarray: ...


@dataclass
class IntentPrediction:
    """Predicted intent with confidence and the runner-ups (for debugging)."""

    intent: str
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)
    backend: str = ""


class TensorFlowIntentBackend:
    """Loads the saved Keras model + vocabulary + label map produced by ``banking_ai.ml.train``."""

    name = "tensorflow-bigru"

    def __init__(self, model: Any, vocab: Vocabulary, classes: list[str], max_len: int) -> None:
        self.model, self.vocab, self.classes, self.max_len = model, vocab, classes, max_len

    @classmethod
    def load(cls, model_dir: Path) -> "TensorFlowIntentBackend":
        model_path = model_dir / "intent_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model at {model_path}. Run: python scripts/train_intent_model.py")
        import tensorflow as tf  # lazy: only needed when a trained model exists

        model = tf.keras.models.load_model(str(model_path))
        vocab = Vocabulary.load(model_dir / "vocab.json")
        classes = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        max_len = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["max_len"]
        return cls(model, vocab, classes, max_len)

    def predict_proba(self, text: str) -> np.ndarray:
        x = self.vocab.encode(text, self.max_len)[None, :]
        return np.asarray(self.model(x, training=False))[0]


class IntentClassifier:
    """Predicts ``(intent, confidence)`` for a user query."""

    def __init__(self, backend: ProbBackend) -> None:
        self.backend = backend

    @classmethod
    def load(cls, settings: Settings) -> "IntentClassifier":
        """Load the TensorFlow model; in ``auto`` mode fall back to the sklearn baseline with a warning."""
        mode = settings.intent_backend
        if mode in {"auto", "tensorflow"}:
            try:
                backend = TensorFlowIntentBackend.load(settings.intent_model_dir)
                logger.info("intent backend: tensorflow")
                return cls(backend)
            except Exception as exc:  # missing model, TF not installed, version mismatch...
                if mode == "tensorflow":
                    raise
                logger.warning("TensorFlow intent model unavailable (%s); falling back to sklearn baseline. "
                               "Run scripts/train_intent_model.py to enable the TensorFlow classifier.", exc)
        return cls(SklearnIntentBackend.train_from_csv(settings.intent_dataset_path))

    def predict(self, text: str, top_k: int = 3) -> IntentPrediction:
        probs = self.backend.predict_proba(text)
        order = np.argsort(probs)[::-1]
        ranked = [(self.backend.classes[i], float(probs[i])) for i in order[:top_k]]
        return IntentPrediction(intent=ranked[0][0], confidence=ranked[0][1], top_k=ranked, backend=self.backend.name)
