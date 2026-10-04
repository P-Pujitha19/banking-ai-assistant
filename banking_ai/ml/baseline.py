"""scikit-learn TF-IDF + logistic-regression intent baseline.

Two roles: (1) a *baseline* to compare the TensorFlow model against, and (2) an automatic fallback so the
assistant still works before the TensorFlow model has been trained (a warning is logged when this happens).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from banking_ai.ml.dataset import load_intent_dataset
from banking_ai.nlp.preprocessing import prepare_for_intent


class SklearnIntentBackend:
    """Probabilistic intent classifier with the same ``predict_proba`` contract as the TF backend."""

    name = "sklearn-tfidf-logreg"

    def __init__(self, pipeline: Pipeline, classes: list[str]) -> None:
        self.pipeline, self.classes = pipeline, classes

    @classmethod
    def fit(cls, texts: list[str], labels: list[str]) -> "SklearnIntentBackend":
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=prepare_for_intent, ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"<num>|<id>|<txn>|<date>|[a-z]+|%")),
            ("clf", LogisticRegression(C=30.0, max_iter=2000)),
        ])
        pipeline.fit(texts, labels)
        return cls(pipeline, list(pipeline.classes_))

    @classmethod
    def train_from_csv(cls, path: Path) -> "SklearnIntentBackend":
        texts, labels = load_intent_dataset(path)
        return cls.fit(texts, labels)

    def predict_proba(self, text: str) -> np.ndarray:
        return self.pipeline.predict_proba([text])[0]
