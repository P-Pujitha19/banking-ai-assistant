"""TensorFlow/Keras model tests (skipped automatically when TensorFlow is not installed)."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from banking_ai.datagen.intent_data import generate_intent_dataset  # noqa: E402
from banking_ai.ml.model import build_intent_model  # noqa: E402
from banking_ai.ml.tokenizer import Vocabulary  # noqa: E402
from banking_ai.ml.train import MAX_LEN, train_intent_model  # noqa: E402


def test_model_output_is_a_probability_distribution():
    model = build_intent_model(vocab_size=100, num_classes=16, max_len=MAX_LEN)
    probs = model.predict(np.random.randint(0, 100, size=(4, MAX_LEN)), verbose=0)
    assert probs.shape == (4, 16)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_model_architecture_contains_embedding_and_recurrent_layers():
    model = build_intent_model(100, 16, MAX_LEN)
    names = [layer.name for layer in model.layers]
    assert "embedding" in names and "bigru" in names and names[-1] == "intent_probs"


def test_train_save_load_predict_roundtrip(tmp_path):
    import csv

    rows = generate_intent_dataset(seed=3, per_intent=25)
    dataset = tmp_path / "ds.csv"
    with dataset.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "intent"])
        writer.writerows(rows)
    metrics = train_intent_model(dataset, tmp_path / "model", epochs=8, batch_size=32, verbose=0)
    assert metrics["test_accuracy"] > 0.5  # tiny training run; just proves learning happens
    from banking_ai.ml.intent_classifier import TensorFlowIntentBackend, IntentClassifier

    backend = TensorFlowIntentBackend.load(tmp_path / "model")
    prediction = IntentClassifier(backend).predict("what is my account balance")
    assert 0.0 < prediction.confidence <= 1.0 and prediction.intent in backend.classes
