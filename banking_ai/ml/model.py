"""TensorFlow/Keras intent-classification model.

Architecture:  token ids -> Embedding(mask) -> Bidirectional GRU -> Dropout -> Dense(ReLU) -> Dropout -> Dense(softmax)

* **Embedding** turns each word id into a learned dense vector (words with similar roles end up close).
* **Bidirectional GRU** reads the short query left-to-right and right-to-left, so word order and context
  ("compare X *and* Y" vs "calculate EMI *for* Y") are captured; GRU has fewer parameters than LSTM, which suits
  a small dataset and keeps inference fast.
* **Softmax** gives a probability per intent; the max probability is used as the *confidence score*.
"""
from __future__ import annotations

from typing import Any


def build_intent_model(vocab_size: int, num_classes: int, max_len: int, embedding_dim: int = 64,
                       rnn_units: int = 48, dropout: float = 0.3) -> Any:
    """Build and compile the Keras model (TensorFlow is imported lazily)."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(max_len,), dtype="int32", name="token_ids")
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="embedding")(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_units), name="bigru")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_1")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = tf.keras.layers.Dropout(dropout * 0.6, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="intent_probs")(x)
    model = tf.keras.Model(inputs, outputs, name="intent_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
