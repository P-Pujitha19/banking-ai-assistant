#!/usr/bin/env python
"""Train the TensorFlow intent classifier: ``python scripts/train_intent_model.py``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from banking_ai.ml.train import main  # noqa: E402

if __name__ == "__main__":
    main()
