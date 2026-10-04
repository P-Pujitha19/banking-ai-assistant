import numpy as np

from banking_ai.datagen.intent_data import INTENTS, generate_intent_dataset
from banking_ai.ml.dataset import split_dataset
from banking_ai.ml.tokenizer import Vocabulary
from banking_ai.nlp.preprocessing import prepare_for_intent, tokenize


def test_numbers_become_placeholders():
    assert prepare_for_intent("Loan of 5 lakh for 5 years") == "loan of <num> lakh for <num> years"
    assert "<id>" in prepare_for_intent("balance of CUST1001")


def test_tokenize_keeps_placeholders():
    assert tokenize("EMI for ₹5,00,000 at 10%") == ["emi", "for", "rs", "<num>", "at", "<num>", "%"]


def test_vocabulary_encode_pads_truncates_and_maps_unknown():
    vocab = Vocabulary.build(["show my balance", "show transactions"])
    ids = vocab.encode("show my zzzunknown", max_len=6)
    assert ids.shape == (6,) and ids.dtype == np.int32
    assert ids[0] != 0 and ids[2] == 1 and ids[3:].sum() == 0  # unknown -> 1, padding -> 0
    assert len(vocab.encode("a b c d e f g h", max_len=3)) == 3


def test_vocabulary_roundtrip(tmp_path):
    vocab = Vocabulary.build(["hello world"])
    vocab.save(tmp_path / "v.json")
    assert Vocabulary.load(tmp_path / "v.json").token_to_id == vocab.token_to_id


def test_synthetic_dataset_covers_all_intents_and_splits_cleanly():
    rows = generate_intent_dataset(seed=1, per_intent=40)
    assert {label for _, label in rows} == set(INTENTS)
    texts, labels = [r[0] for r in rows], [r[1] for r in rows]
    (xt, yt), (xv, yv), (xs, ys) = split_dataset(texts, labels)
    assert len(xt) > len(xv) and len(xt) > len(xs) and len(xt) + len(xv) + len(xs) == len(texts)
