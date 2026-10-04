import pytest

from banking_ai.ml.intent_classifier import IntentClassifier


@pytest.fixture(scope="module")
def classifier(settings):
    return IntentClassifier.load(settings)  # TensorFlow model if trained, otherwise the sklearn baseline


@pytest.mark.parametrize("text,intent", [
    ("What is my account balance?", "balance_inquiry"),
    ("Show my last 5 transactions", "transaction_history"),
    ("What was my last transaction?", "recent_transaction"),
    ("Can I get a personal loan of 5 lakh for 5 years?", "loan_eligibility"),
    ("What is the interest rate for a personal loan?", "interest_rate"),
    ("Calculate EMI for 5 lakh for 5 years at 10%", "emi_calculation"),
    ("Compare personal loan and home loan", "loan_comparison"),
    ("What is the minimum balance requirement?", "minimum_balance"),
    ("Why was money deducted from my account?", "transaction_explanation"),
    ("hello", "greeting"),
])
def test_intent_prediction(classifier, text, intent):
    prediction = classifier.predict(text)
    assert prediction.intent == intent
    assert 0.0 < prediction.confidence <= 1.0
    assert len(prediction.top_k) == 3 and prediction.top_k[0][0] == prediction.intent


def test_out_of_scope_goes_to_fallback(classifier):
    assert classifier.predict("tell me a joke about cats").intent == "fallback"
