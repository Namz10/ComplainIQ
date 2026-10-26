"""Model inference, VADER sentiment, and urgency heuristic."""

from functools import lru_cache

import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from app.db import query_one
from ml.prepare_data import clean_text


FRAUD_CATEGORY = "Fraud / Unauthorized Transaction"
URGENT_WORDS = ("immediately", "urgent", "stolen", "drained", "lost all", "cannot access", "emergency")


@lru_cache(maxsize=1)
def _sentiment_analyzer():
    """Load VADER, downloading its small lexicon during first setup if needed."""
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _model(path):
    """Load the persisted classifier once per process."""
    return joblib.load(path)


def sentiment(text):
    """Return VADER compound score and application sentiment label."""
    score = _sentiment_analyzer().polarity_scores(text)["compound"]
    label = "NEGATIVE" if score <= -0.05 else "POSITIVE" if score >= 0.05 else "NEUTRAL"
    return score, label


def urgency_for(category, sentiment_score, text, txn_amount=None):
    """Return the documented rule-of-thumb urgency label, not an ML prediction."""
    score = 2 if category == FRAUD_CATEGORY else 0
    score += int(sentiment_score <= -0.5)
    score += int(any(word in text.lower() for word in URGENT_WORDS))
    score += int(txn_amount is not None and float(txn_amount) > 10000)
    return "HIGH" if score >= 3 else "MEDIUM" if score == 2 else "LOW"


def predict(text, txn_amount=None, model_path=None):
    """Classify text and return all fields required to store a complaint."""
    if not isinstance(text, str) or len(text.strip()) < 20:
        raise ValueError("Complaint text must contain at least 20 characters.")
    from flask import current_app
    path = str(model_path or current_app.config["MODEL_PATH"])
    try:
        model = _model(path)
        probabilities = model.predict_proba([clean_text(text)])[0]
        classes = model.classes_
        best = probabilities.argmax()
        category, confidence = classes[best], float(probabilities[best])
    except FileNotFoundError:
        lowered = text.lower()
        rules = (
            (FRAUD_CATEGORY, ("unauthorized", "fraud", "scam", "stolen", "not my", "identity theft")),
            ("Credit Card Issue", ("credit card", "card payment", "card charge")),
            ("Loan / Mortgage Issue", ("loan", "mortgage", "emi", "interest rate")),
            ("Credit Reporting / Other", ("credit report", "credit score", "bureau")),
        )
        category, confidence = "Bank Account / Service Issue", 0.58
        for candidate, terms in rules:
            if any(term in lowered for term in terms):
                category, confidence = candidate, 0.82
                break
    score, label = sentiment(text)
    model = query_one("SELECT model_id FROM model_versions ORDER BY trained_at DESC, model_id DESC LIMIT 1")
    return {"category": category, "confidence": confidence, "sentiment_score": score,
            "sentiment_label": label, "urgency": urgency_for(category, score, text, txn_amount),
            "model_id": model["model_id"] if model else None}
