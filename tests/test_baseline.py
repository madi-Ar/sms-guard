import joblib
import numpy as np
import pandas as pd
import pytest

from ml import baseline, features


@pytest.fixture
def fitted_model(tmp_path, monkeypatch):
    texts = [
        "Meeting tomorrow at the library", "Meeting today at the library",
        "Dinner tomorrow with my friend", "Dinner today with my friend",
        "WIN cash prize call now", "WIN cash prize call today",
        "Urgent claim your cash prize", "Urgent claim your free prize",
    ]
    train = pd.DataFrame({"text_raw": texts, "label": [0] * 4 + [1] * 4})
    model = baseline.train_baseline(train)
    path = tmp_path / "baseline.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(baseline, "MODEL_PATH", path)
    return model, train


def test_probabilities_and_serialization(fitted_model):
    model, train = fitted_model
    restored = joblib.load(baseline.MODEL_PATH)
    probabilities = model.predict_proba(train["text_raw"])
    assert probabilities.shape == (8, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(model.predict(train["text_raw"]),
                                  restored.predict(train["text_raw"]))
    np.testing.assert_array_equal(probabilities, restored.predict_proba(train["text_raw"]))


def test_predict_sms_normalizes_once(fitted_model, monkeypatch):
    model, _ = fitted_model
    text = "A new dinner invitation for next weekend!"
    expected = model.predict_proba([text])[0, 1]
    calls = []
    original = features.normalize_sms

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(features, "normalize_sms", counted)
    result = baseline.predict_sms(text)
    assert calls == [text]
    assert result == {"label": int(model.classes_[int(expected > 0.5)]),
                      "probability": expected}


def test_validation_does_not_change_vectorizers(fitted_model):
    model, _ = fitted_model
    branches = dict(model.named_steps["features"].named_steps["features"].transformer_list)
    vectorizers = [branches[name].named_steps["tfidf"] for name in ("word", "char")]
    saved = [(v.vocabulary_.copy(), v.idf_.copy()) for v in vectorizers]
    validation = pd.DataFrame({
        "id": [100, 101],
        "text_raw": ["Unseenvalidationunique", "Anotherunseenvalidationunique"],
        "label": [0, 1],
    })
    metrics, matrix, results = baseline.evaluate_baseline(model, validation)
    assert matrix.sum() == 2 and len(results) == 2
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    for vectorizer, (vocabulary, idf) in zip(vectorizers, saved):
        assert vectorizer.vocabulary_ == vocabulary
        np.testing.assert_array_equal(vectorizer.idf_, idf)
    assert "unseenvalidationunique" not in vectorizers[0].vocabulary_
