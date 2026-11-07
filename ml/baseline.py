"""Базовая Logistic Regression для англоязычных SMS."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from ml.features import create_feature_pipeline
from ml.preprocessing import DATA_DIR

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "sms_baseline.joblib"


def load_partitions():
    messages = pd.read_csv(DATA_DIR / "sms_clean.csv", encoding="utf-8",
                           keep_default_na=False, dtype={"text_raw": str})
    splits = json.loads((DATA_DIR / "sms_split.json").read_text(encoding="utf-8"))
    parts = [splits[name] for name in ("train", "validation", "test")]
    positions = [position for part in parts for position in part]
    if sorted(positions) != list(range(len(messages))):
        raise ValueError("Индексы должны покрывать корпус без повторов и пересечений")
    return messages.iloc[splits["train"]].copy(), messages.iloc[splits["validation"]].copy()


def train_baseline(train: pd.DataFrame) -> Pipeline:
    model = Pipeline([
        ("features", create_feature_pipeline()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        )),
    ])
    model.fit(train["text_raw"], train["label"])
    return model


def evaluate_baseline(model: Pipeline, validation: pd.DataFrame):
    predicted = model.predict(validation["text_raw"])
    probabilities = model.predict_proba(validation["text_raw"])
    spam_column = list(model.classes_).index(1)
    results = validation[["id", "text_raw", "label"]].copy()
    results["predicted"] = predicted
    results["probability"] = probabilities[:, spam_column]
    precision, recall, f1, _ = precision_recall_fscore_support(
        validation["label"], predicted, average="binary", pos_label=1, zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(validation["label"], predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    matrix = confusion_matrix(validation["label"], predicted, labels=[0, 1])
    return metrics, matrix, results


def predict_sms(text: str) -> dict:
    """Загружает локальную модель; нормализация выполняется внутри её Pipeline."""
    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba([text])[0]
    return {
        "label": int(model.classes_[np.argmax(probabilities)]),
        "probability": float(probabilities[list(model.classes_).index(1)]),
    }


def main() -> None:
    train, validation = load_partitions()
    model = train_baseline(train)
    metrics, matrix, results = evaluate_baseline(model, validation)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    restored = joblib.load(MODEL_PATH)
    np.testing.assert_array_equal(model.predict(validation["text_raw"]),
                                  restored.predict(validation["text_raw"]))
    np.testing.assert_array_equal(model.predict_proba(validation["text_raw"]),
                                  restored.predict_proba(validation["text_raw"]))
    print(f"Train: {len(train)}, validation: {len(validation)}")
    print(metrics)
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)
    for title, true, predicted in (("FN", 1, 0), ("FP", 0, 1)):
        errors = results.loc[(results["label"] == true) & (results["predicted"] == predicted)]
        print(f"\n{title}: {len(errors)}, первые пять:")
        print(errors.head(5).to_string(index=False))
    print(f"Модель: {MODEL_PATH.name}, {MODEL_PATH.stat().st_size} байт")


if __name__ == "__main__":
    main()
