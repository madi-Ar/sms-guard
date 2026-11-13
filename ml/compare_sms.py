"""Train-only model search; validation selection; one final test evaluation."""

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             fbeta_score, make_scorer, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.features import create_feature_pipeline
from ml.preprocessing import DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "ml" / "artifacts"
CONFIGS = [dict(include_chars=chars, stop_words=stop)
           for chars in (False, True) for stop in (None, "english")]


def folds():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def load_partitions():
    data = pd.read_csv(DATA_DIR / "sms_clean.csv", keep_default_na=False)
    split = json.loads((DATA_DIR / "sms_split.json").read_text(encoding="utf-8"))
    parts = [split[k] for k in ("train", "validation", "test")]
    if [len(p) for p in parts] != [3594, 770, 771]:
        raise ValueError("Unexpected partition sizes")
    if sorted(i for p in parts for i in p) != list(range(len(data))):
        raise ValueError("Partitions overlap or do not cover the corpus")
    if data.text_clean.duplicated().any():
        raise ValueError("Duplicate normalized texts")
    return tuple(data.iloc[p].copy() for p in parts)


def build_search(name):
    classifiers = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=3000,
                                                  random_state=42),
        "LinearSVC": LinearSVC(class_weight="balanced", dual=False, max_iter=20000,
                               random_state=42),
    }
    pipeline = Pipeline([("features", create_feature_pipeline()),
                         ("classifier", classifiers[name])])
    prefix = ""
    if name == "LinearSVC":
        pipeline = CalibratedClassifierCV(pipeline, cv=folds(), method="sigmoid")
        prefix = "estimator__"
    param = "alpha" if name == "MultinomialNB" else "C"
    values = [0.1, 0.5, 1.0] if param == "alpha" else [0.1, 1.0, 10.0]
    grid = [{prefix + "features": [create_feature_pipeline(**config)],
             prefix + "classifier__" + param: values} for config in CONFIGS]
    return GridSearchCV(pipeline, grid, scoring=make_scorer(fbeta_score, beta=2,
                        pos_label=1, zero_division=0), cv=folds(), n_jobs=1, refit=True)


def spam_probabilities(model, texts):
    return model.predict_proba(texts)[:, list(model.classes_).index(1)]


def choose_threshold(labels, probabilities, min_precision=0.85):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    valid = np.flatnonzero(precision[:-1] >= min_precision)
    if not len(valid):
        return None
    best_recall = recall[valid].max()
    index = valid[recall[valid] == best_recall][0]
    return float(thresholds[index])


def metrics(labels, probabilities, threshold):
    predicted = np.asarray(probabilities) >= threshold
    return dict(accuracy=float(accuracy_score(labels, predicted)),
                precision=float(precision_score(labels, predicted, zero_division=0)),
                recall=float(recall_score(labels, predicted, zero_division=0)),
                f1=float(fbeta_score(labels, predicted, beta=1, zero_division=0)),
                f2=float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
                roc_auc=float(roc_auc_score(labels, probabilities)),
                pr_auc=float(average_precision_score(labels, probabilities)),
                confusion_matrix=confusion_matrix(labels, predicted, labels=[0, 1]).tolist())


def plot_curves(labels, probabilities):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, probs in probabilities.items():
            if kind == "roc":
                x, y, _ = roc_curve(labels, probs)
            else:
                y, x, _ = precision_recall_curve(labels, probs)
            ax.plot(x, y, label=name)
        ax.set(xlabel="False positive rate" if kind == "roc" else "Recall",
               ylabel="True positive rate" if kind == "roc" else "Precision",
               title=f"Validation {kind.upper()}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ROOT / "docs" / f"08_{kind}.png", dpi=150)
        plt.close(fig)


def run_experiment():
    train, validation, test = load_partitions()
    rows, models, probabilities, experiments = [], {}, {}, []
    for name in ("MultinomialNB", "LogisticRegression", "LinearSVC"):
        print(f"Searching {name}", flush=True)
        search = build_search(name)
        started = perf_counter()
        search.fit(train.text_raw, train.label)
        search_seconds = perf_counter() - started
        prefix = "estimator__" if name == "LinearSVC" else ""
        for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
            feature = params[prefix + "features"]
            branches = dict(feature.named_steps["features"].transformer_list)
            experiments.append(dict(algorithm=name, include_chars="char" in branches,
                stop_words=branches["word"].named_steps["tfidf"].stop_words,
                parameter={k: v for k, v in params.items() if k != prefix + "features"},
                cv_f2=float(score)))
        best = experiments[-12:][int(search.best_index_)]
        model = search.best_estimator_
        probs = spam_probabilities(model, validation.text_raw)
        threshold = choose_threshold(validation.label, probs)
        started = perf_counter()
        for _ in range(30):
            spam_probabilities(model, [validation.text_raw.iloc[0]])
        latency = (perf_counter() - started) * 1000 / 30
        row = dict(algorithm=name, configuration=best, threshold=threshold,
                   training_seconds=float(search.refit_time_), search_seconds=search_seconds,
                   prediction_ms=latency, default_metrics=metrics(validation.label, probs, 0.5),
                   validation_metrics=None if threshold is None else metrics(validation.label, probs, threshold))
        rows.append(row)
        models[name], probabilities[name] = model, probs
        print(json.dumps(row, ensure_ascii=False), flush=True)
    eligible = [r for r in rows if r["threshold"] is not None]
    if not eligible:
        raise ValueError("No model meets validation precision >= 0.85")
    selected = max(eligible, key=lambda r: (r["validation_metrics"]["f2"],
                   r["validation_metrics"]["recall"], -r["prediction_ms"]))
    # Selection is complete before the sole test prediction call.
    model = models[selected["algorithm"]]
    threshold = selected["threshold"]
    test_metrics = metrics(test.label, spam_probabilities(model, test.text_raw), threshold)
    final = clone(model).fit(pd.concat([train, validation]).text_raw,
                             pd.concat([train, validation]).label)
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / "sms_model.joblib"
    joblib.dump(final, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(spam_probabilities(final, validation.text_raw),
                                  spam_probabilities(restored, validation.text_raw))
    custom = pd.read_csv(ROOT / "data" / "raw" / "custom_ru_kk.csv")
    languages = {}
    for language, count in (("ru", 100), ("kk", 50)):
        subset = custom[custom.language == language]
        if len(subset) != count:
            raise ValueError(f"Unexpected {language} sample size")
        languages[language] = metrics(subset.label, spam_probabilities(restored, subset.text), threshold)
    meta = dict(version="1.0.0", trained_at=datetime.now(timezone.utc).isoformat(),
                algorithm=selected["algorithm"], parameters=selected["configuration"]["parameter"],
                feature_configuration=selected["configuration"], threshold=threshold,
                test_metrics=test_metrics, test_evaluated_before_refit=True,
                fit_sizes=dict(train=3594, validation=770, test=771, final=4364),
                language_metrics=languages, comparison=rows, experiments=experiments,
                pr_auc_method="average_precision_score (non-interpolated AP)")
    (ARTIFACTS / "sms_model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_curves(validation.label, probabilities)
    return meta


if __name__ == "__main__":
    run_experiment()
