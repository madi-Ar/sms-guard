import json

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from ml.compare_sms import (ARTIFACTS, CONFIGS, build_search, choose_threshold,
                            load_partitions, metrics, spam_probabilities)
from ml.features import create_feature_pipeline


def test_saved_partitions_are_disjoint_and_unchanged():
    train, validation, test = load_partitions()
    assert [len(p) for p in (train, validation, test)] == [3594, 770, 771]
    assert [int(p.label.sum()) for p in (train, validation, test)] == [433, 92, 93]
    assert not (set(train.id) & set(validation.id) or set(train.id) & set(test.id)
                or set(validation.id) & set(test.id))


@pytest.mark.parametrize("name", ["MultinomialNB", "LogisticRegression", "LinearSVC"])
def test_search_is_reproducible_and_calibration_contains_features(name):
    search = build_search(name)
    assert (search.cv.n_splits, search.cv.shuffle, search.cv.random_state) == (5, True, 42)
    assert search.scoring._kwargs["beta"] == 2
    assert len(search.param_grid) == 4
    estimator = search.estimator
    if name == "LinearSVC":
        assert isinstance(estimator, CalibratedClassifierCV)
        assert estimator.cv.random_state == 42
        estimator = estimator.estimator
    assert isinstance(estimator, Pipeline)
    assert isinstance(estimator.named_steps["features"], Pipeline)
    assert clone(search).cv.random_state == 42


def test_four_variants_keep_manual_features_and_do_not_leak():
    texts = ["hello world today", "hello again today", "world meeting tomorrow",
             "meeting again tomorrow"]
    manual = []
    for config in CONFIGS:
        pipeline = create_feature_pipeline(**config).fit(texts)
        branches = dict(pipeline.named_steps["features"].transformer_list)
        vectorizer = branches["word"].named_steps["tfidf"]
        vocab, idf = vectorizer.vocabulary_.copy(), vectorizer.idf_.copy()
        transformed = pipeline.transform(["validationuniqueword! £20"])
        assert vectorizer.vocabulary_ == vocab
        np.testing.assert_array_equal(idf, vectorizer.idf_)
        assert "validationuniqueword" not in vocab
        manual.append(transformed.toarray()[0, -6:])
    for values in manual[1:]:
        np.testing.assert_array_equal(values, manual[0])


def test_threshold_uses_real_scores_and_excludes_empty_predictions():
    y, p = [0, 1, 0, 1], [0.1, 0.2, 0.3, 0.9]
    assert choose_threshold(y, p) == 0.9
    assert choose_threshold([1, 0], [0.1, 0.9]) is None
    assert choose_threshold([0, 1, 1], [0.1, 0.4, 0.8]) == 0.4
    assert choose_threshold([0, 1], [0.5, 0.5]) is None


def test_inner_calibration_vectorizers_only_see_their_training_fold():
    texts = np.array(["hello world meeting tomorrow " + token for token in
                      ["apple", "orange", "pear", "peach", "plum"] * 2] +
                     ["urgent win cash prize " + token for token in
                      ["alpha", "bravo", "delta", "echo", "foxtrot"] * 2])
    labels = np.array([0] * 10 + [1] * 10)
    calibrated = build_search("LinearSVC").estimator.fit(texts, labels)
    vocabularies = []
    for (train, heldout), fitted in zip(calibrated.cv.split(texts, labels),
                                       calibrated.calibrated_classifiers_):
        actual = dict(fitted.estimator.named_steps["features"].named_steps["features"].transformer_list)
        expected = dict(create_feature_pipeline().fit(texts[train]).named_steps["features"].transformer_list)
        for name in ("word", "char"):
            a, e = actual[name].named_steps["tfidf"], expected[name].named_steps["tfidf"]
            assert a.vocabulary_ == e.vocabulary_
            np.testing.assert_array_equal(a.idf_, e.idf_)
        vocabularies.append(actual["char"].named_steps["tfidf"].vocabulary_)
    assert len(vocabularies) == 5


def test_threshold_matches_exhaustive_search():
    rng = np.random.default_rng(42)
    for _ in range(30):
        labels = np.array([0] * 15 + [1] * 5)
        probabilities = np.round(rng.random(20), 1)
        admissible = []
        for threshold in np.unique(probabilities):
            predicted = probabilities >= threshold
            tp = int(np.sum(predicted & (labels == 1)))
            if tp / int(predicted.sum()) >= .85:
                admissible.append((tp / 5, float(threshold)))
        expected = min(admissible, key=lambda r: (-r[0], r[1]))[1] if admissible else None
        assert choose_threshold(labels, probabilities) == expected


def test_final_artifact_and_validation_threshold():
    meta = json.loads((ARTIFACTS / "sms_model_meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == "1.0.0"
    assert meta["test_evaluated_before_refit"]
    assert meta["fit_sizes"]["final"] == 4364
    assert len(meta["experiments"]) == 36
    selected = next(r for r in meta["comparison"] if r["algorithm"] == meta["algorithm"])
    assert selected["threshold"] == meta["threshold"]
    assert selected["validation_metrics"]["precision"] >= 0.85
    model = joblib.load(ARTIFACTS / "sms_model.joblib")
    probabilities = spam_probabilities(model, ["Bring the notebook tomorrow", "WIN a cash prize now!"])
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert metrics([0, 1], probabilities, meta["threshold"])["confusion_matrix"]
