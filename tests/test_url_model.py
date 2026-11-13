import json
import socket
from urllib import request

import joblib
import numpy as np
import pandas as pd
import pytest
import requests
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

from ml import url_model as module
from ml.url_dataset import domain_group, normalize_url
from ml.url_features import extract_url_features


@pytest.fixture
def frame():
    rows = []
    for i in range(15):
        for label, suffix in ((0, ""), (0, "/about"), (1, "/login?password=1"), (1, "/verify//account?code=123")):
            url = f"{'https' if label == 0 else 'http'}://site{i}.test{suffix}"
            rows.append(dict(extract_url_features(url), label=label,
                             group=domain_group(url), url_original=url, source_id=len(rows)))
    return pd.DataFrame(rows)


def test_feature_sets_exclude_all_metadata_and_artifact_indicators(frame):
    for variant, count in (("A", 22), ("B", 19)):
        matrix = module.feature_matrix(frame, module.FEATURE_SETS[variant])
        assert matrix.shape == (60, count)
        assert matrix.columns.tolist() == module.FEATURE_SETS[variant]
        assert all(pd.api.types.is_numeric_dtype(t) for t in matrix.dtypes)
        assert not set(matrix) & {"label", "source_id", "url_original", "url", "group", "split"}
    assert not set(module.REMOVED_FEATURES) & set(module.FEATURE_SETS["B"])


@pytest.mark.parametrize("names", [["label"], ["group"], ["url_length", "url_length"]])
def test_feature_matrix_rejects_metadata_and_duplicate_names(frame, names):
    with pytest.raises(ValueError, match="признаки"):
        module.feature_matrix(frame, names)


@pytest.mark.parametrize("value", [np.nan, np.inf, "not numeric"])
def test_feature_matrix_rejects_invalid_numeric_values(frame, value):
    frame["url_length"] = value
    with pytest.raises(ValueError):
        module.feature_matrix(frame, module.FEATURE_SETS["A"])


def test_group_folds_are_reproducible_disjoint_and_cover_train(frame):
    a, b = module.checked_folds(frame), module.checked_folds(frame)
    heldout_positions = []
    for (train, heldout), (train2, heldout2) in zip(a, b):
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(heldout, heldout2)
        assert not set(frame.iloc[train].group) & set(frame.iloc[heldout].group)
        heldout_positions.extend(heldout)
    assert sorted(heldout_positions) == list(range(len(frame)))


def test_scaler_fits_only_the_training_part_of_each_fold(frame):
    frame.loc[frame.group == "site0.test", "url_length"] = 10000
    x = module.feature_matrix(frame, module.FEATURE_SETS["A"])
    model = module.build_search("LogisticRegression").estimator
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    folds = module.checked_folds(frame)
    results = cross_validate(model, x, frame.label, groups=frame.group,
                             cv=module.group_cv(), return_estimator=True)
    for (train, heldout), fitted in zip(folds, results["estimator"]):
        scaler = fitted.named_steps["scaler"]
        assert scaler.n_samples_seen_ == len(train)
        np.testing.assert_allclose(scaler.mean_, x.iloc[train].mean().to_numpy())
    assert any(not np.allclose(m.named_steps["scaler"].mean_, x.mean().to_numpy())
               for m in results["estimator"])


@pytest.mark.parametrize("algorithm", ["LogisticRegression", "RandomForestClassifier"])
def test_small_training_is_reproducible(algorithm, frame):
    search = module.build_search(algorithm)
    assert search.cv.random_state == 42 and search.cv.n_splits == 5
    assert search.scoring._kwargs["beta"] == 2
    model = search.estimator
    if algorithm == "RandomForestClassifier":
        model.set_params(classifier__n_estimators=10, classifier__max_depth=3)
    x = module.feature_matrix(frame, module.FEATURE_SETS["B"])
    a, b = clone(model).fit(x, frame.label), clone(model).fit(x, frame.label)
    np.testing.assert_array_equal(module.probabilities(a, x), module.probabilities(b, x))


def test_binary_metrics_and_probability_column_mean_phishing_one():
    result = module.metrics([0, 0, 1, 1], [.1, .8, .7, .9], .7)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1
    assert all("macro_" + key in result for key in ("precision", "recall", "f1"))


def test_t2_matches_exhaustive_maximum_f2_with_precision_constraint():
    rng = np.random.default_rng(42)
    for _ in range(30):
        labels = np.array([0] * 15 + [1] * 10)
        scores = np.round(rng.random(25), 2)
        candidates = []
        for threshold in np.unique(scores):
            if threshold <= 0:
                continue
            m = module.metrics(labels, scores, threshold)
            if m["precision"] >= .85:
                candidates.append((m["f2"], m["recall"], m["precision"], -float(threshold)))
        expected = -max(candidates)[-1] if candidates else None
        assert module.choose_t2(labels, scores) == expected
    assert module.choose_t2([1, 0], [.1, .9]) is None


def test_t1_maximizes_coverage_with_low_phishing_contamination():
    labels = np.array([0] * 100 + [1] * 20)
    scores = np.linspace(.001, .99, 120)
    labels[50], labels[100] = 1, 0
    t2 = .9
    t1 = module.choose_t1(labels, scores, t2)
    mask = scores < t1
    assert 0 <= t1 < t2 and mask.any()
    assert labels[mask].mean() <= .01
    eligible = [t for t in np.unique(scores) if 0 < t < t2 and (scores < t).any()
                and labels[scores < t].mean() <= .01]
    assert t1 == max(eligible)
    assert module.choose_t1([1, 0], [.1, .2], .8) == 0


@pytest.mark.parametrize("probability,expected", [(0, "SAFE"), (.199, "SAFE"), (.2, "SUSPICIOUS"),
                                                 (.799, "SUSPICIOUS"), (.8, "DANGEROUS"), (1, "DANGEROUS")])
def test_category_boundaries(probability, expected):
    assert module.category(probability, .2, .8) == expected


@pytest.mark.parametrize("p,t1,t2", [(np.nan,.2,.8), (-.1,.2,.8), (1.1,.2,.8), (.5,.8,.2), (.5,.2,.2)])
def test_invalid_probabilities_or_thresholds(p, t1, t2):
    with pytest.raises(ValueError):
        module.category(p, t1, t2)


def test_zone_summary_uses_exact_t1_t2_rules():
    result = module.zone_summary([0, 1, 1, 0], [.1, .2, .8, .9], .2, .8)
    assert [result[name]["rows"] for name in ("SAFE", "SUSPICIOUS", "DANGEROUS")] == [1, 1, 2]
    assert result["SAFE"]["phishing_fraction"] == 0
    assert result["SUSPICIOUS"]["phishing_fraction"] == 1


def test_selection_uses_f2_constraint_and_deterministic_ties():
    def row(key, f2, precision):
        return dict(key=key, algorithm="LogisticRegression", feature_names=module.FEATURE_SETS["B"],
                    cv_f2_std=.01, t2=.5, validation_metrics=dict(f2=f2, precision=precision, recall=.9, roc_auc=.95))
    assert module.select_result([row("bad", .99, .84), row("best", .9, .9)])["key"] == "best"
    with pytest.raises(ValueError):
        module.select_result([row("bad", .99, .84)])


def test_serialization_metadata_and_new_urls_without_network(frame, tmp_path, monkeypatch):
    names = module.FEATURE_SETS["B"]
    model = module.build_search("LogisticRegression").estimator.fit(module.feature_matrix(frame, names), frame.label)
    path = tmp_path / "url_model.joblib"
    joblib.dump(model, path)
    metadata = dict(version="1.0.0", feature_names=names, t1=.2, t2=.8, dataset=dict(strip_www=True))
    (tmp_path / "url_model_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Предсказание не должно обращаться к сети")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    url = "https://www.new-example.test/Path?Token=XY"
    result = module.predict_url(url)
    restored = joblib.load(path)
    x = module.feature_matrix(pd.DataFrame([extract_url_features(normalize_url(url, strip_www=True))]), names)
    expected = float(module.probabilities(restored, x)[0])
    assert result == dict(probability=expected, category=module.category(expected, .2, .8))
    assert list(restored.feature_names_in_) == names
    np.testing.assert_array_equal(module.probabilities(model, x), module.probabilities(restored, x))


def test_small_training_does_not_change_sms_or_dataset_files(frame):
    before = module.protected_hashes()
    module.build_search("LogisticRegression").estimator.fit(
        module.feature_matrix(frame, module.FEATURE_SETS["A"]), frame.label)
    assert module.protected_hashes() == before
