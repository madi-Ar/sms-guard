import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.naive_bayes import MultinomialNB

from ml.features import create_feature_pipeline, manual_features, split_indices
from ml.preprocessing import normalize_sms


def test_stratified_split_is_reproducible_and_disjoint():
    labels = pd.Series([0] * 80 + [1] * 20)
    splits = split_indices(labels)
    assert splits == split_indices(labels)
    train, validation, test = map(set, splits.values())
    assert (len(train), len(validation), len(test)) == (70, 15, 15)
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == set(range(100))
    assert [labels.iloc[part].sum() for part in splits.values()] == [14, 3, 3]


def test_manual_features_use_raw_and_normalized_text():
    raw = "URGENT! urgent! Bank card payment. Account locked. WIN a prize! Send password. £20 https://x.example"
    clean = normalize_sms(raw)
    actual = manual_features([(raw, clean), ("123", normalize_sms("123"))]).toarray()
    assert actual[0, 0] == len(raw)
    assert actual[0, 1] == 12 / 74
    assert actual[0, 2:].tolist() == [3, 1, 1, 6]
    assert actual[1].tolist() == [3, 0, 0, 0, 0, 0]


def test_transform_keeps_train_vocabulary_and_idf():
    pipeline = create_feature_pipeline()
    train = ["hello world", "hello again", "world today", "today again"]
    pipeline.fit(train)
    x_train = pipeline.transform(train)
    branches = dict(pipeline.named_steps["features"].transformer_list)
    vectorizers = [branches[name].named_steps["tfidf"] for name in ("word", "char")]
    vocabularies = [vectorizer.vocabulary_.copy() for vectorizer in vectorizers]
    weights = [vectorizer.idf_.copy() for vectorizer in vectorizers]

    x_val = pipeline.transform(["HELLO validationunique!"])
    x_test = pipeline.transform(["https://x.example testunique"])

    assert "validationunique" not in vectorizers[0].vocabulary_
    assert "testunique" not in vectorizers[0].vocabulary_
    assert x_train.shape[0] == 4
    for matrix in (x_train, x_val, x_test):
        assert issparse(matrix)
        assert matrix.shape[1] == x_train.shape[1]
        assert (matrix.data >= 0).all()
    assert x_val.shape[0] == x_test.shape[0] == 1
    assert x_val[0, -4] == 1  # !
    assert x_test[0, -3] == 1  # URL
    for vectorizer, vocabulary, idf in zip(vectorizers, vocabularies, weights):
        assert vectorizer.vocabulary_ == vocabulary
        np.testing.assert_array_equal(vectorizer.idf_, idf)


def test_pipeline_accepts_future_classifier_step():
    pipeline = create_feature_pipeline()
    pipeline.steps.append(("classifier", MultinomialNB()))
    assert isinstance(clone(pipeline).named_steps["classifier"], MultinomialNB)
