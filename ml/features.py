"""Разбиение корпуса и извлечение признаков SMS."""

import json
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from ml.preprocessing import DATA_DIR, normalize_sms


SOCIAL_ENGINEERING_GROUPS = tuple(
    re.compile(rf"\b(?:{pattern})\b")
    for pattern in (
        r"urgent|immediately|asap|act now|final notice",
        r"payment|transfer|deposit|cash|refund",
        r"bank|credit card|debit card|card details",
        r"account (?:blocked|locked|suspended)|(?:block|lock|suspend) your account",
        r"win|winner|won|prize|claim your reward",
        r"verification code|security code|one[- ]time code|password|pin",
    )
)


def split_indices(labels: pd.Series) -> dict[str, list[int]]:
    """Возвращает позиции строк CSV с нуля для трёх выборок."""
    train, remainder = train_test_split(
        np.arange(len(labels)), test_size=0.30, random_state=42, stratify=labels,
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=42, stratify=labels.iloc[remainder],
    )
    return {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}


def _prepare_texts(texts):
    return [(text, normalize_sms(text)) for text in texts]


def _clean_texts(pairs):
    return [clean for _, clean in pairs]


def manual_features(pairs) -> csr_matrix:
    """Столбцы: длина, доля заглавных, число !, URL, сумма, группы правил."""
    rows = []
    for raw, clean in pairs:
        letters = sum(char.isalpha() for char in raw)
        uppercase = sum(char.isalpha() and char.isupper() for char in raw)
        rows.append([
            len(raw),
            uppercase / letters if letters else 0.0,
            raw.count("!"),
            int("<url>" in clean),
            int("<money>" in clean),
            sum(bool(group.search(clean)) for group in SOCIAL_ENGINEERING_GROUPS),
        ])
    return csr_matrix(np.asarray(rows, dtype=float).reshape(-1, 6))


def create_feature_pipeline() -> Pipeline:
    """Принимает последовательность исходных SMS и возвращает разреженные признаки."""
    words = Pipeline([
        ("text", FunctionTransformer(_clean_texts)),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
        )),
    ])
    chars = Pipeline([
        ("text", FunctionTransformer(_clean_texts)),
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
    ])
    return Pipeline([
        ("normalize", FunctionTransformer(_prepare_texts)),
        ("features", FeatureUnion([
            ("word", words),
            ("char", chars),
            ("manual", FunctionTransformer(manual_features)),
        ])),
    ])


def _print_top_tfidf(pipeline: Pipeline, x_train, labels: pd.Series) -> None:
    branches = dict(pipeline.named_steps["features"].transformer_list)
    names = [
        (branch, feature)
        for branch in ("word", "char")
        for feature in branches[branch].named_steps["tfidf"].get_feature_names_out()
    ]
    for label, name in ((0, "ham"), (1, "spam")):
        class_matrix = x_train[labels.to_numpy() == label, :len(names)]
        means = np.asarray(class_matrix.mean(axis=0)).ravel()
        top = np.argsort(-means, kind="stable")[:15]
        print(f"\n{name}: топ-15 признаков по среднему TF-IDF на train")
        for rank, index in enumerate(top, start=1):
            branch, feature = names[index]
            print(f"{rank:2d}. {branch:4s} {feature!r:24s} {means[index]:.6f}")


def main() -> None:
    messages = pd.read_csv(DATA_DIR / "sms_clean.csv", encoding="utf-8", keep_default_na=False,
                           dtype={"text_raw": str})
    splits = split_indices(messages["label"])
    train = messages.iloc[splits["train"]]
    validation = messages.iloc[splits["validation"]]
    test = messages.iloc[splits["test"]]

    pipeline = create_feature_pipeline()
    pipeline.fit(train["text_raw"])
    x_train = pipeline.transform(train["text_raw"])
    x_val = pipeline.transform(validation["text_raw"])
    x_test = pipeline.transform(test["text_raw"])

    for name, subset, matrix in (
        ("train", train, x_train), ("validation", validation, x_val), ("test", test, x_test),
    ):
        counts = subset["label"].value_counts()
        print(f"{name}: {len(subset)} SMS, ham={counts[0]} ({counts[0] / len(subset):.2%}), "
              f"spam={counts[1]} ({counts[1] / len(subset):.2%}), X={matrix.shape}")

    _print_top_tfidf(pipeline, x_train, train["label"])
    (DATA_DIR / "sms_split.json").write_text(
        json.dumps(splits, separators=(",", ":")) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
