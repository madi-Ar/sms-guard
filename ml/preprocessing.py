"""Нормализация SMS и подготовка очищенного корпуса."""

import re
import unicodedata
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

AMOUNT = r"(?:\d{1,3}(?:[ ,.]\d{3})+|\d+)(?:[.,]\d+)?"
CURRENCY = r"[$£₸]|(?:тг|руб)\b\.?"
ENTITIES = re.compile(
    r"(?P<url>\b(?:https?://|www\.)[^\s<>]+"
    r"|(?<![\w@.-])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
    r"(?:com|org|net|edu|gov|io|info|biz|ru|kz|uk|рф|қаз|example)"
    r"(?![\w@-])(?:[/?#][^\s<>]*)?)"
    r"|(?P<email>[\w.!#$%&'*+/=?^{}|~-]+@[\w-]+(?:\.[\w-]+)+)"
    rf"|(?P<money>(?:{CURRENCY}) *{AMOUNT}|(?<!\d){AMOUNT} *(?:{CURRENCY}))"
    r"|(?P<phone>(?<![\w+])(?:"
    r"\+\d{7,15}"
    r"|\+\d{1,3}(?:[ .-]?\(\d{2,5}\)|[ .-]\d{2,5})(?:[ .-]\d{2,6}){1,3}"
    r"|(?:[78] ?)?\(\d{3}\) ?\d{3}[ .-]?\d{2}[ .-]?\d{2}"
    r"|(?:[78][ .-])?\d{3}[ .-]\d{3}[ .-]\d{2}[ .-]?\d{2}"
    r"|\d{3}[ .-]\d{3}[ .-]\d{4}|\d{5}[ .-]\d{6}|[078]\d{10}|\d{10}"
    r")(?!\w))"
    rf"|(?P<num>{AMOUNT})"
)
URL_END_PUNCTUATION = "!?.,:;)]}»\"'"


def _entity_token(match: re.Match[str]) -> str:
    if match.lastgroup == "url":
        url = match.group()
        suffix = url[len(url.rstrip(URL_END_PUNCTUATION)):]
        return f" <url> {suffix}"
    return f" <{match.lastgroup}> "


def normalize_sms(
    text: str, *, stop_words: frozenset[str] = frozenset()
) -> str:
    """Нормализует SMS. Для удаления стоп-слов передайте набор слов в нижнем регистре."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(
        " " if char.isspace() else char
        for char in text
        if char.isspace() or unicodedata.category(char) not in {"Cc", "Cf"}
    )
    text = ENTITIES.sub(_entity_token, text)
    text = re.sub(r"([!?])", r" \1 ", text)
    return " ".join(token for token in text.split() if token not in stop_words)


def main() -> None:
    messages = pd.read_csv(
        DATA_DIR / "sms_raw.csv", encoding="utf-8", keep_default_na=False,
        dtype={"text": str},
    ).rename(columns={"text": "text_raw"})
    messages["text_clean"] = messages["text_raw"].map(normalize_sms)
    empty = messages["text_clean"].eq("")
    nonempty = messages.loc[~empty]

    labels_per_text = nonempty.groupby("text_clean")["label"].transform("nunique")
    conflicting = labels_per_text.gt(1)
    consistent = nonempty.loc[~conflicting]
    duplicate = consistent.duplicated(subset="text_clean")
    clean = consistent.loc[~duplicate]

    clean[["id", "text_raw", "text_clean", "label"]].to_csv(
        DATA_DIR / "sms_clean.csv", index=False, encoding="utf-8",
    )

    print(f"Записей до обработки: {len(messages)}")
    print(f"Текст изменился: {messages['text_raw'].ne(messages['text_clean']).sum()}")
    for token in ("url", "email", "phone", "money", "num"):
        count = messages["text_clean"].str.contains(f"<{token}>", regex=False).sum()
        print(f"Строки с <{token}>: {count}")
    print(f"Групп с конфликтом меток: {nonempty.loc[conflicting, 'text_clean'].nunique()}")
    for stage, rows in (
        ("Пустые", messages.loc[empty]),
        ("Конфликтующие", nonempty.loc[conflicting]),
        ("Дубликаты без конфликта", consistent.loc[duplicate]),
        ("Удалено всего", messages.loc[~messages.index.isin(clean.index)]),
        ("Осталось", clean),
    ):
        counts = rows["label"].value_counts()
        print(f"{stage}: {len(rows)} (ham: {counts.get(0, 0)}, spam: {counts.get(1, 0)})")


if __name__ == "__main__":
    main()
