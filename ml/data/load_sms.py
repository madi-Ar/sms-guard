"""Загрузка исходного корпуса UCI SMS Spam Collection."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_DIR / "data" / "raw" / "SMSSpamCollection"
OUTPUT = PROJECT_DIR / "data" / "processed" / "sms_raw.csv"
LABELS = {"ham": 0, "spam": 1}
UCI_ARCHIVE = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"
UCI_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"


def download_sms(destination: Path) -> None:
    with urlopen(UCI_ARCHIVE, timeout=30) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        raw = archive.read("SMSSpamCollection")
    if sha256(raw).hexdigest() != UCI_SHA256:
        raise ValueError("Контрольная сумма SMSSpamCollection не совпала с ожидаемой")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as target:
        target.write(raw)


def load_sms(path: Path = SOURCE) -> pd.DataFrame:
    records = []
    try:
        with path.open(encoding="utf-8", errors="strict") as source:
            for line_number, line in enumerate(source, start=1):
                label, separator, text = line.removesuffix("\n").partition("\t")
                if not separator or label not in LABELS or not text:
                    raise ValueError(f"{path.name}, строка {line_number}: ожидались метка ham/spam, табуляция и текст")
                records.append((line_number, text, LABELS[label]))
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name}: исходный файл должен быть в UTF-8") from error

    if not records:
        raise ValueError(f"{path.name}: исходный файл пуст")

    return pd.DataFrame(records, columns=["id", "text", "label"])


def main() -> None:
    if not SOURCE.exists():
        download_sms(SOURCE)
    messages = load_sms()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    messages.to_csv(OUTPUT, index=False, encoding="utf-8")

    print(f"Количество строк: {len(messages)}")
    print("Столбцы и типы:")
    print(messages.dtypes.to_string())
    for name, label in LABELS.items():
        examples = messages.loc[messages["label"] == label]
        print(f"\n{name} ({label}): {len(examples)} ({len(examples) / len(messages):.2%})")
        print(examples.head(5).to_string(index=False))
    print(f"\nРезультат: {OUTPUT.relative_to(PROJECT_DIR).as_posix()}")


if __name__ == "__main__":
    main()
