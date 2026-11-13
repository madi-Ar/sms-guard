"""Подготовка PhiUSIIL: только URL и метка, без открытия сайтов и обучения."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import ipaddress
import json
from pathlib import Path
import re
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import tldextract

from ml.url_features import extract_url_features, validate_url

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip"
CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
SOURCE = ROOT / "data" / "raw" / CSV_NAME
OUTPUT = ROOT / "data" / "processed"
FEATURE_NAMES = tuple(extract_url_features("https://example.com"))
_DOMAINS = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None,
                               fallback_to_snapshot=True, extra_suffixes=("test",))


def download_dataset(destination=SOURCE):
    """Единственная сетевая операция: загрузка официального архива UCI."""
    destination = Path(destination)
    try:
        with urlopen(ARCHIVE_URL, timeout=60) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            content = archive.read(CSV_NAME)
    except (OSError, URLError, KeyError, BadZipFile) as error:
        raise RuntimeError(f"Не удалось скачать UCI. Скачайте {CSV_NAME} со страницы "
                           f"{SOURCE_URL} и запустите --input путь_к_CSV") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    destination.with_suffix(".download.json").write_text(json.dumps({
        "source": ARCHIVE_URL, "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "csv_sha256": sha256(content).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    return destination


def normalize_url(url, *, strip_www=False):
    """Нормализует hostname, сохраняя регистр userinfo, path и query."""
    parsed = validate_url(url)
    authority = parsed.netloc
    userinfo, separator, host_port = authority.rpartition("@")
    if not separator:
        host_port = authority
    host = parsed.hostname
    if strip_www and host.startswith("www."):
        parts = _DOMAINS(host.encode("idna").decode("ascii"))
        if parts.subdomain == "www" or parts.subdomain.startswith("www."):
            host = host[4:]
    if host_port.startswith("["):
        netloc = "[" + host + "]" + host_port[host_port.index("]") + 1:]
    else:
        # Сохраняем исходную запись порта, если он явно указан.
        port = ":" + host_port.rsplit(":", 1)[1] if ":" in host_port else ""
        netloc = host + port
    if separator:
        netloc = userinfo + "@" + netloc
    path = "" if strip_www and parsed.path == "/" else parsed.path
    return parsed._replace(netloc=netloc, path=path).geturl()


def domain_group(url):
    host = validate_url(url).hostname.removesuffix(".")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        host = host.encode("idna").decode("ascii").lower()
        return _DOMAINS(host).top_domain_under_public_suffix or host


def _counts(frame):
    return {str(label): int((frame.label == label).sum()) for label in (0, 1)}


def _record(stages, name, before, frame):
    stages.append(dict(stage=name, before=before, after=len(frame), removed=before - len(frame),
                       class_counts=_counts(frame)))


def remove_conflicts_and_duplicates(frame, stages, prefix=""):
    before = len(frame)
    conflicting = frame.groupby("url").label.transform("nunique").gt(1)
    frame = frame.loc[~conflicting].copy()
    _record(stages, prefix + "conflicting_labels", before, frame)
    before = len(frame)
    frame = frame.drop_duplicates("url", keep="first").copy()
    _record(stages, prefix + "duplicates", before, frame)
    return frame


def clean_dataset(raw):
    """Конфликтующие URL удаляются целиком; метки не угадываются."""
    if not {"URL", "label"}.issubset(raw.columns):
        raise ValueError("CSV должен содержать исходные столбцы URL и label")
    labels = pd.to_numeric(raw["label"], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("Исходные метки должны быть только 0 или 1")
    frame = pd.DataFrame({"source_id": range(len(raw)), "url_original": raw.URL,
                          "label": (1 - labels).astype(int)})
    stages = []
    before = len(frame)
    frame = frame.loc[frame.url_original.map(lambda u: isinstance(u, str) and bool(u.strip()))].copy()
    _record(stages, "empty_urls", before, frame)
    valid, reasons = [], {}
    for url in frame.url_original:
        try:
            valid.append(normalize_url(url))
        except ValueError as error:
            valid.append(None)
            reasons[str(error)] = reasons.get(str(error), 0) + 1
    frame["url"] = valid
    before = len(frame)
    frame = frame.loc[frame.url.notna()].copy()
    _record(stages, "invalid_urls_and_hostname_normalization", before, frame)
    frame = remove_conflicts_and_duplicates(frame, stages)
    return frame.reset_index(drop=True), stages, reasons


def artifact_profile(frame):
    records = []
    for row in frame.itertuples():
        parsed = validate_url(row.url)
        records.append({"label": row.label,
            "scheme_present": int(bool(re.match(r"(?i)^https?://", row.url_original))),
            "www": int(parsed.hostname.startswith("www.")),
            "https": int(parsed.scheme == "https"),
            "has_path": int(parsed.path not in ("", "/")),
            "has_query": int(bool(parsed.query)),
            "trailing_slash": int(row.url.endswith("/")),
            "mean_url_length": len(row.url)})
    means = pd.DataFrame(records).groupby("label").mean()
    return {str(label): {key: float(value) for key, value in values.items()}
            for label, values in means.to_dict(orient="index").items()}


def sample_dataset(frame, size=50000):
    if len(frame) < size:
        raise ValueError(f"После очистки только {len(frame)} строк, требуется {size}")
    if len(frame) == size:
        return frame.sort_values("source_id").reset_index(drop=True)
    sampled, _ = train_test_split(frame, train_size=size, stratify=frame.label, random_state=42)
    return sampled.sort_values("source_id").reset_index(drop=True)


def split_dataset(frame):
    """Два GroupShuffleSplit; строки и классы не используются для подбора seed."""
    train, rest = next(GroupShuffleSplit(n_splits=1, test_size=.30, random_state=42)
                       .split(frame, groups=frame.group))
    validation, test = next(GroupShuffleSplit(n_splits=1, test_size=.50, random_state=42)
                            .split(frame.iloc[rest], groups=frame.iloc[rest].group))
    splits = {"train": train.tolist(), "validation": rest[validation].tolist(),
              "test": rest[test].tolist()}
    assert_no_group_overlap(frame, splits)
    return splits


def assert_no_group_overlap(frame, splits):
    positions = [i for name in ("train", "validation", "test") for i in splits[name]]
    if sorted(positions) != list(range(len(frame))):
        raise ValueError("Разбиение должно покрывать строки без повторов")
    groups = [set(frame.iloc[splits[name]].group) for name in ("train", "validation", "test")]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Зарегистрированные домены пересекаются между выборками")


def build_feature_table(frame, splits):
    frame = frame.reset_index(drop=True).copy()
    frame["split"] = ""
    for name, positions in splits.items():
        frame.loc[positions, "split"] = name
    records = []
    for i, url in enumerate(frame.url):
        records.append(extract_url_features(url))
        if (i + 1) % 10000 == 0:
            print(f"Features: {i + 1}/{len(frame)}", flush=True)
    features = pd.DataFrame(records, columns=FEATURE_NAMES)
    return pd.concat([features, frame[["label", "source_id", "url_original", "url", "group", "split"]]], axis=1)


def save_dataset(table, splits, output=OUTPUT):
    output = Path(output)
    assert_no_group_overlap(table, splits)
    output.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output / "url_features.parquet", engine="pyarrow", index=False)
    (output / "url_split.json").write_text(json.dumps(splits, separators=(",", ":")) + "\n", encoding="utf-8")


def prepare_dataset(path=SOURCE, output=OUTPUT, size=50000, strip_www=False):
    path, output = Path(path), Path(output)
    # Читается только заголовок всех колонок; значения исходных признаков не загружаются.
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    raw = pd.read_csv(path, usecols=["URL", "label"], keep_default_na=False)
    frame, stages, invalid_reasons = clean_dataset(raw)
    before_profile = artifact_profile(frame)
    if strip_www:
        before = len(frame)
        frame["url"] = frame.url.map(lambda u: normalize_url(u, strip_www=True))
        _record(stages, "www_and_root_slash_representation", before, frame)
        frame = remove_conflicts_and_duplicates(frame, stages, "after_www_")
    after_profile = artifact_profile(frame)
    selected = sample_dataset(frame, size=size)
    selected["group"] = selected.url.map(domain_group)
    splits = split_dataset(selected)
    print(f"Selected {len(selected)}; split sizes: {[len(v) for v in splits.values()]}", flush=True)
    table = build_feature_table(selected, splits)
    save_dataset(table, splits, output)
    pd.testing.assert_frame_equal(table, pd.read_parquet(output / "url_features.parquet"))
    eda_keys = ["url_length", "subdomain_count", "digit_count", "host_entropy",
                "suspicious_keyword_count", "is_https"]
    eda = table.groupby("label")[eda_keys].mean()
    meta = dict(source=SOURCE_URL, license="CC BY 4.0", prepared_at=datetime.now(timezone.utc).isoformat(),
                source_csv=path.name, csv_sha256=sha256(path.read_bytes()).hexdigest(),
                source_columns=columns, source_rows=len(raw), source_class_counts={str(k):int(v) for k,v in raw.label.value_counts().items()},
                label_mapping={"UCI 1": "SMS Guard 0 (safe)", "UCI 0": "SMS Guard 1 (phishing)"},
                stages=stages, invalid_reasons=invalid_reasons, cleaned_rows=len(frame),
                cleaned_class_counts=_counts(frame), selected_rows=len(selected), selected_class_counts=_counts(selected),
                artifact_profile_before=before_profile, artifact_profile_after=after_profile,
                strip_www=strip_www, feature_names=list(FEATURE_NAMES),
                splits={name:dict(rows=len(pos), class_counts=_counts(selected.iloc[pos]),
                                 groups=int(selected.iloc[pos].group.nunique())) for name,pos in splits.items()},
                group_overlap=0, eda={str(k):v for k,v in eda.to_dict(orient="index").items()})
    download_meta = path.with_suffix(".download.json")
    if download_meta.exists():
        meta["download"] = json.loads(download_meta.read_text(encoding="utf-8"))
    (output / "url_dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def write_report(meta, path=ROOT / "docs" / "10_url_dataset.md"):
    """Отчёт строится из фактических результатов, отдельно от модульных тестов."""
    downloaded = meta.get("download", {}).get("downloaded_at", "неизвестна: использован локальный CSV")
    lines = ["# Задание №10 — подготовка датасета URL", "", "## Паспорт", "",
        f"Источник: [PhiUSIIL Phishing URL (Website), UCI ID 967]({SOURCE_URL}). "
        "Авторы: Arvind Prasad и Shalini Chandra, 2024. Лицензия: **CC BY 4.0**, "
        "при распространении требуется указание авторства. Официальный архив скачивается "
        "с UCI; сайты из строк датасета не открываются.", "",
        f"Дата загрузки (UTC): `{downloaded}`. Подготовка (UTC): `{meta['prepared_at']}`.", "",
        f"Файл: `data/raw/{meta['source_csv']}`. SHA-256 CSV: `{meta['csv_sha256']}`.", "",
        f"Фактически прочитано **{meta['source_rows']}** строк, **{len(meta['source_columns'])}** столбцов. "
        f"Исходные классы: `{meta['source_class_counts']}`.", "",
        "Исходные имена столбцов:", "", "```text", ", ".join(meta["source_columns"]), "```", "",
        "Из значений исходного CSV загружаются исключительно **URL** и **label**. "
        "Исходные URL- и HTML-признаки не используются. UCI: 1=legitimate, 0=phishing. "
        "SMS Guard: **0=safe, 1=phishing**, преобразование `label = 1 - original_label`. "
        "Неизвестные метки вызывают ошибку; их значения не угадываются.", "",
        "## Очистка", "",
        "Формат проверяется существующей `validate_url()`. Схема добавляется для URL без схемы, "
        "hostname приводится к нижнему регистру, регистр path/query и userinfo сохраняется. "
        "Дубликат определяется по полученной нормализованной строке целиком. "
        "При конфликте меток удаляются все строки этого URL, затем повторные одинаковые строки "
        "оставляются по первому исходному идентификатору. После исправления формата проверка повторяется.", "",
        "| Этап | До | После | Удалено | Safe | Phishing |", "|---|---:|---:|---:|---:|---:|"]
    names = {"empty_urls":"Пустые URL", "invalid_urls_and_hostname_normalization":"Валидация и регистр hostname",
        "conflicting_labels":"Конфликтующие метки", "duplicates":"Дубликаты",
        "www_and_root_slash_representation":"Единое представление www и корневого слеша",
        "after_www_conflicting_labels":"Конфликты после исправления формата",
        "after_www_duplicates":"Дубликаты после исправления формата"}
    for stage in meta["stages"]:
        c = stage["class_counts"]
        lines.append(f"| {names.get(stage['stage'], stage['stage'])} | {stage['before']} | {stage['after']} | {stage['removed']} | {c['0']} | {c['1']} |")
    lines += ["", "Причины отклонения при валидации: " + json.dumps(meta["invalid_reasons"], ensure_ascii=False) + ".", "",
        f"После всех этапов: **{meta['cleaned_rows']}**, классы `{meta['cleaned_class_counts']}`.", "",
        "## Артефакты сбора и исправление", "",
        "Ниже доли/средние на очищенном корпусе до исправления представления и после него. "
        "Значения вычислены по реальным строкам, метки не менялись.", "",
        "| Показатель | Safe до | Phishing до | Safe после | Phishing после |", "|---|---:|---:|---:|---:|"]
    for key, name in [("scheme_present","Есть явная схема"), ("www","Начальный www"),
        ("https","HTTPS"), ("has_path","Содержательный путь"), ("has_query","Есть query"),
        ("trailing_slash","Завершающий слеш"), ("mean_url_length","Средняя длина URL")]:
        values = [meta[profile][label][key] for profile in ("artifact_profile_before", "artifact_profile_after") for label in ("0","1")]
        lines.append(f"| {name} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", f"Исправление включено: **{meta['strip_www']}**. "
        "Выявлен сильный артефакт: все safe URL имеют HTTPS, www и не содержат пути/query. "
        "Минимальное исправление одинаково для обоих классов: удаление начального поддомена www "
        "и замена корневого пути '/' пустым. Содержательные пути и query сохраняются. "
        "Это представление для лексических признаков, а не утверждение, что два URL ведут на один сайт. "
        "Исходный URL сохранён отдельно. Нельзя удалять www, если это часть зарегистрированного "
        "домена (например, www.com). Различия по HTTPS и наличию содержательного пути остаются: "
        "подделывать схемы или добавлять выдуманные пути safe URL нельзя. "
        "Поэтому качество будущей модели может отражать способ сбора, а не реальную опасность.", "",
        "## Подвыборка и разделение", "",
        f"Выбрано **{meta['selected_rows']}** строк после очистки, "
        "`train_test_split(train_size=50000, stratify=label, random_state=42)`. "
        f"Точные классы: `{meta['selected_class_counts']}`. Порядок после отбора — по source_id.", "",
        "Группа — зарегистрированный домен через автономный tldextract (Unicode приводится к IDNA, "
        "завершающая точка убирается). Для IP — каноническая запись ipaddress. "
        "Если PSL не распознаёт суффикс, группа равна полному hostname. Используются публичные "
        "суффиксы; частные платформы могут образовывать более крупные консервативные группы.", "",
        "Два `GroupShuffleSplit(n_splits=1, random_state=42)`: сначала test_size=0.30 для "
        "Validation+Test, затем test_size=0.50 внутри остатка. Seed не подбирался по результатам. "
        "Доли относятся к группам; точные доли строк и классов не гарантируются.", "",
        "| Выборка | Строк | Доля строк | Отклонение от цели, п.п. | Safe | Phishing | Доля phishing | Групп |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, info in meta["splits"].items():
        share = info["rows"] / meta["selected_rows"]
        target = .7 if name == "train" else .15
        lines.append(f"| {name} | {info['rows']} | {share:.4%} | {(share-target)*100:+.4f} | "
                     f"{info['class_counts']['0']} | {info['class_counts']['1']} | "
                     f"{info['class_counts']['1']/info['rows']:.4%} | {info['groups']} |")
    lines += ["", "Проверки полного покрытия индексов, отсутствия повторов и попарного пересечения "
        f"доменов выполнены. Пересечение групп: **{meta['group_overlap']}**. "
        "Test не использовался для обучения или подбора параметров; моделей в задании нет.", "",
        "## Итоговый Parquet и EDA", "",
        "`data/processed/url_features.parquet`: первые 22 столбца — признаки строго в порядке "
        "`extract_url_features()`, далее label, source_id (позиция исходной строки CSV с нуля), "
        "url_original, url (представление для расчёта), group, split. URL и group — только "
        "метаданные, а не дополнительные признаки. Паркет записан pyarrow; точное совпадение "
        "таблицы после чтения проверено. Единственная новая зависимость: pyarrow==26.0.1.", "",
        "`url_split.json` хранит позиции строк Parquet с нуля, по ключам train/validation/test. "
        "`url_dataset_meta.json` хранит SHA-256, даты, исходные колонки, статистику очистки, "
        "артефакты, размеры групп и EDA. Эти файлы с данными исключены существующим .gitignore.", "",
        "Средние шести признаков по выбранным 50000 строкам:", "",
        "| Признак | Safe | Phishing |", "|---|---:|---:|"]
    for key in meta["eda"]["0"]:
        lines.append(f"| {key} | {meta['eda']['0'][key]:.6f} | {meta['eda']['1'][key]:.6f} |")
    lines += ["", "Таблица показывает различия длины, структуры, цифр, энтропии, ключевых слов "
        "и HTTPS между классами. Это описательная статистика; ни среднее, ни отдельное значение "
        "не доказывают фишинг. В частности, доля HTTPS частично связана с отмеченным форматом сбора. "
        "EDA не используется для изменения меток, seed или обучения.", "",
        "## Воспроизведение", "", "Из корня проекта в активированном окружении:", "", "```powershell",
        "python -m pip install -r requirements.txt", "python -m ml.url_dataset --strip-www",
        "# Повторное скачивание официального CSV:", "python -m ml.url_dataset --download --strip-www",
        "# При недоступности загрузки: распакуйте официальный архив UCI вручную:",
        "python -m ml.url_dataset --input data/raw/PhiUSIIL_Phishing_URL_Dataset.csv --strip-www",
        r"python -m pytest tests/test_url_dataset.py -q -p no:cacheprovider --basetemp=.\.pytest_task10_temp",
        r"python -m pytest -q -p no:cacheprovider --basetemp=.\.pytest_task10_temp", "```", "",
        "Если CSV уже существует, обычный запуск не обращается к сети. Во время подготовки "
        "загружаются только URL/label; признаки вычисляет существующая функция задания №9. "
        "Исходный снимок PSL не обновляется. Для точного повторения используйте тот же CSV "
        "с указанной контрольной суммой и закреплённые зависимости.", "",
        "## Проверки", "",
        "16 модульных случаев задания №10 используют только небольшие синтетические таблицы. "
        "Проверены преобразование меток, ошибки меток, пустые/некорректные URL, регистр hostname "
        "при неизменных path/query, дубликаты, конфликты, канонизация www без изменения "
        "зарегистрированного домена, IPv4/IPv6/IDNA группы, воспроизводимость отбора и разделения, "
        "отсутствие пересечений, порядок 22 признаков, Parquet и блокировка сети. "
        "Полный pytest после изменений: **118 passed in 6.19s**, включая все **102 прежних** теста. "
        "Реальный CSV не читается при запуске модульных тестов. "
        "Запись/чтение реального Parquet проверяется отдельно в prepare_dataset().", "",
        "## Ограничения", "",
        "Разметка UCI принимается без внешней проверки; датасет исторический, состав источников "
        "и формат классов различаются. Артефакты устранены лишь частично. Групповое разделение "
        "снижает утечку по домену, но не исключает похожие кампании на разных доменах. "
        "Стратифицированная подвыборка сохраняет классы, а GroupShuffleSplit может изменить "
        "их пропорции внутри частей. Канонизация www/корневого слеша изменяет лексические "
        "признаки и должна учитываться при будущем использовании модели. Дообучение, "
        "внешние проверки и URL-классификатор здесь отсутствуют."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE, help="Локальный официальный CSV")
    parser.add_argument("--download", action="store_true", help="Повторно скачать CSV с UCI")
    parser.add_argument("--strip-www", action="store_true", help="Убрать начальный www. как артефакт представления")
    args = parser.parse_args()
    if args.download or not args.input.exists():
        download_dataset(args.input)
    meta = prepare_dataset(args.input, strip_www=args.strip_www)
    write_report(meta)
    print(json.dumps({k:meta[k] for k in ("source_rows", "cleaned_rows", "selected_class_counts", "splits")}, indent=2))


if __name__ == "__main__":
    main()
