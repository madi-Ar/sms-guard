"""Разовая калибровка Risk Score без Test и без перезаписи итоговых моделей."""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json

import joblib
import pandas as pd
from sklearn.base import clone
import yaml

from apps.analysis.services import extract_urls
from ml.predictor import ModelPredictor
from ml.risk import CONFIG_PATH, ROOT, compute_risk, load_config
from ml.rules import find_social_engineering_groups
from ml.url_dataset import normalize_url
from ml.url_features import extract_url_features
from ml.url_model import ARTIFACTS, protected_hashes


# Предварительно заданная небольшая сетка. Границы и правила ТЗ не подбираются.
CANDIDATES = ((.5, .3, .2), (.6, .2, .2), (.4, .3, .3), (.3, .3, .4), (.2, .3, .5))
RESULT_PATH = ARTIFACTS / "risk_calibration.json"


def input_hashes():
    result = protected_hashes()
    for path in ARTIFACTS.glob("*model*"):
        result[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return result


def collect_observations(frame, predictor, text_column):
    """Предсказания вычисляются один раз; перебор весов использует готовые числа."""
    rows = []
    for row in frame.itertuples(index=False):
        text = getattr(row, text_column)
        sms = predictor.predict_sms(text)
        urls = []
        for url in extract_urls(text):
            normalized = normalize_url(url, strip_www=predictor.url_metadata["dataset"]["strip_www"])
            flags = extract_url_features(normalized)
            prediction = predictor.predict_url_features(flags)
            urls.append((prediction["probability"], flags))
        rows.append(dict(label=int(row.label), p_sms=sms["probability"],
                         p_url_max=max((p for p, _ in urls), default=None),
                         se_groups=find_social_engineering_groups(text),
                         url_flags=[flags for _, flags in urls]))
    return rows


def evaluate(observations, config, t2):
    scores = [compute_risk(row["p_sms"], row["p_url_max"], row["se_groups"], row["url_flags"],
                           config=config, t2=t2).risk_score for row in observations]
    medium, high = config["levels"]["medium"], config["levels"]["high"]
    positives = sum(row["label"] == 1 for row in observations)
    negatives = sum(row["label"] == 0 for row in observations)
    if not positives or not negatives:
        raise ValueError("Для калибровки требуются оба класса")
    found = sum(row["label"] == 1 and score >= medium for row, score in zip(observations, scores))
    false_high = sum(row["label"] == 0 and score >= high for row, score in zip(observations, scores))
    recall, false_rate = found / positives, false_high / negatives
    return dict(rows=len(observations), positive=positives, negative=negatives,
                detected_medium_high=found, false_high=false_high,
                recall_medium_high=recall, false_high_rate=false_rate,
                meets_targets=recall >= .95 and false_rate <= .03)


def violation(row):
    """Равноправная сумма относительных нарушений четырёх ограничений EN и ru/kk."""
    return sum(max(0., .95 - row[name]["recall_medium_high"]) / .95 +
               max(0., row[name]["false_high_rate"] - .03) / .03
               for name in ("en_validation", "ru_kk_development"))


def calibrate():
    before = input_hashes()
    sms_meta = json.loads((ARTIFACTS / "sms_model_meta.json").read_text(encoding="utf-8"))
    url_meta = json.loads((ARTIFACTS / "url_model_meta.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "data/processed/sms_split.json").read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / "data/processed/sms_clean.csv", keep_default_na=False)
    train = data.iloc[splits["train"]]
    validation = data.iloc[splits["validation"]]
    if len(train) != 3594 or len(validation) != 770 or set(splits["train"]) & set(splits["validation"]):
        raise ValueError("Ожидается прежнее разделение Train/Validation SMS")
    # clone копирует только уже выбранную конфигурацию, не состояние fit на Train+Validation.
    saved_sms = joblib.load(ARTIFACTS / "sms_model.joblib")
    train_only_sms = clone(saved_sms).fit(train.text_raw, train.label)
    url_model = joblib.load(ARTIFACTS / "url_model.joblib")
    predictor = ModelPredictor(sms_model=train_only_sms, url_model=url_model,
                               sms_meta=sms_meta, url_meta=url_meta)
    custom = pd.read_csv(ROOT / "data/raw/custom_ru_kk.csv", keep_default_na=False)
    if custom.language.value_counts().to_dict() != {"ru":100, "kk":50} or not custom.label.isin([0,1]).all():
        raise ValueError("Ожидаются прежние 100 ru и 50 kk примеров")
    print("Collecting train-only Validation and ru/kk development observations", flush=True)
    observations = {"en_validation": collect_observations(validation, predictor, "text_raw"),
                    "ru_kk_development": collect_observations(custom, predictor, "text")}
    for language in ("ru", "kk"):
        observations[language] = [row for row, lang in zip(observations["ru_kk_development"], custom.language) if lang == language]
    base = load_config()
    results = []
    for values in CANDIDATES:
        config = deepcopy(base)
        config["weights"] = dict(zip(("sms", "url", "social_engineering"), values))
        row = {"weights":config["weights"], "config":config}
        row.update({name:evaluate(rows, config, url_meta["t2"]) for name,rows in observations.items()})
        row["normalized_violation"] = violation(row)
        results.append(row)
        print(json.dumps({"weights":row["weights"], "en":row["en_validation"], "ru_kk":row["ru_kk_development"]}), flush=True)
    selected = min(range(len(results)), key=lambda i: (
        results[i]["normalized_violation"],
        -min(results[i][name]["recall_medium_high"] for name in ("en_validation","ru_kk_development")), i))
    if input_hashes() != before:
        raise RuntimeError("Защищённые файлы изменились при калибровке")
    report = dict(calibrated_at=datetime.now(timezone.utc).isoformat(),
        sms_model=dict(algorithm=sms_meta["algorithm"], parameters=sms_meta["parameters"],
                       features=sms_meta["feature_configuration"], train_only_rows=3594,
                       validation_rows=770, reconstruction="clone saved fixed Pipeline, fit only Train; no model saved"),
        url_t2=url_meta["t2"], test_used=False, protected_sha256=before,
        selected_index=selected, selected_config=results[selected]["config"], results=results,
        selection_policy="Minimum sum of normalized violations on EN Validation and combined ru/kk; then largest worst recall; then candidate order",
        custom_set_status="calibration/development, no longer independent test",
        target_semantics="UCI label 1 is spam, a proxy and not confirmed fraud")
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    CONFIG_PATH.write_text("# Выбрано калибровкой №13; границы и override не подбирались.\n"
                           "# URL t2 берётся из метаданных итоговой URL-модели.\n" +
                           yaml.safe_dump(report["selected_config"], sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Selected configuration {selected + 1}; results: {RESULT_PATH.name}", flush=True)
    return report


if __name__ == "__main__":
    calibrate()
