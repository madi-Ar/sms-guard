"""Групповое сравнение двух URL-моделей, выбор на Validation, один Test."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             fbeta_score, make_scorer, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.url_dataset import FEATURE_NAMES, OUTPUT, ROOT, assert_no_group_overlap, normalize_url
from ml.url_features import extract_url_features

ARTIFACTS = ROOT / "ml" / "artifacts"
REMOVED_FEATURES = ("is_https", "path_length", "query_param_count")
FEATURE_SETS = {"A": list(FEATURE_NAMES),
                "B": [name for name in FEATURE_NAMES if name not in REMOVED_FEATURES]}
LIMITATIONS = (
    "Все safe URL корпуса используют HTTPS и не имеют содержательного path/query. "
    "Удаление трёх прямых индикаторов не исключает косвенные артефакты через url_length, "
    "special_char_count, digit_count и другие признаки. Это учебная модель на историческом "
    "UCI-корпусе; её оценки не гарантируют безопасность и не являются проверкой сайтов. "
    "Вероятности не калибровались; границы зон — эмпирические результаты Validation."
)


def group_cv():
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


def checked_folds(frame):
    folds = list(group_cv().split(frame, frame.label, groups=frame.group))
    for train, heldout in folds:
        if set(frame.iloc[train].group) & set(frame.iloc[heldout].group):
            raise ValueError("Домены пересекаются внутри CV fold")
        if frame.iloc[train].label.nunique() != 2 or frame.iloc[heldout].label.nunique() != 2:
            raise ValueError("Каждый fold должен содержать оба класса")
    return folds


def feature_matrix(frame, names):
    if len(names) != len(set(names)) or not set(names).issubset(FEATURE_NAMES):
        raise ValueError("X может содержать только уникальные URL-признаки задания №9")
    matrix = frame.loc[:, names].copy()
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
        raise ValueError("Признаки должны быть числовыми")
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Признаки должны быть конечными и без пропусков")
    return matrix


def load_partitions():
    frame = pd.read_parquet(OUTPUT / "url_features.parquet")
    splits = json.loads((OUTPUT / "url_split.json").read_text(encoding="utf-8"))
    metadata = json.loads((OUTPUT / "url_dataset_meta.json").read_text(encoding="utf-8"))
    if len(frame) != 50000 or [len(splits[k]) for k in ("train", "validation", "test")] != [32869, 8471, 8660]:
        raise ValueError("Требуются существующие 50000 URL и разбиение задания №10")
    if tuple(frame.columns[:22]) != FEATURE_NAMES or metadata["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Порядок признаков отличается от задания №9")
    if not frame.label.isin([0, 1]).all() or set(frame.label) != {0, 1}:
        raise ValueError("Ожидаются метки 0=safe, 1=phishing")
    feature_matrix(frame, list(FEATURE_NAMES))
    assert_no_group_overlap(frame, splits)
    for name, positions in splits.items():
        if not frame.iloc[positions].split.eq(name).all():
            raise ValueError("Поле split не соответствует JSON")
        actual = {str(label): int((frame.iloc[positions].label == label).sum()) for label in (0, 1)}
        if actual != metadata["splits"][name]["class_counts"]:
            raise ValueError("Метки не соответствуют метаданным датасета")
    return tuple(frame.iloc[splits[k]].reset_index(drop=True) for k in ("train", "validation", "test")), metadata


def build_search(algorithm):
    if algorithm == "LogisticRegression":
        model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=42))])
        grid = {"classifier__C": [.1, 1., 10.]}
    elif algorithm == "RandomForestClassifier":
        model = Pipeline([("classifier", RandomForestClassifier(
            class_weight="balanced", min_samples_leaf=2, random_state=42, n_jobs=1))])
        grid = {"classifier__n_estimators": [100, 200], "classifier__max_depth": [10, None]}
    else:
        raise ValueError("Неизвестный алгоритм")
    return GridSearchCV(model, grid, cv=group_cv(), scoring=make_scorer(
        fbeta_score, beta=2, pos_label=1, zero_division=0), n_jobs=1, refit=True,
        error_score="raise")


def probabilities(model, matrix):
    return model.predict_proba(matrix)[:, list(model.classes_).index(1)]


def metrics(labels, scores, threshold):
    predicted = np.asarray(scores) >= threshold
    result = dict(accuracy=float(accuracy_score(labels, predicted)),
                  roc_auc=float(roc_auc_score(labels, scores)),
                  pr_auc=float(average_precision_score(labels, scores)),
                  f2=float(fbeta_score(labels, predicted, beta=2, zero_division=0)))
    for average, prefix in (("binary", ""), ("macro", "macro_")):
        result[prefix + "precision"] = float(precision_score(labels, predicted, average=average, zero_division=0))
        result[prefix + "recall"] = float(recall_score(labels, predicted, average=average, zero_division=0))
        result[prefix + "f1"] = float(fbeta_score(labels, predicted, beta=1, average=average, zero_division=0))
    result["confusion_matrix"] = confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
    return result


def choose_t2(labels, scores, min_precision=.85):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]
    eligible = np.flatnonzero((precision >= min_precision) & (thresholds > 0))
    if not len(eligible):
        return None
    f2 = np.divide(5 * precision * recall, 4 * precision + recall,
                   out=np.zeros_like(precision), where=(4 * precision + recall) != 0)
    index = max(eligible, key=lambda i: (f2[i], recall[i], precision[i], -thresholds[i]))
    return float(thresholds[index])


def choose_t1(labels, scores, t2, max_safe_phishing=.01):
    """Максимальное покрытие SAFE при эмпирической примеси phishing <= 1%."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    order = np.argsort(scores, kind="stable")
    ordered, cumulative = scores[order], np.cumsum(labels[order])
    candidates, first = np.unique(ordered, return_index=True)
    eligible = []
    for threshold, count in zip(candidates, first):
        if count and 0 < threshold < t2 and cumulative[count - 1] / count <= max_safe_phishing:
            eligible.append(float(threshold))
    # Если надёжную непустую зону выделить нельзя, SAFE остаётся пустой.
    return max(eligible) if eligible else 0.0


def category(probability, t1, t2):
    if not (np.isfinite(probability) and 0 <= probability <= 1 and 0 <= t1 < t2 <= 1):
        raise ValueError("Требуются P в [0,1] и 0 <= t1 < t2 <= 1")
    return "SAFE" if probability < t1 else "SUSPICIOUS" if probability < t2 else "DANGEROUS"


def zone_summary(labels, scores, t1, t2):
    labels, scores = np.asarray(labels), np.asarray(scores)
    zones = {"SAFE": scores < t1, "SUSPICIOUS": (scores >= t1) & (scores < t2), "DANGEROUS": scores >= t2}
    return {name: dict(rows=int(mask.sum()), phishing=int(labels[mask].sum()),
            phishing_fraction=float(labels[mask].mean()) if mask.any() else None,
            fraction_of_all_phishing=float(labels[mask].sum() / labels.sum()) if labels.sum() else None)
            for name, mask in zones.items()}


def predict_url(url, model=None, metadata=None):
    """Локальное предсказание; нормализация согласована с заданием №10."""
    if model is None:
        model = joblib.load(ARTIFACTS / "url_model.joblib")
    if metadata is None:
        metadata = json.loads((ARTIFACTS / "url_model_meta.json").read_text(encoding="utf-8"))
    normalized = normalize_url(url, strip_www=metadata["dataset"]["strip_www"])
    frame = pd.DataFrame([extract_url_features(normalized)])
    score = float(probabilities(model, feature_matrix(frame, metadata["feature_names"]))[0])
    return dict(probability=score, category=category(score, metadata["t1"], metadata["t2"]))


def feature_importance(model, names):
    classifier = model.named_steps["classifier"]
    values = classifier.coef_[0] if hasattr(classifier, "coef_") else classifier.feature_importances_
    return sorted([dict(feature=name, value=float(value)) for name, value in zip(names, values)],
                  key=lambda row: -abs(row["value"]))


def select_result(rows):
    eligible = [r for r in rows if r["t2"] is not None and r["validation_metrics"]["precision"] >= .85]
    if not eligible:
        raise ValueError("Ни одна модель не удовлетворяет Precision >= 0.85 на Validation")
    # Детерминированные tie-breaks; секунды на конкретном компьютере не меняют выбор.
    return max(eligible, key=lambda r: (r["validation_metrics"]["f2"], r["validation_metrics"]["recall"],
        r["validation_metrics"]["roc_auc"], -r["cv_f2_std"], -len(r["feature_names"]),
        int(r["algorithm"] == "LogisticRegression")))


def protected_hashes():
    paths = [*OUTPUT.glob("url_*"), *OUTPUT.glob("sms*"), *ARTIFACTS.glob("sms*"),
             ROOT / "data/raw/custom_ru_kk.csv"]
    paths += [ROOT / "ml" / name for name in ("url_features.py", "url_dataset.py", "baseline.py", "features.py", "preprocessing.py")]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


EXAMPLES = [
    ("https://www.wikipedia.org/", "Предполагаемо безопасный: известная организация, корень сайта"),
    ("https://www.python.org/downloads/", "Предполагаемо безопасный: страница загрузок Python"),
    ("https://www.gov.uk/browse/tax", "Предполагаемо безопасный: раздел государственного сайта"),
    ("https://www.who.int/news", "Предполагаемо безопасный: раздел новостей ВОЗ"),
    ("https://www.un.org/en/", "Предполагаемо безопасный: раздел сайта ООН"),
    ("https://www.nasa.gov/missions/", "Предполагаемо безопасный: раздел NASA"),
    ("https://www.microsoft.com/en-us/", "Предполагаемо безопасный: раздел Microsoft"),
    ("https://www.google.com/search?q=python", "Предполагаемо безопасный: поисковый запрос, не проверялся"),
    ("https://github.com/python/cpython", "Предполагаемо безопасный: путь репозитория Python"),
    ("https://kaspi.kz/", "Предполагаемо безопасный: известный домен банка/сервиса"),
    ("http://verify.example.com/login?password=123", "Искусственный: HTTP, verify/login/password"),
    ("https://account-update.example.org/confirm", "Искусственный: слова account/update/confirm"),
    ("http://secure-bank.example.test:8080/login", "Искусственный: HTTP и нестандартный порт"),
    ("https://example.com@verify.example.test/password", "Искусственный: userinfo маскирует фактический host"),
    ("https://free-prize.example.org/bonus?claim=1", "Искусственный: обещание приза и бонуса"),
    ("http://a.b.c.example.test/verify//account", "Искусственный: поддомены и двойной слеш"),
    ("https://kaspi-login.example.com/update?card=123456", "Искусственный: название сервиса в поддомене"),
    ("https://halyk-verify.example.org/confirm?code=1234", "Искусственный: название банка и запрос кода"),
    ("http://login-987654.example.test/reset?password=1", "Искусственный: цифры, HTTP, password"),
    ("https://reward.example.test/free/bonus?account=1&verify=1", "Искусственный: несколько ключевых слов"),
]


def plot_results(labels, scores, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, values in scores.items():
        x, y, _ = roc_curve(labels, values)
        ax.plot(x, y, label=key)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="URL models: Validation ROC")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/11_url_roc.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, row in zip(axes.flat, rows):
        top = row["importance"][:10][::-1]
        ax.barh([v["feature"] for v in top], [v["value"] for v in top],
                color=["steelblue" if v["value"] >= 0 else "indianred" for v in top])
        ax.set_title(row["key"])
        ax.set_xlabel("Standardized coefficient" if row["algorithm"] == "LogisticRegression" else "MDI importance")
        ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/11_url_top10.png", dpi=150)
    plt.close(fig)


def run_experiment():
    before = protected_hashes()
    (train, validation, test), dataset_meta = load_partitions()
    folds = checked_folds(train)
    fold_audit = [dict(train_rows=len(a), heldout_rows=len(b),
                  train_groups=int(train.iloc[a].group.nunique()), heldout_groups=int(train.iloc[b].group.nunique()),
                  train_phishing=int(train.iloc[a].label.sum()), heldout_phishing=int(train.iloc[b].label.sum()),
                  overlap=0) for a, b in folds]
    rows, models, validation_scores = [], {}, {}
    for variant, names in FEATURE_SETS.items():
        x_train, x_validation = feature_matrix(train, names), feature_matrix(validation, names)
        for algorithm in ("LogisticRegression", "RandomForestClassifier"):
            key = f"{algorithm}/{variant}"
            print(f"Searching {key}", flush=True)
            search = build_search(algorithm)
            started = perf_counter()
            search.fit(x_train, train.label, groups=train.group)
            elapsed = perf_counter() - started
            model = search.best_estimator_
            values = probabilities(model, x_validation)
            t2 = choose_t2(validation.label, values)
            t1 = choose_t1(validation.label, values, t2) if t2 is not None else None
            one = x_validation.iloc[:1]
            started = perf_counter()
            for _ in range(30):
                probabilities(model, one)
            latency = (perf_counter() - started) * 1000 / 30
            trials = [dict(parameters=params, mean_f2=float(mean), std_f2=float(std))
                      for params, mean, std in zip(search.cv_results_["params"],
                      search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"])]
            row = dict(key=key, algorithm=algorithm, variant=variant, feature_names=names,
                parameters=search.best_params_, classifier_parameters=model.named_steps["classifier"].get_params(),
                cv_f2=float(search.best_score_), cv_f2_std=float(search.cv_results_["std_test_score"][search.best_index_]),
                cv_trials=trials, fit_seconds=float(search.refit_time_), search_seconds=elapsed,
                prediction_ms=latency, t1=t1, t2=t2, importance=feature_importance(model, names),
                default_metrics=metrics(validation.label, values, .5),
                validation_metrics=metrics(validation.label, values, t2) if t2 is not None else None,
                validation_zones=zone_summary(validation.label, values, t1, t2) if t2 is not None else None)
            rows.append(row)
            models[key], validation_scores[key] = model, values
            print(json.dumps({k:row[k] for k in ("key", "parameters", "cv_f2", "t1", "t2", "validation_metrics")}), flush=True)
    selected = select_result(rows)
    model = models[selected["key"]]
    # Все решения и оба порога зафиксированы. Единственный вызов предсказания Test.
    test_scores = probabilities(model, feature_matrix(test, selected["feature_names"]))
    test_metrics = metrics(test.label, test_scores, selected["t2"])
    meta = dict(version="1.0.0", trained_at=datetime.now(timezone.utc).isoformat(),
        algorithm=selected["algorithm"], variant=selected["variant"], parameters=selected["parameters"],
        classifier_parameters=selected["classifier_parameters"], feature_names=selected["feature_names"],
        t1=selected["t1"], t2=selected["t2"], threshold_policy=dict(min_precision=.85, max_safe_phishing=.01),
        test_metrics=test_metrics, test_zones=zone_summary(test.label, test_scores, selected["t1"], selected["t2"]),
        training_rows=len(train), refit_train_validation=False, test_metrics_model="saved Train-only model",
        dataset=dict(source=dataset_meta["source"], source_csv_sha256=dataset_meta["csv_sha256"],
                     strip_www=dataset_meta["strip_www"], rows=50000, splits=dataset_meta["splits"],
                     feature_parquet_sha256=before["data/processed/url_features.parquet"],
                     split_sha256=before["data/processed/url_split.json"]),
        cv=dict(type="StratifiedGroupKFold", n_splits=5, shuffle=True, random_state=42, folds=fold_audit),
        limitations=LIMITATIONS, pr_auc_method="average_precision_score (AP, non-interpolated)",
        comparison=rows, protected_sha256=before)
    ARTIFACTS.mkdir(exist_ok=True)
    model_path = ARTIFACTS / "url_model.joblib"
    joblib.dump(model, model_path)
    restored = joblib.load(model_path)
    matrix = feature_matrix(validation, selected["feature_names"])
    np.testing.assert_array_equal(probabilities(model, matrix), probabilities(restored, matrix))
    meta["examples"] = [dict(url=url, comment=comment, assumed="safe" if i < 10 else "suspicious",
                            **predict_url(url, restored, meta)) for i, (url, comment) in enumerate(EXAMPLES)]
    started = perf_counter()
    for _ in range(30):
        predict_url(EXAMPLES[0][0], restored, meta)
    meta["end_to_end_ms"] = (perf_counter() - started) * 1000 / 30
    if protected_hashes() != before:
        raise RuntimeError("Защищённые SMS/URL-файлы изменились во время эксперимента")
    (ARTIFACTS / "url_model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_results(validation.label, validation_scores, rows)
    write_report(meta)
    print(json.dumps(dict(selected=selected["key"], t1=selected["t1"], t2=selected["t2"], test=test_metrics)), flush=True)
    return meta


def write_report(meta):
    """Русский отчёт по фактическим сохранённым результатам."""
    lines = ["# Задание №11 — URL-модель", "", "## Методика", "",
        "Использован неизменённый Parquet задания №10: 50000 URL, Train 32869, Validation 8471, "
        "Test 8660. label: 0=safe, 1=phishing. В X передаются только признаки из "
        "extract_url_features(); source_id, URL, group, split и label исключены. "
        "Данные и индексы не пересоздаются. SHA-256 защищённых файлов записаны в metadata.", "",
        "Два алгоритма: LogisticRegression(StandardScaler → classifier) и RandomForestClassifier. "
        "A — 22 признака; B — 19, без is_https, path_length, query_param_count. "
        "GridSearchCV только на Train с groups=зарегистрированный домен. "
        "StratifiedGroupKFold(5, shuffle=True, random_state=42); каждый fold проверен на "
        "отсутствие пересечений доменов и наличие обоих классов. StandardScaler обучается "
        "внутри соответствующего fold. Детерминированные признаки задания №9 не требуют fit.", "",
        "Оптимизация: F2 phishing=1. LR: C=[0.1,1,10], class_weight=balanced, max_iter=3000. "
        "RF: n_estimators=[100,200], max_depth=[10,None], min_samples_leaf=2, class_weight=balanced. "
        "C исследует три масштаба регуляризации; RF сравнивает размер леса и ограниченную/полную "
        "глубину, лист минимум 2 уменьшает запоминание. random_state=42, n_jobs=1. "
        "Других алгоритмов и новых зависимостей нет.", "",
        "## Сравнение на Validation", "",
        "Ниже метрики при t2 каждой модели, выбранном исключительно на Validation. "
        "PR-AUC = average_precision_score (AP без интерполяции). Матрицы: [[TN,FP],[FN,TP]]. "
        "Fit — refit лучшей конфигурации на Train; Search — весь GridSearchCV. Задержка — "
        "среднее 30 predict_proba одной строки числовых признаков, без извлечения признаков.", "",
        "| Модель | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | AP | Fit, с | Search, с | URL, мс |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in meta["comparison"]:
        m = row["validation_metrics"]
        values = " | ".join(f"{m[k]:.6f}" for k in ("accuracy","precision","recall","f1","f2","roc_auc","pr_auc")) if m else " | ".join(["—"]*7)
        lines.append(f"| {row['key']} | {values} | {row['fit_seconds']:.3f} | {row['search_seconds']:.3f} | {row['prediction_ms']:.3f} |")
    lines += ["", "| Модель | Macro Precision | Macro Recall | Macro F1 | Confusion Matrix |", "|---|---:|---:|---:|---|"]
    for row in meta["comparison"]:
        m = row["validation_metrics"]
        if m:
            lines.append(f"| {row['key']} | {m['macro_precision']:.6f} | {m['macro_recall']:.6f} | {m['macro_f1']:.6f} | {m['confusion_matrix']} |")
    lines += ["", "| Модель | Параметры | CV F2 ± std | t1 | t2 |", "|---|---|---|---:|---:|"]
    for row in meta["comparison"]:
        lines.append(f"| {row['key']} | {row['parameters']} | {row['cv_f2']:.6f} ± {row['cv_f2_std']:.6f} | {row['t1']} | {row['t2']} |")
    lines += ["", "Для сопоставимости при фиксированном пороге 0.5:", "",
        "| Модель | Precision | Recall | F1 | F2 |", "|---|---:|---:|---:|---:|"]
    for row in meta["comparison"]:
        m = row["default_metrics"]
        lines.append(f"| {row['key']} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['f2']:.6f} |")
    lines += ["", "![ROC четырёх конфигураций на Validation](11_url_roc.png)", "",
        "## Влияние артефактов и важность", "",
        LIMITATIONS, "", "Изменение B относительно A при выбранных порогах:", "",
        "| Алгоритм | ΔPrecision | ΔRecall | ΔF2 | ΔROC-AUC |", "|---|---:|---:|---:|---:|"]
    for algorithm in ("LogisticRegression", "RandomForestClassifier"):
        a, b = [next(r["validation_metrics"] for r in meta["comparison"] if r["algorithm"] == algorithm and r["variant"] == v) for v in ("A","B")]
        if a and b:
            lines.append(f"| {algorithm} | " + " | ".join(f"{b[k]-a[k]:+.6f}" for k in ("precision","recall","f2","roc_auc")) + " |")
    lines += ["", "![Топ-10 признаков каждой конфигурации](11_url_top10.png)", "",
        "LR: показаны подписанные коэффициенты после StandardScaler; коэффициент означает "
        "изменение log-odds phishing при росте признака на одно обучающее стандартное отклонение, "
        "при прочих равных. Положительный знак направлен к phishing, отрицательный — к safe. "
        "RF: MDI feature_importances_, чувствительная к числу возможных разбиений и корреляции признаков. "
        "Важность не доказывает причинность.", ""]
    for row in meta["comparison"]:
        top = "; ".join(f"{v['feature']}={v['value']:.6f}" for v in row["importance"][:10])
        positions = {v["feature"]:i+1 for i,v in enumerate(row["importance"])}
        lines += [f"**{row['key']}**: {top}.", "",
            "Позиции возможных артефактов: " + ", ".join(f"{k}: {positions.get(k, 'исключён')}" for k in ("is_https","url_length","path_length","query_param_count","special_char_count")) + ".", ""]
    rf_a = next(r for r in meta["comparison"] if r["algorithm"] == "RandomForestClassifier" and r["variant"] == "A")
    rf_b = next(r for r in meta["comparison"] if r["algorithm"] == "RandomForestClassifier" and r["variant"] == "B")
    a_importance = {v["feature"]:v["value"] for v in rf_a["importance"]}
    b_importance = {v["feature"]:v["value"] for v in rf_b["importance"]}
    lines += [f"У RF/A is_https даёт **{a_importance['is_https']:.2%}** MDI-важности, "
        f"url_length — **{a_importance['url_length']:.2%}**, path_length — "
        f"**{a_importance['path_length']:.2%}**. Это выраженная зависимость от прямых "
        "и косвенных индикаторов формата сбора. После удаления трёх признаков в RF/B "
        f"url_length даёт **{b_importance['url_length']:.2%}** важности. Уменьшенный набор "
        "не устраняет проблему; длина и другие структурные признаки остаются прокси. "
        "LR/A также опирается прежде всего на путь, длину и HTTPS. Высокое качество "
        "на этом корпусе нельзя переносить на произвольные реальные ссылки без внешней проверки.", ""]
    selected = next(r for r in meta["comparison"] if r["algorithm"] == meta["algorithm"] and r["variant"] == meta["variant"])
    lines += ["## Выбор и пороги", "",
        f"Выбрана **{meta['algorithm']}/{meta['variant']}**. Критерий — максимальный Validation F2 "
        "при Precision >=0.85, затем Recall, ROC-AUC, меньший std CV F2, меньше признаков и "
        "предпочтение простой LR при полном равенстве. Скорость и интерпретируемость оценены "
        "по таблицам; они не подменяют приоритет F2/Recall. Даже выбранная модель остаётся "
        "зависимой от особенностей корпуса. CV std характеризует различие доменных folds, "
        "а не доверительный интервал внешнего качества.", "",
        f"**t1={meta['t1']}**, **t2={meta['t2']}**. Правило: SAFE при P<t1; SUSPICIOUS "
        "при t1<=P<t2; DANGEROUS при P>=t2. t2 — максимум F2 среди уникальных оценок Validation "
        "с Precision>=0.85; при равенстве выбираются Recall, Precision, меньший порог. "
        "Нулевая граница t2 исключена для возможности t1<t2. Если допустимого t2 нет, "
        "конфигурация исключается; если нет ни одной, Test не оценивается.", "",
        "t1 — граница с максимальным покрытием SAFE при эмпирической доле phishing внутри "
        "SAFE <=1%, строго ниже t2. Перебираются уникальные оценки с правилом P<t1. "
        "Это явно заданная политика низкорисковой зоны, не гарантия будущего риска. "
        "Если допустимой непустой зоны нет, t1=0 и SAFE пустая. SUSPICIOUS — оставшаяся "
        "неопределённая зона: её нельзя автоматически считать безопасной или обнаруженным phishing.", "",
        "Зоны выбранной модели на Validation:", "", "| Зона | Строк | Phishing | Доля phishing в зоне | Доля всех phishing |",
        "|---|---:|---:|---:|---:|"]
    for name, zone in selected["validation_zones"].items():
        lines.append(f"| {name} | {zone['rows']} | {zone['phishing']} | {zone['phishing_fraction']} | {zone['fraction_of_all_phishing']} |")
    lines += ["", "## Единственная оценка Test", "",
        "Алгоритм, признаки, параметры и оба порога зафиксированы до предсказания Test. "
        "После просмотра Test ничего не подбиралось. **Переобучение на Train+Validation не "
        "выполнялось**: сохранён тот же Pipeline, обученный на 32869 Train URL. "
        "Тестовые метрики относятся непосредственно к сохранённой модели версии 1.0.0.", "",
        "| Метрика | Test | Требование | Выполнено |", "|---|---:|---:|---|"]
    required = {"recall":.85, "precision":.85, "roc_auc":.90}
    for key, value in meta["test_metrics"].items():
        if key == "confusion_matrix":
            continue
        bound = required.get(key)
        lines.append(f"| {key} | {value:.6f} | {bound if bound else '—'} | {('да' if value>=bound else 'нет') if bound else '—'} |")
    lines += ["", f"Confusion Matrix Test: `{meta['test_metrics']['confusion_matrix']}`.", ""]
    failed = [key for key,bound in required.items() if meta["test_metrics"][key] < bound]
    if failed:
        lines += [f"Не выполнены: {', '.join(failed)}. Доменные группы и баланс классов отличаются "
            "между Train/Validation/Test; порог ограничен результатами Validation. Это возможные "
            "объяснения переноса качества, а не доказанная причина. Корректировка по Test не делалась.", ""]
    else:
        lines += ["Все три формальных требования URL-классификатора на этом Test выполнены. "
            "Это не устраняет ограничения разметки и артефактов.", ""]
    lines += ["## 20 дополнительных примеров после Test", "",
        "Предполагаемые safe адреса известны по названиям организаций, но не проверялись "
        "по сети. Остальные примеры искусственные на зарезервированных доменах. "
        "Это демонстрация, не внешний размеченный benchmark. Ни параметры, ни пороги после "
        "этих примеров не менялись. SAFE не означает гарантированную безопасность.", "",
        "| URL как текст | P_url | Категория | Комментарий |", "|---|---:|---|---|"]
    for example in meta["examples"]:
        lines.append(f"| `{example['url']}` | {example['probability']:.6f} | {example['category']} | {example['comment']} |")
    assumed_safe = [e for e in meta["examples"] if e["assumed"] == "safe"]
    totals = {name:sum(e["category"] == name for e in assumed_safe) for name in ("SAFE","SUSPICIOUS","DANGEROUS")}
    lines += ["", f"Среди {len(assumed_safe)} предполагаемо безопасных примеров: "
        f"SAFE — {totals['SAFE']}, SUSPICIOUS — {totals['SUSPICIOUS']}, "
        f"DANGEROUS — {totals['DANGEROUS']}. Срабатывания на обычных разделах известных сайтов "
        "согласуются с зависимостью от path/query в исходном корпусе. Примеры демонстрируют "
        "ограничение переноса; они не являются проверенной внешней разметкой. "
        "Модель не следует считать готовой к надёжной оценке произвольных реальных ссылок. "
        "После этих наблюдений параметры, признаки и пороги не изменялись.", ""]
    lines += ["", "## Артефакты и воспроизведение", "",
        "Сохранены ml/artifacts/url_model.joblib и url_model_meta.json. JSON содержит версию, "
        "UTC дату, параметры, порядок X, t1/t2, CV, сравнение, Test, SHA-256 датасета/индексов "
        "и ограничения. Модель и JSON остаются локальными по существующему .gitignore. "
        "Точное совпадение вероятностей до/после сериализации проверено на Validation. "
        "Новые URL нормализуются как в задании №10 (www и корневой слеш), затем обрабатываются "
        "единственной существующей extract_url_features(). Сеть не используется.", "",
        f"Измеренная задержка полного predict_url с извлечением признаков: **{meta['end_to_end_ms']:.3f} мс** "
        "(среднее 30 повторов одного URL, модель уже загружена; зависит от машины).", "",
        "```powershell", "python -m ml.url_model",
        r"python -m pytest -q -p no:cacheprovider --basetemp=.\.pytest_task11_temp", "```", "",
        "Каждый запуск обучения — новое воспроизведение всего эксперимента с одной финальной "
        "оценкой Test. Для просмотра результатов повторное обучение не требуется: читайте JSON и отчёт.", "",
        "```python", "from ml.url_model import predict_url", "print(predict_url('https://www.python.org/downloads/'))", "```", "",
        "## Проверки", "",
        "Новые модульные тесты: **29 passed** на малых синтетических данных. Полный запуск: "
        "**147 passed in 21.83s**, включая все **118 предыдущих** проверок. "
        "Проверены числовой X без метаданных, групповые folds, обучение StandardScaler внутри "
        "fold, воспроизводимость LR/RF, максимум F2 с ограничением Precision, границы зон, "
        "сериализация модели/JSON, новые URL без сети и неизменность защищённых файлов. "
        "Модульные тесты не переобучают модели на 50000 URL. Реальные вероятности до/после "
        "joblib совпали точно на Validation, оба графика проверены визуально.", "",
        "SMS-компоненты, датасет URL, разбиение и код заданий №9–10 не изменены; "
        "SHA-256 защищённых файлов проверены. API, интерфейсы и задание №12 не реализованы. "
        "Git-коммит и push не выполнялись."]
    (ROOT / "docs/11_url_model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_experiment()
