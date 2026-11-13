"""Build the task 08 report and notebook from recorded experiment results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import nbformat as nb
    meta = json.loads((ROOT / "ml/artifacts/sms_model_meta.json").read_text(encoding="utf-8"))
    lines = ["# Сравнение алгоритмов классификации SMS — задание №8", "",
             "## Методика", "",
             "Использовано прежнее разбиение sms_split.json: Train 3594 (433 spam), "
             "Validation 770 (92 spam), Test 771 (93 spam). Новое разбиение не создавалось. "
             "Пересечения индексов и дубликаты нормализованных текстов проверяются при загрузке.", "",
             "GridSearchCV работает исключительно на Train: StratifiedKFold(5, shuffle=True, random_state=42), "
             "целевая метрика F2 для spam=1. Для каждого алгоритма проверены четыре конфигурации: "
             "слова / слова + char_wb(3,5), с удалением английских стоп-слов и без удаления. "
             "Словесные N-граммы (1,2), min_df=2, max_df=0.95. Шесть ручных признаков одинаковы "
             "во всех конфигурациях. Удаление стоп-слов затрагивает только словесный TF-IDF; "
             "символьная и ручная ветви получают прежний нормализованный текст.", "",
             "Сетки: NB alpha=[0.1,0.5,1.0] — слабое/умеренное сглаживание; "
             "LR и SVC C=[0.1,1,10] — три порядка силы регуляризации. "
             "Для LR и SVC class_weight=balanced, random_state=42. LR: max_iter=3000; "
             "SVC: dual=False, max_iter=20000. В предварительном запуске dual='auto' дал "
             "предупреждения о несходимости; этот запуск остановлен до Test, затем сравнение "
             "повторено с решением прямой задачи. Все TF-IDF обучаются внутри folds. "
             "В LinearSVC CalibratedClassifierCV(method='sigmoid') охватывает полный Pipeline, "
             "поэтому даже внутренние folds калибровки имеют независимые словари и IDF. "
             "Внутренняя калибровка использует такой же пятиfoldовый стратифицированный splitter. "
             "Validation, Test и ru/kk не используются ни для GridSearchCV, ни для калибровки.", "",
             "## Сравнение на Validation", "",
             "Настройки каждого алгоритма выбираются по среднему CV F2 на Train. "
             "Основная таблица использует отдельный порог каждого алгоритма, выбранный на Validation. "
             "PR-AUC обозначает average_precision_score: AP без интерполяции, сумма приращений Recall, "
             "взвешенных Precision; это не трапецеидальная площадь.", "",
             "| Алгоритм | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC (AP) | Fit, с | Поиск, с | 1 SMS, мс |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    keys = ("accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc")
    for row in meta["comparison"]:
        m = row["validation_metrics"]
        values = " | ".join(f"{m[k]:.6f}" for k in keys) if m else " | ".join(["—"] * 7)
        lines.append(f"| {row['algorithm']} | {values} | {row['training_seconds']:.3f} | {row['search_seconds']:.3f} | {row['prediction_ms']:.3f} |")
    lines += ["", "Fit — время refit лучшей конфигурации на Train; поиск включает все CV fits и refit. "
              "Задержка — среднее 30 последовательных predict_proba для одного и того же SMS Validation, "
              "включая нормализацию и признаки. Это локальный замер, зависящий от оборудования и нагрузки.", "",
              "| Алгоритм | Char | Стоп-слова | Параметр | CV F2 | Порог |", "|---|---|---|---|---:|---:|"]
    for r in meta["comparison"]:
        c = r["configuration"]
        lines.append(f"| {r['algorithm']} | {c['include_chars']} | {c['stop_words']} | {c['parameter']} | {c['cv_f2']:.6f} | {r['threshold']} |")
    lines += ["", "Для сопоставления при стандартном пороге 0.5:", "",
              "| Алгоритм | Precision | Recall | F1 | F2 |", "|---|---:|---:|---:|---:|"]
    for r in meta["comparison"]:
        m = r["default_metrics"]
        lines.append(f"| {r['algorithm']} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['f2']:.6f} |")
    lines += ["", "Сравнение признаков: лучший средний CV F2 на Train по трём значениям параметра "
              "внутри каждой конфигурации. Все 36 результатов доступны в metadata и notebook.", "",
              "| Char | Стоп-слова | NB | LR | SVC |", "|---|---|---:|---:|---:|"]
    for chars in (False, True):
        for stop in (None, "english"):
            scores = [max(e["cv_f2"] for e in meta["experiments"]
                          if e["algorithm"] == name and e["include_chars"] == chars
                          and e["stop_words"] == stop)
                      for name in ("MultinomialNB", "LogisticRegression", "LinearSVC")]
            lines.append(f"| {chars} | {stop} | " + " | ".join(f"{s:.6f}" for s in scores) + " |")
    lines += ["", "![ROC на Validation](08_roc.png)", "", "![PR на Validation](08_pr.png)", "",
              "## Порог и итоговый выбор", "",
              "Проверены уникальные вероятности Validation с правилом probability >= threshold. "
              "Из допустимых точек Precision >= 0.85 выбирается максимальный Recall, "
              "затем наименьший порог. Пустые положительные прогнозы не считаются допустимыми; "
              "если допустимых точек нет, возвращается None и алгоритм исключается из выбора. "
              "При отсутствии допустимых моделей эксперимент останавливается до Test.", "",
              f"Выбран **{meta['algorithm']}**, порог **{meta['threshold']}**. "
              "Среди допустимых моделей применяется порядок: F2, затем Recall, затем меньшая задержка. "
              "Скорость служит критерием при равных основных метриках. NB и LR допускают прямой анализ "
              "вкладов признаков; калиброванный ансамбль SVC сложнее интерпретировать. "
              "Интерпретируемость учитывается качественно и не заменяет приоритет F2/Recall.", "",
              "## Однократная оценка Test", "",
              "Алгоритм, конфигурация и порог зафиксированы до единственного вычисления вероятностей Test. "
              "Порог после оценки не менялся. **Все тестовые метрики ниже получены до финального "
              "переобучения на Train + Validation.** Сохранённый Pipeline обучен на 4364 SMS, "
              "к нему эти значения нельзя приписывать как результаты повторной оценки.", "",
              "| Метрика | Test | Требование | Выполнено |", "|---|---:|---:|---|"]
    for key in keys:
        value = meta["test_metrics"][key]
        required = key in ("precision", "recall", "f1")
        lines.append(f"| {key} | {value:.6f} | {'≥ 0.90' if required else '—'} | {('да' if value >= .9 else 'нет') if required else '—'} |")
    lines += ["", f"Confusion Matrix Test [[TN, FP], [FN, TP]]: `{meta['test_metrics']['confusion_matrix']}`."]
    failed = [k for k in ("precision", "recall", "f1") if meta["test_metrics"][k] < .9]
    if failed:
        lines += ["", f"Не выполнены: {', '.join(failed)}. Порог оптимизировался на Validation при "
                  "минимальном Precision 0.85, что не гарантирует Precision 0.90 на независимом Test. "
                  "Различия состава небольших выборок и перенос порога могут приводить к ухудшению. "
                  "Это объяснение методических ограничений, а не доказательство конкретной причины ошибок; "
                  "порог по Test не исправлялся."]
    lines += ["", "## Отдельная проверка ru/kk", "",
              "После финального refit загружен сохранённый артефакт. Использован прежний порог. "
              "Ни одно из 150 сообщений не участвует в обучении или настройке.", "",
              "| Язык | SMS | Precision | Recall | F1 | Confusion Matrix [[TN,FP],[FN,TP]] |",
              "|---|---:|---:|---:|---:|---|"]
    for lang, count in (("ru", 100), ("kk", 50)):
        m = meta["language_metrics"][lang]
        lines.append(f"| {lang} | {count} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['confusion_matrix']} |")
    lines += ["", "Модель обучена на английском UCI-корпусе. Набор ru/kk искусственный и небольшой; "
              "результаты не подтверждают полноценную поддержку языков. Английские словари и правила "
              "не покрывают русскую и казахскую лексику. Spam в UCI не обязательно мошенничество. "
              "Validation используется и для порога, и для выбора алгоритма, поэтому её метрики оптимистичны. "
              "Переобучение может изменить распределение вероятностей, хотя зафиксированный порог сохраняется.", "",
              "## Воспроизведение и артефакты", "",
              "Из корня проекта в активированном окружении:", "", "```powershell",
              "python -m ml.compare_sms", "python scripts/report_task08.py",
              r"python -m pytest -q -p no:cacheprovider --basetemp=.\.pytest_task08_temp", "```", "",
              "При отсутствии данных восстановите их по docs/06_feature_extraction.md. "
              "Модуль сравнения читает существующий JSON, не создавая разбиение. "
              "Каждый новый запуск — отдельное воспроизведение эксперимента, перезаписывающее финальные артефакты.", "",
              "- ml/compare_sms.py — обучение, сравнение, выбор, оценка и сохранение.",
              "- ml/features.py — необязательные параметры с сохранением прежних defaults.",
              "- ml/artifacts/sms_model.joblib — полный финальный Pipeline/калибратор.",
              "- ml/artifacts/sms_model_meta.json — версия 1.0.0, UTC-дата обучения, параметры, "
              "порог, все 36 CV результатов, Validation/Test и ru/kk.",
              "- notebooks/03_sms_model_comparison.ipynb — воспроизведение, таблицы и графики.",
              "- docs/08_roc.png, docs/08_pr.png — общие кривые трёх моделей.",
              "- tests/test_sms_comparison.py — проверки разбиения, порога, признаков, калибровки и артефактов.", "",
              "Артефакты моделей исключены существующим .gitignore и остаются локальными. "
              "Baseline и predict_sms() не изменялись. Новых зависимостей нет. Коммит и push не выполнялись."]
    lines += ["", "Для использования финальной модели применяйте сохранённый порог к вероятности spam. "
              "Обычный model.predict() использует стандартное правило классификатора и не подставляет "
              "порог из JSON автоматически:", "", "```python", "import json, joblib",
              "from ml.compare_sms import ARTIFACTS, spam_probabilities",
              "model = joblib.load(ARTIFACTS / 'sms_model.joblib')",
              "meta = json.loads((ARTIFACTS / 'sms_model_meta.json').read_text(encoding='utf-8'))",
              "probability = float(spam_probabilities(model, ['Your input SMS'])[0])",
              "label = int(probability >= meta['threshold'])", "```"]
    lines += ["", "## Проверки", "",
              "Тесты покрывают разбиение, независимость TF-IDF во внутренних folds калибровки, "
              "неизменность словарей/IDF при transform, выбор порога полным перебором и загрузку артефакта. "
              "Для текущей поставки полный pytest: 34 passed; тесты задания №7 проходят. "
              "Notebook выполнен без повторного обучения. SHA-256 baseline и разбиения не изменились.", "",
              "CV и параметры воспроизводимы при прежних данных и зависимостях. При точном равенстве "
              "F2 и Recall итоговый выбор учитывает измеренную задержку: другое оборудование может "
              "изменить выбор между равными по этим метрикам алгоритмами. Выбор этого запуска записан в JSON."]
    (ROOT / "docs/08_sms_model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    book = nb.v4.new_notebook()
    book.cells = [nb.v4.new_markdown_cell("# Сравнение алгоритмов SMS — задание №8\n\n"
        "По умолчанию показаны сохранённые результаты. RUN_EXPERIMENT=True полностью повторяет эксперимент "
        "и перезаписывает артефакты. Test используется один раз после выбора. Ручные признаки одинаковы "
        "во всех вариантах. Подробнее: docs/08_sms_model_comparison.md."),
        nb.v4.new_code_cell("from pathlib import Path\nimport os, sys, json\n"
        "root = Path.cwd()\nif not (root / 'ml').is_dir():\n    root = root.parent\n"
        "os.chdir(root)\nif str(root) not in sys.path:\n    sys.path.insert(0, str(root))\n"
        "from ml.compare_sms import run_experiment, ARTIFACTS\nRUN_EXPERIMENT = False\n"
        "meta = run_experiment() if RUN_EXPERIMENT else json.loads((ARTIFACTS / 'sms_model_meta.json').read_text(encoding='utf-8'))"),
        nb.v4.new_code_cell("import pandas as pd\nrows=[]\nfor r in meta['comparison']:\n"
        "    rows.append(dict(algorithm=r['algorithm'], **r['validation_metrics'], threshold=r['threshold'], "
        "training_seconds=r['training_seconds'], prediction_ms=r['prediction_ms']))\n"
        "pd.DataFrame(rows).drop(columns='confusion_matrix')"),
        nb.v4.new_code_cell("pd.DataFrame(meta['experiments'])"),
        nb.v4.new_code_cell("from IPython.display import display, Image\nfor kind in ('roc', 'pr'):\n"
        "    display(Image(filename=str(root / 'docs' / f'08_{kind}.png')))"),
        nb.v4.new_code_cell("display(meta['algorithm'], meta['threshold'], meta['test_metrics'], meta['language_metrics'])"),
        nb.v4.new_markdown_cell("Тестовые метрики получены **до** переобучения на Train + Validation. "
        "PR-AUC = average_precision_score (AP). Искусственные ru/kk не участвуют в обучении и настройке. "
        "Модель обучена на английском UCI; полноценная поддержка ru/kk не подтверждена.")]
    book.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nb.write(book, ROOT / "notebooks/03_sms_model_comparison.ipynb")


if __name__ == "__main__":
    main()
