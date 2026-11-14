# SMS Guard

Учебная Django-система анализа SMS и структуры URL. Исполнители — Арыстанов Мади и Тураров Алишер, SIS-2305, Международный университет информационных технологий.

Реализованы задания №1–17, материалы комплексной проверки №18 находятся в `docs/18_final_testing.md`. Проект использует сохранённые scikit-learn Pipeline, правила социальной инженерии и Risk Score из YAML. URL разбираются локально, сайты не открываются.

## Установка на Windows

Нужны 64-битный Python 3.11–3.13 и Git. Проверка чистой установки выполнена на Python 3.12.14. Для загрузки зависимостей требуется интернет. Команды выполняются из каталога с manage.py.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python scripts/init_env.py
python manage.py migrate
python manage.py check
```

При WinError 5 в стандартном pip wheel cache задайте отдельный доступный каталог.

```powershell
$installCache = Join-Path $env:TEMP ('sms_guard_pip_' + [guid]::NewGuid().ToString('N'))
python -m pip install -r requirements.txt --cache-dir $installCache
```

init_env.py создаёт локальный `.env` с индивидуальным SECRET_KEY и сохраняет существующий ключ при повторном запуске. Файл исключён из Git. Параметры описаны в `.env.example`. Для локального HTTP используются DJANGO_COOKIE_SECURE=False и dev-режим DEBUG=True. Для демонстрационной конфигурации задайте DEBUG=False, для реального HTTPS включите Secure-cookie. Реальное HTTPS-развёртывание этим проектом не проверено.

SQLite выбран для локального учебного запуска без отдельного сервера БД. Миграции и тесты проверены на SQLite, серверная производительность на 1000 записях соответствует измеренным ограничениям. PostgreSQL-настройки предусмотрены, но отдельное PostgreSQL-развёртывание и драйвер не проверялись.

## Существующие модели и данные

Git хранит код, документы и небольшой `data/raw/custom_ru_kk.csv`. Большие данные, производные CSV/Parquet, split JSON и joblib/metadata исключены из Git. Одного клонирования недостаточно для анализа SMS/URL.

Для запуска приложения перенесите подготовленный доверенный bundle предыдущих этапов в `ml/artifacts/`.

- sms_model.joblib
- sms_model_meta.json
- url_model.joblib
- url_model_meta.json

Пример переноса из имеющейся локальной папки артефактов

```powershell
$artifactSource = Read-Host 'Папка с существующими артефактами SMS Guard'
foreach ($name in @('sms_model.joblib','sms_model_meta.json','url_model.joblib','url_model_meta.json')) {
    Copy-Item -LiteralPath (Join-Path $artifactSource $name) -Destination (Join-Path 'ml/artifacts' $name)
}
python manage.py sync_model_versions
```

Без bundle приложение запускается, health сообщает отсутствующие файлы, анализ возвращает 503. После переноса файлов перезапустите сервер, поскольку ошибка загрузки кешируется. В проверке №18 обучение повторно не запускается.

Для полной проверки ML нужны также исходные и обработанные наборы со split из предыдущих этапов. Автотест CLI baseline дополнительно использует прежний `sms_baseline.joblib`, перенесите и его для полного pytest. Используйте сохранённую локальную копию, сверяя SHA-256 с метаданными. При отсутствии SMS-данных их можно восстановить в новой рабочей копии по docs/06_feature_extraction.md.

```powershell
python ml/data/load_sms.py
python -m ml.preprocessing
python -m ml.features
```

Подготовка URL-данных из официального имеющегося CSV описана в docs/10_url_dataset.md, команда `python -m ml.url_dataset --strip-www`. Она создаёт данные, split и отчёт, поэтому выполняйте её в отдельной копии. Для проверки зафиксированной модели используйте исходный сохранённый split, а не новый произвольный набор. Сами модели для запуска переобучать не требуется.

## Запуск и страницы

```powershell
python manage.py runserver 127.0.0.1:8000
```

Для локальной демонстрации с DEBUG=False стандартный runserver требует явного обслуживания static. Параметр --insecure ниже применяется только для локального показа на loopback.

```powershell
$env:DJANGO_DEBUG='False'
$env:DJANGO_COOKIE_SECURE='False'
python manage.py runserver 127.0.0.1:8000 --insecure
```

| Страница | Назначение |
|---|---|
| /login/, /register/ | Вход и регистрация |
| / | Dashboard с периодами 7/30/90 дней |
| /check/sms/ | Анализ SMS до 1000 символов и до трёх URL |
| /check/url/ | Отдельный URL до 2048 символов |
| /history/ | Своя история, фильтры, 20 записей на страницу |
| /checks/id/ | Собственный сохранённый результат |
| /profile/ | Язык, согласие, смена пароля, удаление истории |
| /api/v1/health/ | Статус приложения и наличие артефактов |

Все основные страницы требуют входа. Администратор видит общие агрегаты и версии моделей, подробности чужих проверок остаются недоступными. Исходный SMS сохраняется только при согласии, по умолчанию оно выключено. В истории хранятся безопасные части с масками телефонов, карт, сумм и обнаруженных URL.

## Демонстрационные данные и показ

Пароль пользователя sms_guard_demo задайте самостоятельно, без сохранения в репозитории.

```powershell
$demoSecret = Read-Host 'Пароль демопользователя' -AsSecureString
$env:SMS_GUARD_DEMO_PASSWORD = [System.Net.NetworkCredential]::new('', $demoSecret).Password
python manage.py seed_demo
Remove-Item Env:SMS_GUARD_DEMO_PASSWORD
```

seed_demo создаёт 30 явно искусственных результатов без запуска ML и не предназначен для подтверждения качества. Реальный показ на новом вводе описан в docs/18_demo_scenarios.md. Сценарии A–E используют сохранённые модели и показывают в том числе ошибки, затем историю и Dashboard. Там же приведён план видеозаписи на 4 минуты 30 секунд, видеозапись необходимо сделать самостоятельно.

## Тесты и отчёты

Для каждого запуска используйте новый каталог системного TEMP.

```powershell
$testTemp = Join-Path $env:TEMP ('sms_guard_tests_' + [guid]::NewGuid().ToString('N'))
python -m pytest -q -p no:cacheprovider --basetemp=$testTemp
python manage.py makemigrations --check --dry-run
```

Полный аудит сохранённых моделей и фиксированных выборок выполняется без обучения.

```powershell
python -m scripts.evaluate_task18
$env:TASK18_EVIDENCE_PATH = Join-Path (Get-Location) 'docs/18_runtime_results.json'
$testTemp = Join-Path $env:TEMP ('sms_guard_coverage_' + [guid]::NewGuid().ToString('N'))
python -m pytest -q -p no:cacheprovider --basetemp=$testTemp --cov=ml --cov=apps.analysis.services --cov-report=json:docs/18_coverage.json --junitxml=docs/18_pytest.xml
python -m scripts.report_task18
Remove-Item Env:TASK18_EVIDENCE_PATH
```

Последняя команда сборки отчётов требует также сохранённого `docs/18_environment_results.json` с результатами проверки окружения и браузера. Для новой машины эти проверки необходимо повторить, старые результаты нельзя выдавать за новый запуск.

Основные документы

- docs/18_final_testing.md — результаты, покрытие, метрики, ошибки, ограничения.
- docs/18_traceability_matrix.md — все требования ТЗ и их статусы.
- docs/18_test_cases.md — тест-кейсы с фактическими результатами.
- docs/18_demo_scenarios.md — 13 примеров, пять сценариев и план видео.
- docs/03_dataset_description.md и docs/10_url_dataset.md — источники данных.
- docs/08_sms_model_comparison.md и docs/11_url_model.md — предыдущие эксперименты.

## Ограничения качества

SMS-модель обучена на английском ham/spam, что не равно доказанному мошенничеству. На фиксированном Test сохранённого артефакта Precision около 0,744 и F1 около 0,829, ниже минимума ТЗ 0,90. Recall на искусственных RU/KK составляет 0,50/0,52, поддержка этих языков не подтверждена. Набор RU/KK использовался при калибровке Risk, поэтому его показатели совместного анализа не являются независимым тестом.

URL-модель имеет высокие метрики на PhiUSIIL, но зависит от смещения HTTPS/path/query и ошибается на обычных разделах сайтов. Словарь не покрывает все формулировки и отрицания. Оценка вероятностная, не гарантирует безопасность. Будущие улучшения описаны в итоговом отчёте, текущие модели и правила в задании №18 не меняются.
