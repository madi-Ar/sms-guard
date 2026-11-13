# Задание №14 — backend и REST API

## Цель и границы

API Django 5.2 / DRF вызывает существующие `analyze_sms()` и `analyze_url()`.
ML, Risk Score, данные, разбиения, сохранённые модели и зависимости не менялись.
Интерфейс, расширенная БД/согласия и Dashboard заданий №15–17 не реализуются.
Сайты не открываются, DNS/HTTP-проверок ссылок нет.

## Маршруты и сессия

Все пути начинаются с `/api/v1/`.

| Метод | Путь | Доступ и успешный ответ |
|---|---|---|
| POST | auth/register/ | Анонимный + CSRF, 201: id, username, email |
| POST | auth/login/ | Анонимный + CSRF, 200: id, username; sessionid cookie |
| POST | auth/logout/ | Сессия + CSRF, 204 |
| POST | check/sms/ | Сессия + CSRF, 201: результат и ID новой записи |
| POST | check/url/ | Сессия + CSRF, 201: результат URL и ID |
| GET | checks/ | Сессия, 200: count, next, previous, results |
| GET | checks/{id}/ | Сессия, 200: сохранённый безопасный результат |
| DELETE | checks/{id}/ | Сессия + CSRF, 204 |
| GET | stats/summary/ | Сессия, 200: агрегаты |
| GET | health/ | Публичный, 200: status, models, csrf_token |

Регистрация не выполняет автоматический вход. Пример тела:
`{"username":"student_1","email":"student@example.org","password":"StudyPass293"}`.
Вход принимает `username`, `password`. Используются стандартные Django User,
`create_user()`, `authenticate()`, `login()`, `logout()` и стандартный password hashing.
Пароль не возвращается и не журналируется. Отсутствующая сессия даёт 401
(включая после logout), CSRF-нарушение у вошедшего пользователя — 403.

Username: 3–30 ASCII-букв, цифр или `_`; email валидируется, username/email
проверяются на уникальность без учёта регистра. Пароль: минимум восемь символов,
хотя бы одна буква и цифра. Встроенный User имеет уникальный username на уровне
БД; уникальность непустого email дополнительно защищает уникальный индекс
LOWER(email) в миграции accounts. Пустые email у прежних пользователей допустимы;
API требует email. Индекс работает на SQLite/PostgreSQL, рассчитан на стандартный
Django User из ТЗ. Если в старой БД уже есть одинаковые непустые email, миграция
остановится: исправлять пользовательские записи автоматически она не будет.

Пять последовательных неверных паролей дают 401, последующие попытки для того
же имени блокируются на 60 секунд (429 и Retry-After), даже с правильным паролем.
Успешный вход до блокировки сбрасывает счётчик. Ответ для неизвестного имени
и неверного пароля одинаков. Ключ кеша содержит SHA-256 имени, без пароля.

## CSRF и cookie

Сначала GET health, сохранить cookie и `csrf_token` из JSON; затем передавать
его в `X-CSRFToken`. После входа Django меняет токен: повторить GET health.
CSRF проверяется явно и для анонимной регистрации/входа; для остальных
изменяющих запросов — сессионной аутентификацией DRF. Проверяются также Origin
и HTTPS Referer стандартными средствами Django. CSRF в API-тестах включён.

Cookie sessionid и csrftoken: HttpOnly, SameSite=Lax. Для HTTPS установить
`DJANGO_COOKIE_SECURE=True`; при DEBUG=False это значение по умолчанию.
В `.env.example` показан локальный HTTP-режим False, при развёртывании его
нужно переключить. HTTPS должен быть настроен на сервере; доверие proxy-заголовкам
не включено автоматически. HttpOnly CSRF cookie читается сервером, клиент получает
маскированный токен из health. Токен не записывается в логи.

## Валидация и результаты

SMS: `{"text":"Please bring the notebook tomorrow."}`. Принимается только
строка длиной 1–1000 **до** очистки, пробельный/пустой текст отклоняется.
Cc-символы удаляются; CR/LF/TAB заменяются пробелами, чтобы не склеивать слова.
Регистр, пунктуация и слова сохраняются: NLP-нормализацию делает только прежний
ML Pipeline. URL: `{"url":"https://example.org/path?code=123"}`, только строка
до 2048 символов, прежний офлайн-валидатор `validate_url()`.

Пример формы SMS-ответа ниже использует **фиктивные значения API-тестов**,
а не новое измерение модели; id/дата зависят от БД:

```json
{
  "id": 1,
  "type": "sms",
  "language": "en",
  "category": "SAFE",
  "risk_level": "LOW",
  "risk_score": 10,
  "p_sms": 0.1,
  "p_url_max": null,
  "urls": [],
  "features": {},
  "explanation": ["Вероятностная оценка."],
  "recommendation": "Проверяйте отправителя.",
  "disclaimer": "Оценка носит вероятностный характер и не гарантирует безопасность или отсутствие мошенничества. SMS-модель оценивает spam, а не доказанный факт мошенничества.",
  "model_versions": {"sms": "test", "url": "test"},
  "created_at": "2026-11-13T12:00:00+00:00"
}
```

В реальном ответе поля берутся из `AnalysisResult.to_dict()`, включая версии,
найденные URL с p_url, их отдельными категориями и признаками. API добавляет
только id и created_at. Для URL возвращается `URLAnalysis.to_dict()` плюс
`id`, `created_at`, `type="url"`, `p_sms=null`, `risk_level=null`, `risk_score=null`.
Например, фиктивный тестовый URL-результат содержит `p_url=0.7`,
`category="DANGEROUS"`, `features={"is_https":1}`. Алгоритм №13 требует p_sms,
поэтому для отдельной URL общий Risk Score не вычисляется и не выдумывается.
URL сохраняет категории SAFE/SUSPICIOUS/DANGEROUS. Они не являются доказательством
мошенничества; ограничения прежних моделей и языков остаются актуальны.

## Минимальная БД и приватность

Одна новая таблица `analysis_check` (модель Check): id, user FK с CASCADE,
type, created_at, category, nullable risk_level/risk_score/p_sms/p_url/p_url_max,
masked_input и JSONField result_json. Порядок: created_at DESC, id DESC.
Миграция `apps/analysis/migrations/0001_initial.py` зависит от Django User.
`apps/accounts/migrations/0001_unique_email.py` добавляет только индекс email.
Новых профилей, таблиц признаков/версий/вложенных URL нет.

Исходное SMS не хранится. В masked_input все непробельные символы заменяются
на `•`, пробельные — на пробел. Полные URL также маскируются, включая вложенные
URL внутри SMS: query может содержать код или пароль. Сохраняется только контракт
результата; полный исходный ввод и найденные URL дополнительно редактируются
в строковых полях JSON. Незнакомые верхнеуровневые поля не сохраняются.
История возвращает безопасную сохранённую копию и masked_input; сам POST-ответ
содержит фактический результат анализа, в том числе URL, и не должен
автоматически журналироваться клиентом. Маска раскрывает длину и расположение
пробелов, признаки — числовые характеристики, но не полный исходный текст.

Создание записи, получение id/даты и обновление JSON выполняются в одной
`transaction.atomic`: при ошибке частичная запись не остаётся. Старую SQLite-БД
не удаляли. Тестовая БД pytest создаётся отдельно с применением миграций.

История всегда фильтруется по текущему user, в том числе для staff. Чужой id
при GET/DELETE даёт 404. Пагинация фиксированно по 20; фильтры `date_from`,
`date_to` — включительные даты YYYY-MM-DD в часовом поясе Django; `type` sms/url,
`category` SAFE/SUSPICIOUS/FRAUDULENT/DANGEROUS, `risk_level` LOW/MEDIUM/HIGH.
Неверные значения/обратный диапазон дат дают 400. У URL-only risk_level null,
поэтому фильтр по уровню не включает такие записи.

Статистика: `sms_count`, `url_count`, `categories` (объект category→count),
`average_risk_score` (null при отсутствии рассчитанных баллов). Обычный
пользователь видит свои агрегаты, staff — все агрегаты без текстов и результатов.

## Лимиты, ошибки и журнал

CheckThrottle имеет единый scope `checks`: всего 30 запросов/минуту на пользователя
для SMS и URL вместе, следующий — 429. Лимит учитывает также невалидные попытки
проверки. Стандартный кеш Django LocMemCache и DRF достаточны для локального
учебного процесса, но не дают общий лимит между процессами; DRF throttling
также не гарантирует точный предел при конкурентных запросах. Для production
нужны общий кеш и более строгая синхронизация. Redis/Celery не добавлялись.

```json
{"error":{"code":"MODEL_UNAVAILABLE","message":"Модели анализа временно недоступны."}}
```

| HTTP | error.code |
|---|---|
| 400 | VALIDATION_ERROR |
| 401 | UNAUTHENTICATED |
| 403 | FORBIDDEN |
| 404 | NOT_FOUND |
| 405 | METHOD_NOT_ALLOWED |
| 429 | RATE_LIMITED |
| 503 | MODEL_UNAVAILABLE |
| 500 | INTERNAL_ERROR |

Обработчик DRF не раскрывает traceback, пути или внутреннее сообщение ошибки
модели. Для валидации возвращается первая понятная причина. Ошибки загрузки/
предсказания прежнего ModelPredictor превращаются в 503, результат не сохраняется.
Остальные непредвиденные исключения API дают обезличенный 500.
Журнал проверок содержит duration_ms, check_id, category и status, без ввода,
паролей/токенов и содержимого исключений. Удаление журналирует id и статус.

Health сохраняет `status="ok"` и отдельно показывает доступность четырёх файлов
моделей/JSON. Проверяет чтение одного байта, не десериализует и не обучает модели.
Это проверка наличия/читаемости, а не проверка корректности joblib. Повреждение
обнаруживается при ленивой загрузке модели и приводит к 503. Кеш моделей и
поведение после ошибки загрузки остаются прежними (нужен перезапуск после ремонта).
Старый тест №2 теперь проверяет status по ключу, разрешая новые поля health.

## Ручной запуск (PowerShell)

Из корня проекта, локальный HTTP с DEBUG=True и COOKIE_SECURE=False:

```powershell
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver
```

Во втором терминале PowerShell (демонстрационный пароль не использовать в жизни):

```powershell
$base = 'http://127.0.0.1:8000/api/v1'
$health = Invoke-RestMethod "$base/health/" -SessionVariable guardSession
$headers = @{ 'X-CSRFToken' = $health.csrf_token }
$register = @{ username='student_1'; email='student@example.org'; password='StudyPass293' } | ConvertTo-Json
Invoke-RestMethod "$base/auth/register/" -Method Post -WebSession $guardSession -Headers $headers -ContentType 'application/json' -Body $register
$credentials = @{ username='student_1'; password='StudyPass293' } | ConvertTo-Json
Invoke-RestMethod "$base/auth/login/" -Method Post -WebSession $guardSession -Headers $headers -ContentType 'application/json' -Body $credentials
$health = Invoke-RestMethod "$base/health/" -WebSession $guardSession
$headers = @{ 'X-CSRFToken' = $health.csrf_token }
$body = @{ text='Please bring the notebook tomorrow.' } | ConvertTo-Json
$result = Invoke-RestMethod "$base/check/sms/" -Method Post -WebSession $guardSession -Headers $headers -ContentType 'application/json' -Body $body
$result | ConvertTo-Json -Depth 10
$urlBody = @{ url='https://example.org/path?code=123' } | ConvertTo-Json
Invoke-RestMethod "$base/check/url/" -Method Post -WebSession $guardSession -Headers $headers -ContentType 'application/json' -Body $urlBody
Invoke-RestMethod "$base/checks/?type=sms" -WebSession $guardSession
Invoke-RestMethod "$base/checks/$($result.id)/" -WebSession $guardSession
Invoke-RestMethod "$base/stats/summary/" -WebSession $guardSession
Invoke-RestMethod "$base/checks/$($result.id)/" -Method Delete -WebSession $guardSession -Headers $headers
Invoke-RestMethod "$base/auth/logout/" -Method Post -WebSession $guardSession -Headers $headers
```

Повторная регистрация этого username/email ожидаемо даёт 400; для следующих
демонстраций достаточно входа. Анализируется строка URL, сайт не открывается.

## Проверка

```powershell
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.\.pytest_task14_temp
```

Фактические результаты 13.11.2026:

- `manage.py check`: ошибок нет (0 silenced).
- `manage.py migrate`: обе новые миграции применены, пользовательская БД сохранена.
- `manage.py makemigrations --check --dry-run`: No changes detected.
- Полный запуск с `--basetemp=.\.pytest_task14_temp`: **303 passed in 128.76s**.
- После добавления пяти дополнительных проверок заключительный полный запуск
  с отдельной папкой `--basetemp=.\.pytest_task14_final`: **308 passed in 145.15s**.
  Это **238 прежних + 70 API-тестов**, включая параметризованные случаи.
- `git diff --check` и синтаксический разбор новых Python-файлов: без ошибок.
- SHA-256 защищённых моделей, метаданных и данных совпали с записями задания №13;
  risk_config.yaml совпадает с выбранной конфигурацией калибровки.

Проверены все 25 обязательных направлений ТЗ, включая настоящие сессии и CSRF,
успех/ошибки регистрации и входа, блокировку 60 секунд и сброс счётчика, выход,
401/403/404/429/503, строгие типы и длины, очистку Cc без NLP-нормализации,
общий лимит SMS/URL, фильтры/пагинацию, staff/владение, пустую статистику и
исключение null из среднего, приватность БД/логов, SQLi/XSS как данные,
Secure cookie, индекс email и откат частично выполненного сохранения.
API-тесты используют реальные сессии, CSRF и ORM, но фиктивные результаты
ML-сервисов; прежние тесты моделей продолжают проверять их отдельно.
Создание тестовой БД включает применение миграций на чистой SQLite.

При разработке исправлены циклический импорт DRF и слишком раннее отклонение
нулевого символа стандартным валидатором: очистка SMS теперь происходит до
запуска валидаторов поля, но после проверки исходной длины. В первом отдельном
прогоне было 64 passed / 1 failed; в обоих последующих полных — ошибок нет.
Предупреждения Git LF→CRLF относятся к настройке строк Windows и не являются
ошибками проверки. Временные папки pytest очищены. Коммит и push не выполнялись.
