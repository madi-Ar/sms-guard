"""Сборка отчётов из фактических JSON, JUnit и coverage задания №18."""
from collections import Counter
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


def load(name):
    return json.loads((DOCS / name).read_text(encoding='utf-8'))


def write(name, lines):
    (DOCS / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('<', '&lt;').replace('>', '&gt;')


def main():
    model = load('18_model_results.json')
    runtime = load('18_runtime_results.json')
    environment = load('18_environment_results.json')
    coverage = load('18_coverage.json')
    ml_rows = [entry['summary'] for name, entry in coverage['files'].items() if name.startswith('ml')]
    ml_covered = sum(row['covered_lines'] for row in ml_rows)
    ml_total = sum(row['num_statements'] for row in ml_rows)
    calibration = json.loads((ROOT / 'ml/artifacts/risk_calibration.json').read_text(encoding='utf-8'))
    calibrated = calibration['results'][calibration['selected_index']]
    xml = ET.parse(DOCS / '18_pytest.xml')
    nodes = list(xml.iter('testcase'))
    counts = Counter('ERROR' if n.find('error') is not None else 'FAILED' if n.find('failure') is not None else 'SKIPPED' if n.find('skipped') is not None else 'PASSED' for n in nodes)
    cases = []

    def case(group, title, file, stem, data, expected):
        matches = [n for n in nodes if n.get('classname', '').endswith(file) and n.get('name', '').split('[')[0] == stem]
        status = 'NOT TESTED' if not matches else 'FAIL' if any(n.find('failure') is not None or n.find('error') is not None for n in matches) else 'NOT TESTED' if any(n.find('skipped') is not None for n in matches) else 'PASS'
        cases.append(dict(id=f'TC-{len(cases)+1:02}', group=group, title=title, file=file, stem=stem, data=data, expected=expected, status=status, runs=len(matches)))

    case('Регистрация и авторизация', 'Регистрация и salted hash', 'test_api', 'test_register_hashes_password', 'Логин new_user, тестовый email и пароль фикстуры', '201; пароль хеширован, исходный пароль не возвращается')
    case('Регистрация и авторизация', 'Повтор логина/email', 'test_api', 'test_duplicate_registration', 'Существующий alice и email в другом регистре', '400, второй пользователь не создаётся')
    case('Регистрация и авторизация', 'Слабые пароли', 'test_api', 'test_weak_password', 'short1, onlyletters, 123456789', '400 для всех вариантов')
    case('Регистрация и авторизация', 'Недопустимый логин', 'test_api', 'test_invalid_username', 'Длина 2/31, кириллица, SQL/HTML', '400')
    case('Регистрация и авторизация', 'Вход и cookie', 'test_api', 'test_login_creates_session', 'Корректная тестовая сессия', 'Сессия активна; HttpOnly и SameSite=Lax')
    case('Регистрация и авторизация', 'Блокировка перебора', 'test_api', 'test_five_failures_lock_for_sixty_seconds', '5 неверных паролей и моделируемые 59/61 секунд', 'После 5 ошибок блокировка, через 61 с доступ восстановлен')
    case('Регистрация и авторизация', 'Выход', 'test_api', 'test_logout_destroys_session', 'POST logout с CSRF', '204; следующий защищённый запрос отклонён')
    case('Проверка SMS', 'Пустой и слишком длинный ввод', 'test_api', 'test_invalid_sms', 'Пустая строка, пробелы, неверный тип, 1001 символ', '400, анализа нет')
    case('Проверка SMS', 'Граница 1000 символов', 'test_api', 'test_maximum_sms_length_allowed', '1000 символов x', '201, сервис получает все 1000 символов')
    case('Проверка SMS', 'Unicode и регистр', 'test_preprocessing', 'test_mixed_case_and_nfkc', 'Unicode NFKC и смешанный регистр', 'Нормализованная строка по утверждениям теста')
    case('Проверка SMS', 'Маркеры сумм', 'test_preprocessing', 'test_money', 'Суммы ₸, тг, руб, $, £', 'Маркеры money, сохранён остальной текст')
    case('Проверка SMS', 'Казахские буквы', 'test_preprocessing', 'test_kazakh_letters', 'Казахский текст', 'Буквы сохранены')
    case('Проверка SMS', 'Единая нормализация', 'test_analysis_services', 'test_real_final_models_normalize_sms_once_and_preserve_artifacts', 'Существующие реальные артефакты', 'Нормализация один раз, контрольные суммы совпадают')
    case('Проверка URL', 'Числовые признаки', 'test_url_features', 'test_normal_https_url_has_all_expected_values', 'Обычный HTTPS URL', 'Полный числовой набор признаков')
    case('Проверка URL', 'URL без схемы', 'test_url_features', 'test_missing_scheme_only_changes_parsing', 'Домен без http://', 'http добавляется только для разбора')
    case('Проверка URL', 'IPv4', 'test_url_features', 'test_ipv4', 'URL с IPv4', 'IP-флаг установлен')
    case('Проверка URL', 'IPv6', 'test_url_features', 'test_ipv6', 'URL с IPv6', 'Корректный разбор и IP-флаг')
    case('Проверка URL', 'Punycode', 'test_url_features', 'test_punycode_and_unicode_have_same_masking_flags', 'IDNA и Unicode домены', 'Эквивалентные флаги')
    case('Проверка URL', 'Сокращатели', 'test_url_features', 'test_shortener_list_and_exact_domain_matching', 'Список сервисов и похожие домены', 'Не менее 15, точное сопоставление hostname')
    case('Проверка URL', 'Бренды и смешение алфавитов', 'test_url_features', 'test_brand_distance_and_exact_brand_subdomain', 'Похожие брендам hostname', 'Проверены расстояние и поддомены')
    case('Проверка URL', 'Некорректные адреса', 'test_api', 'test_invalid_url', 'Некорректный hostname, схема, пустой ввод', '400, HTTP 500 отсутствует')
    case('Совместный анализ SMS и URL', 'SMS без ссылки', 'test_analysis_services', 'test_sms_without_urls_fills_risk_but_does_not_fake_storage', 'SMS без URL', 'URL-модель не вызывается, Risk Score заполнен')
    case('Совместный анализ SMS и URL', 'Одна ссылка', 'test_analysis_services', 'test_sms_with_one_url', 'SMS с одной ссылкой', 'Ссылка анализируется, p_url_max согласована')
    case('Совместный анализ SMS и URL', 'Не более трёх URL', 'test_analysis_services', 'test_only_first_three_urls_are_analyzed_in_order', 'От одной до нескольких ссылок', 'Первые три в исходном порядке')
    case('Расчёт Risk Score', 'CODE_REQUEST минимум', 'test_risk', 'test_code_request_has_minimum_seventy', 'Группа CODE_REQUEST', 'Risk Score не ниже 70')
    case('Расчёт Risk Score', 'Порог URL из metadata', 'test_risk', 'test_combined_override_uses_metadata_t2_not_hardcoded_number', 'Разные t2 во временных metadata', 'Используется заданный t2')
    case('Расчёт Risk Score', 'Границы и округление', 'test_risk', 'test_boundaries_and_half_up_rounding', 'Значения около границ LOW/MEDIUM/HIGH', 'Корректное half-up округление и уровень')
    case('Расчёт Risk Score', 'Несколько override', 'test_risk', 'test_all_overrides_use_maximum_not_sum_and_ignore_order', 'Несколько групп/URL-флагов', 'Максимум обязательных минимумов, порядок не влияет')
    case('Правила социальной инженерии', 'Шесть групп трёх языков', 'test_analysis_services', 'test_all_six_rule_groups_have_ten_expressions_per_language_and_match', 'Все словарные выражения', '6 групп, не менее 10 выражений на язык')
    case('Правила социальной инженерии', 'Границы и повторения', 'test_analysis_services', 'test_phrase_whitespace_case_boundaries_and_group_deduplication', 'Повторы, регистр, пробелы', 'Группа учитывается один раз, соблюдены границы слов')
    case('Сохранение результатов', 'Check и связанные данные', 'test_history', 'test_related_records', 'SMS с URL и URGENCY', 'Check, UrlResult и DetectedFeature согласованы')
    case('Сохранение результатов', 'Откат транзакции', 'test_history', 'test_atomic_rollback', 'Исключение при сохранении дочерней записи', 'Все записи откатываются')
    case('История и фильтрация', 'Свои записи', 'test_history', 'test_history_own_only', 'Два пользователя', 'В истории только записи владельца')
    case('История и фильтрация', 'Комбинация фильтров', 'test_history', 'test_history_filters', 'Даты, тип, категория, уровень', 'Выборка соответствует всем фильтрам')
    case('История и фильтрация', 'Пагинация', 'test_history', 'test_pagination_filters_order', '21 запись', '20/1 строк, новые сверху, параметры сохраняются')
    case('История и фильтрация', 'Удаление с подтверждением', 'test_history', 'test_delete_requires_post_confirmation_and_cascades', 'GET, POST без/с подтверждением', 'GET 405, без подтверждения запись остаётся, затем каскадное удаление')
    case('Профиль пользователя', 'Согласие по умолчанию', 'test_history', 'test_profile_created_default_false', 'Новый пользователь', 'store_sms_text=False')
    case('Профиль пользователя', 'Переключение согласия', 'test_history', 'test_consent_only_new_sms', 'Включение и выключение', 'Только новые SMS сохраняют полный текст при согласии')
    case('Профиль пользователя', 'Пароль и сессия', 'test_history', 'test_password_change_session', 'Старый/новый пароль', 'Пароль изменён, сессия сохранена')
    case('Dashboard', 'Карточки и NULL', 'test_dashboard', 'test_cards_categories_and_null_average', 'Все 4 категории и NULL/0 Risk Score', 'Корректные количества, среднее 50')
    case('Dashboard', 'Периоды', 'test_dashboard', 'test_period_affects_all_sections', '7/30/90 дней и пограничные даты', 'Все разделы используют один период')
    case('Dashboard', 'Дни без проверок', 'test_dashboard', 'test_daily_real_dates_and_zero_days', 'Два дня с данными, промежуток без данных', 'Нули сохранены, реальные даты совпадают')
    case('Dashboard', 'Топ-5 фактических групп', 'test_dashboard', 'test_top_five_actual_groups_and_distinct_checks', '6 групп и повтор кода', '5 фактически наиболее частых, DISTINCT по проверкам')
    case('Dashboard', 'Администратор', 'test_dashboard', 'test_admin_all_aggregates_without_foreign_detail_access', 'is_staff и второй пользователь', 'Общие агрегаты, чужие подробности 404')
    case('REST API', 'Health без ML загрузки', 'test_api', 'test_health_without_model_loading', 'Нет части файлов артефактов', '200, корректная доступность, модели не загружаются')
    case('REST API', 'JSON-контракт', 'test_analysis_services', 'test_json_serialization_contains_all_required_fields', 'Результат сервиса', 'Все обязательные поля сериализуются')
    case('REST API', 'Пять потоков на реальных моделях', 'test_final_validation', 'test_five_real_model_flows_save_history_detail_dashboard', 'Сценарии A–E', '201, детали/история доступны, Dashboard увеличивает счётчики')
    case('Безопасность', 'SQL Injection', 'test_final_validation', 'test_sql_injection_input_and_filters', "' OR 1=1 --", 'Ввод остаётся данными, фильтры 400, чужие записи недоступны')
    case('Безопасность', 'XSS', 'test_final_validation', 'test_exact_xss_payload_history_result_dashboard', '<script>alert(1)</script>', 'История экранирует, Dashboard не выдаёт текст, безопасный JS-вывод')
    case('Безопасность', 'CSRF', 'test_final_validation', 'test_state_changes_reject_missing_csrf', 'POST без токена, enforce_csrf_checks=True', '403')
    case('Безопасность', 'HTTPS cookie', 'test_api', 'test_secure_cookies_for_https', 'HTTPS и Secure=True', 'Session и CSRF cookie Secure')
    case('Безопасность', 'Удаление чужой истории через ID', 'test_final_validation', 'test_delete_history_ignores_foreign_user_parameter', 'user/user_id другого пользователя', 'Удаляется только собственная история')
    case('Надёжность', 'Бинарный JSON', 'test_final_validation', 'test_binary_or_invalid_json_is_400', 'Неверный UTF-8, бинарные байты, список и null', '400, необработанного 500 нет')
    case('Надёжность', 'Отсутствие и повреждение артефактов', 'test_final_validation', 'test_missing_and_corrupt_artifacts_return_503', 'Временные копии без каждой модели/metadata и невалидный JSON', 'Health 200, анализ 503, записи не создаются')
    case('Надёжность', 'Автономная работа', 'test_analysis_services', 'test_analysis_does_not_access_network', 'Запрещённые DNS/HTTP/socket вызовы', 'Анализ работает без сети')
    case('Производительность', '100 реальных API-запросов', 'test_final_validation', 'test_runtime_latency_one_hundred_real_requests', 'SMS с тремя зарезервированными URL', '100 успешных запросов; среднее/максимум измерены отдельно')
    case('Производительность', 'Dashboard с 1000 записями', 'test_final_validation', 'test_dashboard_series_one_thousand_rows', '1000 искусственных записей, 10+10 запросов', 'Измерены API и HTML отдельно')
    case('Производительность', 'Лимит 30/мин', 'test_final_validation', 'test_forty_requests_enforce_thirty_per_minute', '40 запросов в одном виртуальном интервале', '30 ответов 201, 10 ответов 429')
    case('Конфиденциальность данных', 'Текст отсутствует в JSON', 'test_history', 'test_json_does_not_store_source', 'SMS с телефоном и суммой без согласия', 'input_text=NULL, чувствительные значения скрыты')
    case('Конфиденциальность данных', 'Маскирование', 'test_history', 'test_sensitive_mask', 'Телефоны, карты 13–19 цифр, валюты', 'Замена на ***, безопасная часть остаётся')
    case('Конфиденциальность данных', 'Логи', 'test_api', 'test_sms_and_embedded_url_absent_from_database_and_logs', 'Уникальный SMS и URL с кодом', 'Исходный текст и код отсутствуют в логах/сохранённом JSON')
    case('Регистрация и авторизация', 'Защита Dashboard', 'test_dashboard', 'test_page_requires_login', 'GET / без сессии', '302 на /login/?next=/')
    case('Регистрация и авторизация', 'Защита истории и профиля', 'test_history', 'test_login_required', 'GET /history/ и /profile/ без входа', '302 на вход')
    case('Регистрация и авторизация', 'Защита страниц проверки', 'test_ui', 'test_anonymous_redirects_to_login', 'SMS/URL/результат без сессии', '302 на вход')
    case('Проверка SMS', 'Язык и предупреждения', 'test_analysis_services', 'test_language_warnings', 'ru, kk, en и недостаточно букв', 'Контракт текущей эвристики и предупреждения подтверждены, other отдельно не реализован')
    case('Проверка URL', 'Подозрительные слова', 'test_url_features', 'test_keyword_occurrences_are_case_insensitive', 'login/verify/secure в разных регистрах', 'Частоты совпадают с ожидаемыми')

    lines = ['# Тест-кейсы задания №18', '', 'Каждый автокейс выполнен на отдельной тестовой БД Django. Общие предусловия — установленное окружение, локальные данные и артефакты, фикстуры указанного теста. Работающие алгоритмы и прежние тесты не изменялись.', '',
             'Статус PASS получен из JUnit XML по реальному имени теста, включая все его параметризованные варианты. PASSED/FAILED/ERROR/SKIPPED учитываются раздельно. PASS автокейса не доказывает качество классификации, ожидаемые категории демонстрационных примеров оцениваются отдельно.', '']
    for c in cases:
        nodeid = f"tests/{c['file']}.py::{c['stem']}"
        lines += [f"## {c['id']}. {c['title']}", '',
            f"- Категория — {c['group']}.", '- Предусловия — отдельная тестовая БД и фикстуры выбранного теста. Негативные ML-сценарии используют временные копии или mocks.',
            f"- Шаги — выполнить `python -m pytest -q {nodeid}` с уникальным `--basetemp` в TEMP. Запросы, проверяемые действия и утверждения приведены в исходнике теста.",
            f"- Входные данные — {c['data']}.", f"- Ожидается — {c['expected']}.",
            f"- Фактически — {c['status']}, выполнено вариантов {c['runs']}. Результат утверждений зарегистрирован в JUnit.",
            f"- Статус — **{c['status']}**.", f"- Доказательство — [{nodeid}](../tests/{c['file']}.py), [JUnit](18_pytest.xml).", '']
    audits = [
        ('QC-01', 'Минимальные метрики SMS', 'Фиксированные 771 SMS, сохранённый Pipeline и прежний порог', 'Precision, Recall и F1 ≥0.90', model['sms']['saved_model'], all(model['sms']['saved_model'][key] >= .9 for key in ('precision', 'recall', 'f1'))),
        ('QC-02', 'Минимальные метрики URL', 'Фиксированные 8660 URL и сохранённые доменные группы', 'Precision/Recall ≥0.85, ROC-AUC ≥0.90', model['url']['metrics'], model['url']['metrics']['precision'] >= .85 and model['url']['metrics']['recall'] >= .85 and model['url']['metrics']['roc_auc'] >= .9),
        ('QC-03', 'Recall RU', '100 прежних искусственных RU SMS, прежний SMS-порог', 'Recall fraud ≥0.80', model['languages']['ru']['classifier'], model['languages']['ru']['classifier']['recall'] >= .8),
        ('QC-04', 'Recall KK', '50 прежних искусственных KK SMS, прежний SMS-порог', 'Recall fraud ≥0.80', model['languages']['kk']['classifier'], model['languages']['kk']['classifier']['recall'] >= .8),
        ('BROW-01', 'Двойной клик', 'Chrome и отдельная копия БД; безопасный английский ввод A', 'Два быстрых клика дают одну запись', environment['browser']['double_click'], environment['browser']['double_click']['duplicates'] == 0),
        ('ENV-01', 'Конфигурация DEBUG', 'Исходная dev-конфигурация и отдельная чистая установка', 'В демонстрационной конфигурации DEBUG=False, секреты вне Git', {'working_debug': environment['working_debug'], 'fresh_install': environment['fresh_install']}, not environment['working_debug']),
        ('ENV-02', 'Ротация логов', 'Конфигурация Django и события анализа', 'RotatingFileHandler с ограничением размера и backupCount', environment['logging'], environment['logging']['handler'].endswith('RotatingFileHandler') and environment['logging']['backupCount'] > 0),
        ('ENV-03', 'Неподдерживаемый язык', 'Текущая эвристика detect_language', 'Японский и французский получают other', environment['language_diagnostics'], all(value == 'other' for value in environment['language_diagnostics'].values())),
        ('COV-01', 'Покрытие строк', 'Полный pytest-cov и 18_coverage.json', 'Покрытие ml/ и каждого проверяемого модуля ≥60%', {'ml_covered': ml_covered, 'ml_total': ml_total, 'ml_percent': 100 * ml_covered / ml_total}, all(row['percent_covered'] >= 60 for row in ml_rows)),
        ('QC-05', 'Ранее выбранная калибровка Risk', 'Неизменённый risk_calibration.json задания №13', 'Recall medium/high ≥0.95 и ложный HIGH ≤0.03 на EN и RU/KK', {name: calibrated[name] for name in ('en_validation', 'ru_kk_development')}, all(calibrated[name]['meets_targets'] for name in ('en_validation', 'ru_kk_development'))),
    ]
    lines += ['## Проверки качества и аудит артефактов', '', 'Эти кейсы используют фактические JSON и браузерные наблюдения, их FAIL не равен ошибке утверждения pytest.', '']
    for ident, title, preconditions, expected, actual, passed in audits:
        lines += [f'### {ident}. {title}', '', f'- Предусловия и вход — {preconditions}.',
            '- Шаги — воспроизвести соответствующий аудит по scripts/evaluate_task18.py или браузерному плану, сверить записанные JSON и установленную границу.',
            f'- Ожидается — {expected}.', f'- Фактически — `{json.dumps(actual, ensure_ascii=False)}`.',
            f"- Статус — **{'PASS' if passed else 'FAIL'}**.", '- Доказательство — 18_model_results.json, 18_environment_results.json, 18_coverage.json и снимки браузера.', '']
    manual = [
        ('MAN-01', 'Адаптивность 360/768/1366 px', 'Открыть все страницы в DevTools на трёх точных ширинах, проверить отсутствие горизонтальных потерь и доступность кнопок. Выполненная проверка окна Chrome 643 px не заменяет эти измерения.'),
        ('MAN-02', 'Полная загрузка Dashboard в браузере', 'Создать 1000 тестовых записей в отдельной БД, измерить DevTools Network/Performance, включая CDN, API и отрисовку canvas. Серверные замеры приведены отдельно.'),
        ('MAN-03', 'Видео 3–5 минут', 'Записать последовательность из 18_demo_scenarios.md и приложить видео. Сценарий подготовлен, видеозапись не создавалась.'),
        ('MAN-04', 'Пояснительная записка 40–60 страниц', 'Собрать документы заданий, Activity/Sequence/ER/Use Case, пять экранов, репозиторий и материалы защиты по §8.4. Отдельная итоговая записка на 40–60 страниц в этой работе не создана.'),
    ]
    lines += ['## Ручные проверки', '']
    for ident, title, action in manual:
        lines += [f'### {ident}. {title}', '', '- Предусловия — работающий браузер и отдельная демонстрационная база.', f'- Шаги и вход — {action}', '- Ожидается — требования исходного ТЗ подтверждены измерением или артефактом.', '- Фактически — соответствующий полный артефакт отсутствует.', '- Статус — **NOT TESTED**.', '- Доказательство — ограничение зафиксировано в 18_final_testing.md.', '']
    write('18_test_cases.md', lines)
    ids = {c['stem']: c['id'] for c in cases}
    mapping = {
        'ФТ-1.1': ['test_register_hashes_password', 'test_weak_password', 'test_invalid_username'],
        'ФТ-1.2': ['test_duplicate_registration'], 'ФТ-1.3': ['test_five_failures_lock_for_sixty_seconds'],
        'ФТ-1.4': ['test_register_hashes_password'], 'ФТ-1.5': ['test_logout_destroys_session'],
        'ФТ-1.6': ['test_page_requires_login', 'test_login_required', 'test_anonymous_redirects_to_login'],
        'ФТ-1.7': ['test_history_own_only', 'test_delete_history_ignores_foreign_user_parameter'],
        'ФТ-1.8': ['test_login_creates_session', 'test_secure_cookies_for_https', 'test_state_changes_reject_missing_csrf'],
        'ФТ-1.9': ['test_admin_all_aggregates_without_foreign_detail_access'],
        'ФТ-2.1': ['test_invalid_sms', 'test_maximum_sms_length_allowed'],
        'ФТ-2.2': ['test_language_warnings'], 'ФТ-2.3': ['test_money', 'test_mixed_case_and_nfkc', 'test_kazakh_letters'],
        'ФТ-2.4': ['test_real_final_models_normalize_sms_once_and_preserve_artifacts'],
        'ФТ-2.5': ['test_all_six_rule_groups_have_ten_expressions_per_language_and_match'],
        'ФТ-2.6': ['test_json_serialization_contains_all_required_fields'],
        'ФТ-2.7': ['test_only_first_three_urls_are_analyzed_in_order'],
        'ФТ-2.8': ['test_language_warnings'], 'ФТ-3.1': ['test_missing_scheme_only_changes_parsing', 'test_invalid_url'],
        'ФТ-3.2': ['test_normal_https_url_has_all_expected_values'], 'ФТ-3.3': ['test_ipv4', 'test_ipv6'],
        'ФТ-3.4': ['test_normal_https_url_has_all_expected_values'], 'ФТ-3.5': ['test_normal_https_url_has_all_expected_values'],
        'ФТ-3.6': ['test_shortener_list_and_exact_domain_matching'], 'ФТ-3.7': ['test_keyword_occurrences_are_case_insensitive'],
        'ФТ-3.8': ['test_punycode_and_unicode_have_same_masking_flags', 'test_brand_distance_and_exact_brand_subdomain'],
        'ФТ-3.9': ['test_json_serialization_contains_all_required_fields'], 'ФТ-3.10': ['test_exact_xss_payload_history_result_dashboard'],
        'НФТ-1': ['test_runtime_latency_one_hundred_real_requests'], 'НФТ-2': ['test_dashboard_series_one_thousand_rows'],
        'НФТ-4': ['test_sql_injection_input_and_filters'], 'НФТ-5': ['test_exact_xss_payload_history_result_dashboard'],
        'НФТ-6': ['test_forty_requests_enforce_thirty_per_minute'], 'НФТ-9': ['test_sms_and_embedded_url_absent_from_database_and_logs'],
        'НФТ-11': ['test_invalid_sms', 'test_invalid_url'], 'НФТ-12': ['test_binary_or_invalid_json_is_400', 'test_invalid_sms', 'test_invalid_url'],
        'НФТ-14': ['test_missing_and_corrupt_artifacts_return_503'],
    }
    special = {
        'ФТ-2.2': ('FAIL', 'Эвристика возвращает und вместо other для японского текста, а Bonjour определяется как en. ru/kk/en проверены, полное множество языков не реализовано.'),
        'ФТ-2.8': ('FAIL', 'Предупреждения ru/kk есть, но неподдерживаемый латинский язык определяется как en и отдельного предупреждения other нет.'),
        'ФТ-3.11': ('NOT TESTED', 'Необязательная интеграция внешней репутации не реализована; сеть для анализа не требуется.'),
        'НФТ-2': ('NOT TESTED', 'Серверные API/HTML укладываются в 2 с; полная браузерная загрузка с 1000 строк не измерена, MAN-02.'),
        'НФТ-3': ('PASS', 'Реальный двойной клик в Chrome во время анализа дал одну новую запись, browser evidence в 18_environment_results.json.'),
        'НФТ-7': ('FAIL' if environment['working_debug'] else 'PASS', 'Секреты исключены из Git. Рабочий HTTP dev-режим имеет DEBUG=' + str(environment['working_debug']) + '; чистая демонстрационная проверка запускалась с DEBUG=False.'),
        'НФТ-8': ('NOT TESTED', 'Условное требование. Загрузка пользовательских файлов отсутствует.'),
        'НФТ-10': ('NOT TESTED', 'Три точные ширины не проверялись, MAN-01. Снимки реального Chrome подтверждают только текущее окно.'),
        'НФТ-13': ('PASS', 'Проверены RotatingFileHandler, формат логов и ротация в конфигурации; реальные события анализа зарегистрированы без исходного SMS.'),
        'НФТ-14': ('FAIL', 'Деградация 503 и безопасный лог подтверждены, но загрузка ленивая и ошибка появляется при первом анализе, а не при старте.'),
        'НФТ-15': ('PASS' if coverage['totals']['percent_covered'] >= 60 else 'FAIL', 'Точные проценты приведены в 18_coverage.json и итоговом отчёте.'),
    }
    lines = ['# Матрица трассировки требований', '',
        'Исходный PDF содержит коды ФТ-1.1–1.9, ФТ-2.1–2.8, ФТ-3.1–3.11 и НФТ-1–15. Разделы §2.4–2.7 в PDF не имеют отдельных кодов. Прежние локальные ФТ-4.x–7.x из документа задания №1 ниже обозначены ссылками на реальные разделы, чтобы не выдавать их за коды исходного ТЗ.', '',
        'Приоритеты перенесены из 01_requirements_moscow.csv задания №1. PASS требует фактического теста или проверенного артефакта. NOT TESTED означает отсутствие полной проверки, BLOCKED — невозможность проверки без недостающего артефакта, FAIL — обнаруженное несоответствие.', '',
        '| Код исходного ТЗ | Требование | Приоритет | Тест-кейсы | Результат | Комментарий и доказательство |', '|---|---|---|---|---|---|']
    requirements = list(csv.DictReader((DOCS / '01_requirements_moscow.csv').open(encoding='utf-8-sig')))
    for req in requirements:
        code = req['code']
        if req['provenance'] != 'ТЗ':
            continue
        stems = mapping.get(code, [])
        references = ', '.join(ids.get(stem, stem) for stem in stems) or 'Аудит/артефакт'
        references = {'НФТ-3': 'BROW-01', 'НФТ-7': 'ENV-01', 'НФТ-13': 'ENV-02', 'НФТ-15': 'COV-01',
                      'НФТ-10': 'MAN-01', 'НФТ-2': ids['test_dashboard_series_one_thousand_rows'] + ', MAN-02'}.get(code, references)
        if code in {'ФТ-2.2', 'ФТ-2.8'}:
            references += ', ENV-03'
        status, note = special.get(code, ('PASS', 'Подтверждено тестами JUnit; исходники по ссылкам в 18_test_cases.md.'))
        if code not in special and stems:
            matched = [n for n in nodes if n.get('name', '').split('[')[0] in stems]
            if any(n.find('failure') is not None or n.find('error') is not None for n in matched):
                status = 'FAIL'
            elif not all(any(n.get('name', '').split('[')[0] == stem for n in matched) for stem in stems):
                status = 'NOT TESTED'
        if code == 'НФТ-1' and not runtime['sms_100_requests']['passes']:
            status = 'FAIL'
        lines.append('| ' + ' | '.join(map(cell, [code, req['desc'], req['priority'], references, status, note + ' См. 18_pytest.xml, 18_model_results.json, 18_environment_results.json.'])) + ' |')
    extra = [
        ('§2.4, модели', 'Сравнение ≥3 SMS и ≥2 URL алгоритмов, метаданные', 'Must', 'PASS', 'Предыдущие отчёты №8/11 и неизменённые comparison в metadata; новый поиск не выполнялся.'),
        ('§2.4, кеш', 'Однократная загрузка моделей при старте', 'Must', 'FAIL', 'Кеш один на процесс и потокобезопасен, но загрузка при первом запросе. tests/test_analysis_services.py::test_each_final_model_loads_once_even_across_instances_threads.'),
        ('§2.5', 'Risk Score, веса YAML, независимые минимумы, объяснение и рекомендация', 'Must', 'PASS', 'TC-25–28; tests/test_risk.py и risk_config.yaml.'),
        ('§2.6', 'Счётчики, категории, дни, топ-5, последние проверки', 'Must', 'PASS', 'TC-40–44, реальный Chrome и 18_dashboard.png.'),
        ('§2.6, уровни', 'Распределение уровней риска', 'Must', 'FAIL', 'На Dashboard нет отдельного распределения LOW/MEDIUM/HIGH. Функция в задании №17 не добавлялась.'),
        ('§2.7', 'Пагинация, фильтры, детали, удаление и согласие', 'Must', 'PASS', 'TC-31–39 и TC-59–61; подробные связи и privacy проверены тестами.'),
        ('§2.7, экспорт', 'Экспорт собственной истории в CSV', 'Could', 'NOT TESTED', 'Дополнительная функция не реализована.'),
        ('§3.3', 'Split, доменные группы и отсутствие утечки в обучении', 'Must', 'PASS', 'Сохранённые индексы и хеши сверены; tests/test_features.py, test_sms_comparison.py, test_url_dataset.py и test_url_model.py. RU/KK использовался для калибровки Risk в №13, поэтому не является независимым тестом совместной системы.'),
        ('§3.3, доли URL', 'Разделение 70/15/15 с сохранением классов', 'Must', 'FAIL', 'Фактические URL части 32869/8471/8660 из 50000, то есть 65.738/16.942/17.320%. Групповое разделение сохраняет доменную изоляцию, но не точные доли и баланс в частях. Прежний split оставлен без изменений.'),
        ('§3.4, SMS', 'Recall/Precision/F1 ≥0.90', 'Must', 'FAIL', 'Сохранённый артефакт имеет Precision/F1 ниже минимума; 18_model_results.json.'),
        ('§3.4, URL', 'Recall/Precision ≥0.85, ROC-AUC ≥0.90', 'Must', 'PASS', 'Все минимумы выполнены на фиксированном Test, домены не пересекаются.'),
        ('§3.3, ru/kk', 'Recall fraud ≥0.80 для заявления поддержки', 'Must', 'FAIL', 'RU=0.50, KK=0.52 у SMS-классификатора; искусственный малый набор.'),
        ('Задание №13, калибровка', 'Recall medium/high ≥0.95 и ложные HIGH ≤3%', 'Must', 'FAIL', 'Ранее выбранная конфигурация имеет EN Recall 0.923913 и RU/KK 0.48, ложные HIGH 0. Новая калибровка не проводилась.'),
        ('§8.2', 'Пять сценариев и обновление истории/Dashboard', 'Must', 'PASS', 'Реальные HTTP-потоки и браузерные сценарии на копии БД; учебные ожидания B/D/E расходятся с моделями.'),
        ('§8.3', '13 искусственных образцов с фактическими результатами', 'Must', 'PASS', '18_demo_scenarios.md и 18_model_results.json. В №6/8 заменены нерезервированные адреса, изменение явно отмечено.'),
        ('§8.4', 'Итоговая записка 40–60 страниц, Activity/Sequence и все пять экранов', 'Must', 'NOT TESTED', 'Документы этапов имеются, полный оформленный комплект записки не подготовлен, MAN-04.'),
        ('Задание №18, видео', 'Видеозапись 3–5 минут', 'Must', 'NOT TESTED', 'Подготовлен сценарий записи, самого видео нет, MAN-03.'),
        ('Свежая установка', 'Установка и запуск из чистого каталога', 'Must', 'PASS', 'Новое venv, pip check, новый .env и БД, запуск health/login и inference после переноса доверенного bundle. Без bundle модели отсутствуют в Git.'),
        ('Свежий клон, ML без bundle', 'Запуск анализа без отдельно перенесённых артефактов', 'Must', 'BLOCKED', 'joblib/metadata исключены из Git. Для анализа нужны существующие доверенные артефакты; после их переноса inference проверен.'),
    ]
    extra_refs = {'§3.4, SMS': 'QC-01', '§3.4, URL': 'QC-02', '§3.3, ru/kk': 'QC-03, QC-04',
                  '§8.4': 'MAN-04', 'Задание №18, видео': 'MAN-03', 'Задание №13, калибровка': 'QC-05', '§8.2': ids['test_five_real_model_flows_save_history_detail_dashboard']}
    for code, desc, priority, status, note in extra:
        lines.append('| ' + ' | '.join(map(cell, [code, desc, priority, extra_refs.get(code, 'Артефакты №18'), status, note])) + ' |')
    write('18_traceability_matrix.md', lines)

    demo_lines = ['# Демонстрационные сценарии SMS Guard', '', '## 13 образцов §8.3', '',
        'Результаты вычислены текущими сохранёнными моделями с исходными порогами и YAML. Ожидания — гипотезы автора ТЗ. В образце №6 IP 192.0.2.15 заменён на account-update.example, в №8 bit.ly заменён на short.example, чтобы все демонстрационные URL соответствовали пользовательскому ограничению зарезервированных доменов. Эти два образца являются явно обозначенными адаптациями. Оригинальные IP и сокращатели отдельно проверены автотестами.', '',
        '| № | Язык | Искусственный ввод | Ожидание ТЗ | Фактическая категория | Risk Score | Группы / признаки | Сопоставление |', '|---|---|---|---|---|---|---|---|']
    for row in model['demo']:
        result = row['result']
        groups = result.get('features', {}).get('social_engineering_groups', [])
        if row['language'] == 'URL':
            groups = [k for k in ('is_https', 'has_punycode', 'suspicious_word_count', 'brand_similarity') if result.get('features', {}).get(k)]
        values = [row['number'], row['language'], row['input'], '/'.join(row['expected']), result['category'], result.get('risk_score', '—'), ', '.join(groups) or 'нет групп', 'совпало' if row['matches'] else 'РАСХОЖДЕНИЕ']
        demo_lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    demo_lines += ['', '## Пять обязательных сценариев A–E', '', 'Используйте отдельную демонстрационную базу, согласие на сохранение SMS оставьте выключенным. Откройте приложение, войдите, выберите нужную страницу и введите текст вручную. Сервер читает реальные joblib; жёстко заданных ответов нет. Повторные проверки создают отдельные записи после завершения анализа, защита от двойного клика действует на выполняющуюся форму.', '']
    flow_rows = runtime.get('five_real_http_flows', [])
    for letter, number in [('A', 0), ('B', 7), ('C', 5), ('D', 12), ('E', 9)]:
        row = model['demo'][number-1] if number else None
        text = 'Please bring the blue notebook tomorrow.' if letter == 'A' else row['input']
        flow = next((x for x in flow_rows if x['scenario'] == letter), {})
        score_text = flow.get('risk_score') if flow.get('risk_score') is not None else 'не рассчитывается'
        demo_lines += [f'### Сценарий {letter}', '', f'Ввод `{text}`.', '',
            f"1. Открыть {'/check/url/' if letter == 'D' else '/check/sms/'}, вставить ввод и нажать кнопку проверки.",
            '2. Показать индикатор загрузки, фактическую категорию, вероятность и объяснения. Для SMS со ссылкой раскрыть независимый URL-результат.',
            f"3. Зафиксированный результат реального HTTP-потока — {flow.get('category', 'NOT TESTED')}, Risk Score {score_text}.",
            '4. Перейти в /history/, найти новую запись, нажать «Подробнее». Проверить дату, маску и сохранённый результат без повторного вызова ML.',
            '5. Открыть Dashboard за 30 дней и показать увеличение общего количества и соответствующей категории.',
            '6. При расхождении с ожидаемой угрозой объяснить ограничение, сохранить фактический результат. Предварительные ответы не подставлять.', '']
    demo_lines += ['Для B и E реальные модели пропускают угрозу, для D обычный example.com/about отмечен DANGEROUS. Сценарии демонстрируют ограничения, а не гарантированную точность. Реальные браузерные действия и проверка истории/Dashboard отражены в 18_environment_results.json. Полное воспроизведение HTTP-переходов всех пяти сценариев подтверждено отдельным автотестом.', '',
        '## Сценарий резервной записи на 4 минуты 30 секунд', '',
        '| Время | Экран и действия | Пример речи |', '|---|---|---|',
        '| 0:00–0:25 | Вход и Dashboard | «SMS Guard — учебный проект. Результаты вероятностные, корпуса ограничивают перенос качества». |',
        '| 0:25–0:55 | A, безопасное английское SMS | «Ввод проходит через сохранённый Pipeline, показаны вероятность и причины». |',
        '| 0:55–1:30 | B, перевод денег без ссылки | «Здесь модель пропускает угрозу. Русский язык не имеет подтверждённой поддержки». |',
        '| 1:30–2:05 | C, SMS со ссылкой | «Ссылка извлечена автоматически и отдельно оценена URL-моделью». |',
        '| 2:05–2:40 | D, отдельный URL | «Сайт не открывается. Обычный URL получает ложную тревогу из-за смещения корпуса». |',
        '| 2:40–3:10 | E, запрос кода | «Фраза “назовите код из SMS” не покрыта текущим словарём. Это будущая работа». |',
        '| 3:10–3:45 | История, подробный результат и Dashboard | «Проверки сохранены, тексты замаскированы, счётчики обновились». |',
        '| 3:45–4:15 | Новый ввод от комиссии | «Текст отсутствует в сценариях, расчёт идёт в реальном времени». |',
        '| 4:15–4:30 | Выводы | «URL-метрики на корпусе высокие, но SMS Precision/F1 и RU/KK не достигли ТЗ». |', '',
        'Видеофайл не создан. Запишите экран и голос по этому плану, исключите реальные SMS и секреты из кадра. Для устного показа следуйте порядку §8.2, оставляя время на объяснение каждого несовпадения.', '',
        '![Сценарий A](figures/18_scenario_a.png)', '![Пропуск сценария B](figures/18_scenario_b.png)', '![Совместный анализ C](figures/18_scenario_c.png)', '![Пропуск сценария E](figures/18_scenario_e.png)']
    write('18_demo_scenarios.md', demo_lines)

    lines = ['# Задание №18. Комплексное тестирование и демонстрация', '', '## 1. Цель и объект', '',
        f"Проверен существующий SMS Guard, HEAD `{environment['head']}`. До работы изменения заданий №1–17 были сохранены; прежние папки pytest задания №16 оставлены локально. Архитектура, модели, разбиения, Risk Score, правила и миграции не менялись. Новые возможности приложения не добавлялись.", '',
        '## 2. Окружение', '', f"Python {environment['python']}, Windows, Django 5.2.8, scikit-learn 1.6.1, SQLite. Версии зависимостей перечислены в requirements.txt. Исходная рабочая база сохранена отдельно. Тесты используют Django test DB и уникальные каталоги системного TEMP.", '',
        '## 3. Трассировка и тест-кейсы', '', f"Подготовлено {len(cases)} автокейсов, {len(audits)} проверок качества/аудита и четыре ручных проверки NOT TESTED в [18_test_cases.md](18_test_cases.md). Все исходные коды ФТ/НФТ и не имеющие кодов разделы включены в [матрицу](18_traceability_matrix.md).", '',
        '## 4. Полный pytest', '', f"PASSED {counts['PASSED']}, FAILED {counts['FAILED']}, ERROR {counts['ERROR']}, SKIPPED {counts['SKIPPED']}. [JUnit XML](18_pytest.xml). Сначала 401 прежний тест прошёл отдельно; затем добавлены содержательные проверки безопасности, надёжности, реальных HTTP-потоков и подготовительных модулей. Прежние ожидания не изменялись. Предупреждения Matplotlib/Pyparsing о deprecated API отражены в журнале pytest и не являются FAILED/ERROR.", '',
        '## 5. Покрытие строк', '', '| Модуль | Покрыто / всего | Точное покрытие |', '|---|---|---|']
    for name, entry in coverage['files'].items():
        s = entry['summary']
        lines.append(f"| {name} | {s['covered_lines']} / {s['num_statements']} | {s['percent_covered']:.6f}% |")
    s = coverage['totals']
    lines += [f"| Итого ml/, включая risk | {ml_covered} / {ml_total} | {100 * ml_covered / ml_total:.6f}% |",
        f"| Итого с сервисами анализа | {s['covered_lines']} / {s['num_statements']} | {s['percent_covered']:.6f}% |", '',
        'Отдельно SMS preprocessing — preprocessing.py, SMS classifier — baseline.py и compare_sms.py, URL classifier — url_model.py, URL features — url_features.py, общий predictor — predictor.py, rules/risk — одноимённые модули. Полный исходный [coverage JSON](18_coverage.json) содержит строки. В покрытие входят функции драйверов обучения, но повторные production fit/search/calibrate в этом задании не запускались. Существующие unit-тесты создают малые синтетические объекты в памяти; новые проверки CLI используют готовую модель или временные данные.', '',
        '## 6. Безопасность', '',
        'SQL Injection проверена точной строкой `\' OR 1=1 --` в SMS и фильтрах истории, чужие записи не раскрыты. XSS с точным `<script>alert(1)</script>` проверен экранированием истории, отсутствием текста в Dashboard и безопасными DOM-операциями. Браузерная проверка фактического XSS описана в environmental evidence. CSRF проверен Django Client(enforce_csrf_checks=True). Подмена ID не даёт просмотреть, удалить чужую запись или удалить чужую историю. Перебор ограничен после пяти неудач минимум на 60 с, подтверждён тестовыми часами.', '',
        'HttpOnly/SameSite=Lax подтверждены, Secure проверен для HTTPS в тесте. Локальный HTTP использует Secure=False, что соответствует этому режиму. Это не проверка реально развёрнутого TLS. .env исключён из Git; реальные секреты в артефакты отчёта не копируются. DEBUG текущей рабочей dev-конфигурации отражён в матрице отдельно от чистого запуска DEBUG=False.', '',
        '## 7. Надёжность', '',
        'Проверены пустой/пробельный ввод, 1000/1001 символов, бинарный и невалидный UTF-8 JSON, неправильные схемы/hostname, отсутствие домена, IP/Punycode, SMS без/с несколькими URL. Потеря каждого файла модели и metadata и повреждённый JSON моделировались только на временных копиях. Health остаётся 200 с доступностью, анализ отдаёт 503, записей не остаётся. Рабочие файлы не переименовывались. Внешняя репутация URL не используется, поэтому её сбой неприменим; автономность собственного анализа подтверждена запретом сетевых вызовов в тесте.', '',
        'Модели загружаются лениво один раз на процесс, после неуспешной загрузки для восстановления нужен перезапуск. Буквальное требование сообщения об отсутствующей модели при старте не выполнено, хотя безопасная деградация первого анализа подтверждена.', '',
        '## 8. Производительность', '', '| Проверка | Запросов | Среднее, мс | Максимум, мс | Результат |', '|---|---|---|---|---|']
    latency = runtime['sms_100_requests']
    lines.append(f"| НФТ-1, реальное SMS с 3 URL, API и сохранение | 100 | {latency['mean_ms']:.6f} | {latency['max_ms']:.6f} | {'PASS' if latency['passes'] else 'FAIL'}, предел 500 мс |")
    for key, value in runtime['dashboard_1000_rows'].items():
        lines.append(f"| НФТ-2, серверный {key}, 1000 строк | {value['requests']} | {value['mean_ms']:.6f} | {value['max_ms']:.6f} | {'PASS' if value['passes'] else 'FAIL'} серверной части |")
    rate = runtime['rate_limit_40_requests']
    lines.append(f"| НФТ-6, фиксированное тестовое время | 40 | {rate['mean_ms']:.6f} | {rate['max_ms']:.6f} | 30 принятых, 10 ответов 429 |")
    click_ms = environment['browser']['double_click']['server_analysis_ms']
    lines.append(f"| НФТ-3, реальный двойной клик, серверный лог | 2 клика / 1 запрос | {click_ms:.6f} | {click_ms:.6f} | Одна новая запись, N=1 |")
    lines += ['', f"Первый прогрев перед 100 запросами занял {latency['warmup_ms']:.6f} мс и показан отдельно. Модели реальные, внешних API нет. Чтобы не отключать throttle для серии, между запросами моделировалось 3 секунды на часах только CheckThrottle. Измерялись настоящие длительности каждого запроса через perf_counter, не виртуальное время. Ограничение 30/мин проверено отдельно по 40 запросам в одном интервале.", '',
        'НФТ-3 проверен реальным двойным кликом Chrome, счётчик копии БД вырос с 4 до 5, результат A один. Сервер не предоставляет idempotency key для произвольных повторных HTTP-запросов после завершения формы. Среднее/максимум обработчика двойного клика в браузере не измерялись; наблюдались два клика и одна сохранённая запись. Полная загрузка Dashboard с сетью/CDN/canvas при 1000 строк NOT TESTED; серверный HTML-каркас не выдаётся за браузерное время.', '',
        '## 9. Итоговые метрики моделей', '', 'Порог SMS и оба URL-порога неизменны. SMS Test содержит 771 строку из прежнего sms_split.json, URL Test — 8660 из url_split.json. Доменные группы пересчитаны из URL и не пересекаются, SHA-256 Parquet и split совпадают с metadata. Фактические URL Train/Validation/Test составляют 32869/8471/8660, то есть 65.738/16.942/17.320% вместо буквальных 70/15/15, как следствие группового разделения; баланс классов между частями также различается. Это отмечено отдельно от успешно подтверждённого отсутствия доменной утечки. Test уже оценивался в прежних заданиях, нынешний прогон — аудит того же Test, не новая независимая выборка. Подбора по нему не было.', '',
        f"Бинарная метка SMS равна 1 при p_sms ≥ {model['sms']['threshold']}. Для URL бинарный phishing=1 означает p_url ≥ t2={model['url']['t2']}; зона SUSPICIOUS между t1={model['url']['t1']} и t2 остаётся ниже этой бинарной границы.", '',
        '| Метрика | Сохранённая SMS-модель | Сохранённая URL-модель |', '|---|---|---|']
    for key in ['accuracy', 'precision', 'recall', 'f1', 'macro_precision', 'macro_recall', 'macro_f1', 'roc_auc', 'pr_auc', 'f2']:
        lines.append(f"| {key} | {model['sms']['saved_model'][key]:.9f} | {model['url']['metrics'][key]:.9f} |")
    lines += ['', f"SMS Confusion Matrix [[TN,FP],[FN,TP]] — `{model['sms']['saved_model']['confusion_matrix']}`.",
        f"URL Confusion Matrix — `{model['url']['metrics']['confusion_matrix']}`.", '',
        f"Прежняя SMS metadata содержит метрики train-only до refit — `{model['sms']['historical_pre_refit']['confusion_matrix']}`. Они не приписываются артефакту, обученному затем на 4364 Train+Validation. Текущая оценка выполнена для этого сохранённого артефакта, порог {model['sms']['threshold']}. URL-модель сохранена Train-only, повторные метрики совпали с metadata.", '',
        f"SMS удовлетворяет Recall ≥0.90, но Precision/F1 ниже 0.90 — **FAIL**. URL удовлетворяет Precision/Recall ≥0.85 и ROC-AUC ≥0.90 — **PASS** на этом корпусе. PR-AUC рассчитан как average_precision_score (AP без интерполяции). SMS spam является прокси и не подтверждает мошенничество.", '',
        f"Время одного SMS predict_proba, 100 повторов после прогрева — среднее {model['sms']['latency']['mean_ms']:.6f}, максимум {model['sms']['latency']['max_ms']:.6f} мс. URL на одной строке готовых числовых признаков — среднее {model['url']['latency_numeric_features']['mean_ms']:.6f}, максимум {model['url']['latency_numeric_features']['max_ms']:.6f} мс. Извлечение URL-признаков в последнюю оценку не входит.", '',
        '## 10. Русский и казахский языки', '', '| Язык | SMS | Precision | Recall | F1 | CM | Recall совместной системы non-SAFE / FRAUDULENT |', '|---|---|---|---|---|---|---|']
    for lang, value in model['languages'].items():
        m = value['classifier']
        lines.append(f"| {lang} | {value['rows']} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['confusion_matrix']} | {value['joint_recall_non_safe']:.6f} / {value['joint_recall_fraudulent']:.6f} |")
    lines += ['', 'Оба SMS Recall ниже 0.80, поддержка RU/KK не подтверждена. Набор искусственный, малый и вручную проверен ранее. Для SMS-классификатора он не обучающий, но в №13 использовался при подборе весов Risk, поэтому показатели совместного сервиса на нём являются development-оценкой. Английский корпус не покрывает русскую/казахскую лексику. Полные дополнительные метрики в 18_model_results.json.', '',
        '## 11. Фактические ошибки моделей', '',
        'Для каждой модели приведены первые пять FP и пять FN фиксированного Test. Тексты SMS взяты из публичного UCI, цифры заменены *, адреса скрыты. URL полностью обезличены зарезервированным hostname, сохранены исходный source_id, test_position и измеренные признаки для проверки на локальном корпусе. Причины ниже являются гипотезами, выводимыми из ограничений корпуса, и не доказанными причинными объяснениями.', '',
        '| Модель / тип | ID источника | Язык | Безопасный фрагмент | Истина → прогноз | P(1) | Возможная причина | Будущая работа |', '|---|---|---|---|---|---|---|---|']
    for row in model['errors']:
        reason = ('Смещение к spam при низком пороге, короткий разговорный текст' if row['kind'] == 'FP' else 'Необычная лексика spam, близкая к обычной переписке') if row['model'] == 'SMS' else 'Артефакты корпуса HTTPS/path/query, ограниченный перенос на новые домены'
        future = 'Расширить независимый корпус и заново валидировать порог/признаки' if row['model'] == 'SMS' else 'Собрать независимые URL без форматного смещения и повторить групповую оценку'
        lines.append('| ' + ' | '.join(map(cell, [row['model'] + '/' + row['kind'], row['source_id'], row['language'] or '—', row['input'][:210], f"{row['true']} → {row['predicted']}", f"{row['probability']:.9f}", reason, future])) + ' |')
    lines += ['', 'Ранее сохранённая калибровка №13 на train-only реконструкции дала Recall medium/high EN Validation 0.923913 и RU/KK development 0.48 при нулевых ложных HIGH. Цель Recall ≥0.95 не достигнута. Это исторические результаты фиксированной калибровки, их нельзя выдавать за новую оценку конечного сервиса. Правила, YAML и калибровка в №18 не изменялись.', '', '## 12. Известные недостатки и предложения', '',
        '- Русские переводы денег и запросы кодов могут получать SAFE, как подтверждают сценарии B/E. Расширение многоязычного корпуса и словаря — будущая работа.',
        '- «Назовите код из SMS» и «Продиктуйте код из сообщения» не дают CODE_REQUEST. «Сообщите код подтверждения» срабатывает, но отрицание «Никому не сообщайте код подтверждения» тоже срабатывает. Не хватает формулировок, морфологии и контекстной обработки отрицаний.',
        f"- `https://www.python.org/downloads/` без открытия сайта получил {model['diagnostics']['python_url']['category']}, p_url={model['diagnostics']['python_url']['p_url']:.9f}. Это диагностический предполагаемо безопасный URL, отдельный от Test. Высокие метрики PhiUSIIL не устраняют смещение HTTPS/path/query. Изменение данных/моделей в этой работе не проводилось.",
        '- Требования other-языка, распределения уровней риска Dashboard и раннего сообщения о модели при старте полностью не выполнены. Новые функции ради закрытия матрицы не добавлялись.',
        '- Извлечение голого short.example/path без схемы даёт пустой список из-за проверки суффикса. Поэтому адаптированный демонстрационный short.example записан с http://; сокращатели и реальные домены без схемы проверяются отдельно.',
        '- Сравнение spam/ham в UCI не равно подтверждённому fraud/safe. Вероятности не являются гарантией безопасности.', '',
        '## 13. Демонстрация и материалы защиты', '', '[13 фактических образцов, пять сценариев и план видео](18_demo_scenarios.md). Браузер проверен на отдельной копии БД и реальных артефактах. Несовпадения с ожиданиями сохранены. План выступления готов; видео, три точные ширины и полная записка 40–60 страниц остаются ручными действиями. Это не полное выполнение §8.4.', '',
        '## 14. Свежая установка', '',
        'Создана отдельная копия только отслеживаемых файлов, новое venv без system-site-packages, зависимости установлены из requirements.txt. Первая попытка упала с WinError 5 на стандартном pip wheel cache; повтор с отдельным --cache-dir прошёл без прав администратора. pip check успешен. Созданы новый .env и пустая БД, все миграции применены, check и health/login/redirect успешны. После копирования четырёх существующих доверенных ML-файлов в новое окружение реальное предсказание SMS/URL совпало с рабочим.', '',
        'Git не содержит joblib/metadata и больших данных. Полная ML-функция после одного лишь клонирования BLOCKED до переноса существующего bundle. README теперь описывает это условие, команды подготовки данных и перенос артефактов без повторного обучения. Обучение для проверки свежей установки не выполнялось.', '',
        '## 15. Целостность и финальные проверки', '',
        'Результаты check, makemigrations, SQLite integrity_check, сравнение таблиц и SHA-256 защищённых файлов приведены в 18_environment_results.json. .env, базы, кеши и pytest temp не добавляются в Git; отчёты не содержат реальных пользовательских SMS и паролей. Секреты в JUnit/coverage не выводятся, но абсолютные пути окружения в технических XML/JSON возможны.', '',
        '## 16. Вывод', '',
        'Проект пригоден для учебного показа работы существующей архитектуры с открытым обсуждением ограничений. Полное соответствие ТЗ и готовность к практическому выявлению мошенничества не подтверждены. Перед защитой следует вручную проверить точные разрешения, записать видео, собрать итоговую записку и продемонстрировать известные FP/FN. Коммит и push оставлены пользователю после ручной проверки.', '',
        '## Приложение. Диаграммы фактического процесса', '',
        'ER приведена в docs/16_database_history.md, Use Case — в docs/01_domain_analysis.md. Ниже Activity и Sequence отражают существующий код. Полная вёрстка пояснительной записки остаётся отдельным ручным действием.', '',
        '```mermaid', 'flowchart TD', '    A[Ввод SMS в браузере] --> B[Сессия, CSRF, throttle и валидация]',
        '    B --> C[Извлечение до трёх URL и определение языка]',
        '    C --> D[Сохранённый SMS Pipeline и URL classifier]',
        '    D --> E[Группы правил и Risk Score из YAML]',
        '    E --> F[Атомарное сохранение с учётом согласия]',
        '    F --> G[JSON и безопасный вывод результата]',
        '    G --> H[История и Dashboard]', '```', '',
        '```mermaid', 'sequenceDiagram', '    actor U as Пользователь', '    participant B as Браузер',
        '    participant V as Django/DRF views', '    participant S as Сервис анализа', '    participant M as ModelPredictor', '    participant DB as SQLite',
        '    U->>B: Ввод и нажатие кнопки', '    B->>V: POST SMS, сессия и CSRF', '    V->>V: Проверка лимита и сериализатор',
        '    V->>S: analyze_sms(text)', '    S->>M: predict_sms и до трёх predict_url_features', '    M-->>S: Вероятности из кешированных моделей',
        '    S->>S: Правила и compute_risk', '    S-->>V: AnalysisResult', '    V->>DB: transaction.atomic, Check и дочерние записи',
        '    DB-->>V: ID и дата', '    V-->>B: HTTP 201 и результат', '    B-->>U: textContent и объяснения',
        '    U->>B: История или Dashboard', '    B->>V: GET защищённого представления', '    V->>DB: ORM выборка и агрегаты',
        '    DB-->>V: Сохранённые данные', '    V-->>B: HTML или JSON', '```']
    write('18_final_testing.md', lines)
    print(f'Generated reports from {len(nodes)} tests and {len(cases)} documented cases.')


if __name__ == '__main__':
    main()
