"""Dashboard использует отдельную тестовую БД, без вызовов ML."""
from datetime import datetime, time, timedelta, timezone as dt_timezone
import json
from time import perf_counter
from unittest.mock import Mock

from django.contrib.auth import get_user_model
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone
import pytest

from apps.analysis.models import Check, DetectedFeature, ModelVersion
from apps.analysis.services import AnalysisResult

pytestmark = pytest.mark.django_db
ENDPOINT = '/api/v1/stats/summary/'


@pytest.fixture
def owner(client, monkeypatch):
    now = datetime(2026, 11, 14, 12, tzinfo=dt_timezone.utc)
    monkeypatch.setattr(timezone, 'now', lambda: now)
    user = get_user_model().objects.create_user('dashboard_owner')
    client.force_login(user)
    return user


@pytest.fixture
def make_check(owner):
    def create(user=None, days=0, category='SAFE', kind='sms', score=10):
        check = Check.objects.create(user=user or owner, type=kind, category=category,
            risk_level='LOW' if score is not None else None, risk_score=score,
            masked_input='***', input_text='PRIVATE_SMS_SECRET_782341',
            result_json={'text': 'PRIVATE_JSON_SECRET_893452'})
        day = timezone.localdate() - timedelta(days=days)
        created = timezone.make_aware(datetime.combine(day, time(12)))
        Check.objects.filter(pk=check.pk).update(created_at=created)
        check.created_at = created
        return check
    return create


def test_page_requires_login(client):
    response = client.get('/')
    assert response.status_code == 302 and response.url == '/login/?next=/'


def test_page_after_login(client, owner):
    response = client.get('/')
    assert response.status_code == 200
    html = response.content.decode()
    for value in ['Dashboard', 'dashboard.js', 'chart.js@4.4.8', 'categories-chart', 'daily-chart', 'features-chart', 'Пока нет проверок']:
        assert value in html
    assert 'Активные версии моделей' not in html


def test_api_requires_login(client):
    assert client.get(ENDPOINT).status_code == 401


def test_cards_categories_and_null_average(client, owner, make_check):
    make_check(category='SAFE', score=0)
    make_check(category='SUSPICIOUS', score=50)
    make_check(category='FRAUDULENT', score=100)
    make_check(category='DANGEROUS', kind='url', score=None)
    data = client.get(ENDPOINT).json()
    assert (data['sms_count'], data['url_count']) == (3, 1)
    assert (data['safe_count'], data['suspicious_count'], data['dangerous_count']) == (1, 1, 2)
    assert data['average_risk_score'] == 50
    assert data['categories'] == {'SAFE': 1, 'SUSPICIOUS': 1, 'FRAUDULENT': 1, 'DANGEROUS': 1}
    assert [r['count'] for r in data['category_distribution']] == [1, 1, 2]
    assert data['period'] == 30


def test_average_all_null(client, owner, make_check):
    make_check(kind='url', score=None)
    assert client.get(ENDPOINT).json()['average_risk_score'] is None


def test_daily_real_dates_and_zero_days(client, owner, make_check):
    make_check(days=2, category='FRAUDULENT')
    make_check(days=2, category='DANGEROUS', kind='url', score=None)
    make_check(category='SAFE')
    daily = client.get(ENDPOINT + '?period=7').json()['daily']
    assert len(daily) == 7
    assert daily[-3] == {'date': '2026-11-12', 'SAFE': 0, 'SUSPICIOUS': 0, 'DANGEROUS': 2}
    assert daily[-2] == {'date': '2026-11-13', 'SAFE': 0, 'SUSPICIOUS': 0, 'DANGEROUS': 0}
    assert daily[-1]['SAFE'] == 1


def test_top_five_actual_groups_and_distinct_checks(client, owner, make_check):
    codes = ['PRIZE', 'URGENCY', 'BANK_CARD', 'ACCOUNT_BLOCK', 'MONEY_TRANSFER', 'CODE_REQUEST']
    for count, code in enumerate(codes, start=1):
        for _ in range(count):
            check = make_check()
            DetectedFeature.objects.create(check_record=check, code=code, title='Ignored title')
    check = Check.objects.first()
    DetectedFeature.objects.create(check_record=check, code='CODE_REQUEST', title='Duplicate group')
    other = get_user_model().objects.create_user('feature_other')
    foreign = make_check(user=other)
    DetectedFeature.objects.create(check_record=foreign, code='PRIZE', title='Foreign')
    old = make_check(days=31)
    DetectedFeature.objects.create(check_record=old, code='PRIZE', title='Old')
    data = client.get(ENDPOINT).json()['top_features']
    assert [r['code'] for r in data] == list(reversed(codes[1:]))
    assert [r['count'] for r in data] == [6, 5, 4, 3, 2]
    assert data[0]['title'] == 'запрос кодов и паролей'


@pytest.mark.parametrize('period', [7, 30, 90])
def test_period_affects_all_sections(client, owner, make_check, period):
    inside = make_check(days=period - 1)
    outside = make_check(days=period)
    for check in [inside, outside]:
        DetectedFeature.objects.create(check_record=check, code='URGENCY', title='Urgency')
    data = client.get(ENDPOINT, {'period': period}).json()
    assert data['period'] == period and len(data['daily']) == period
    assert data['sms_count'] == 1 and data['safe_count'] == 1
    assert sum(day['SAFE'] for day in data['daily']) == 1
    assert data['top_features'][0]['count'] == 1
    assert [r['id'] for r in data['recent_checks']] == [inside.pk]


@pytest.mark.parametrize('period', ['', '0', '-7', '8', 'all', '7.0', '030', ' 7', '91'])
def test_invalid_period_rejected(client, owner, period):
    assert client.get(ENDPOINT, {'period': period}).status_code == 400
    assert client.get('/', {'period': period}).status_code == 400


def test_timezone_and_boundary(client, owner, make_check):
    with timezone.override('Asia/Qyzylorda'):
        inside = make_check()
        outside = make_check()
        future = make_check()
        # Local midnight 08.11 is 07.11 19:00 UTC in this timezone.
        start = timezone.make_aware(datetime(2026, 11, 8))
        Check.objects.filter(pk=inside.pk).update(created_at=start)
        Check.objects.filter(pk=outside.pk).update(created_at=start - timedelta(microseconds=1))
        Check.objects.filter(pk=future.pk).update(created_at=timezone.make_aware(datetime(2026, 11, 15)))
        data = client.get(ENDPOINT + '?period=7').json()
    assert data['timezone'] == 'Asia/Qyzylorda'
    assert data['sms_count'] == 1 and data['daily'][0]['SAFE'] == 1


def test_recent_ten_no_foreign_or_raw_input(client, owner, make_check):
    mine = [make_check() for _ in range(12)]
    other = get_user_model().objects.create_user('foreign_user')
    make_check(user=other)
    response = client.get(ENDPOINT, {'user': other.pk, 'is_staff': 'true', 'scope': 'all_users'})
    data = response.json()
    assert data['scope'] == 'current_user' and data['sms_count'] == 12
    assert [r['id'] for r in data['recent_checks']] == [c.pk for c in reversed(mine[-10:])]
    assert all(r['detail_url'] == f"/checks/{r['id']}/" for r in data['recent_checks'])
    assert b'PRIVATE_' not in response.content and data['active_models'] == []
    assert 'input_text' not in response.content.decode() and 'password' not in response.content.decode()


def test_admin_all_aggregates_without_foreign_detail_access(client, owner, make_check):
    owner.is_staff = True
    owner.save()
    own = make_check()
    other = get_user_model().objects.create_user('another_owner')
    foreign = make_check(user=other, category='FRAUDULENT', score=90)
    DetectedFeature.objects.create(check_record=foreign, code='PRIZE', title='Prize')
    data = client.get(ENDPOINT).json()
    assert data['scope'] == 'all_users' and data['sms_count'] == 2
    assert data['average_risk_score'] == 50 and data['dangerous_count'] == 1
    assert data['daily'][-1]['DANGEROUS'] == 1
    assert data['top_features'][0]['code'] == 'PRIZE'
    assert data['recent_checks'][0]['detail_url'] is None
    assert data['recent_checks'][1]['detail_url'] == f'/checks/{own.pk}/'
    assert client.get(f'/checks/{foreign.pk}/').status_code == 404
    assert {r['model_type'] for r in data['active_models']} == {'sms', 'url'}
    assert 'Активные версии моделей' in client.get('/').content.decode()
    assert 'PRIVATE_' not in json.dumps(data)


def test_admin_reads_active_registry_metrics(client, owner):
    owner.is_staff = True
    owner.save()
    for kind in ('sms', 'url'):
        ModelVersion.objects.create(model_type=kind, version='saved', algorithm='RecordedAlgorithm',
            metrics_json={'f1': .75, 'password': 'SECRET_METADATA'}, is_active=True)
    ModelVersion.objects.create(model_type='sms', version='old', algorithm='Old', is_active=False)
    response = client.get(ENDPOINT)
    data = response.json()['active_models']
    assert len(data) == 2 and all(r['version'] == 'saved' for r in data)
    assert all(r['metrics'] == {'f1': .75} and r['trained_at'] is None for r in data)
    assert b'SECRET_METADATA' not in response.content


def test_empty_database(client, owner):
    data = client.get(ENDPOINT).json()
    assert data['sms_count'] == data['url_count'] == data['dangerous_count'] == 0
    assert data['average_risk_score'] is None
    assert data['top_features'] == data['recent_checks'] == []
    assert all(day[key] == 0 for day in data['daily'] for key in ['SAFE', 'SUSPICIOUS', 'DANGEROUS'])


def test_updated_after_new_api_check(client, owner, monkeypatch):
    analyze = Mock(return_value=AnalysisResult(category='SAFE', risk_level='LOW', risk_score=10, p_sms=.1))
    monkeypatch.setattr('apps.analysis.services.analyze_sms', analyze)
    assert client.get(ENDPOINT).json()['sms_count'] == 0
    response = client.post('/api/v1/check/sms/', {'text': 'New artificial test SMS'})
    assert response.status_code == 201
    assert client.get(ENDPOINT).json()['sms_count'] == 1
    analyze.assert_called_once()
    assert client.get('/history/').status_code == 200
    assert client.get(f"/checks/{response.json()['id']}/").status_code == 200


def test_performance_1000(client, owner, capsys):
    checks = [Check(user=owner, type='sms', category=Check.CATEGORIES[i % 4],
        risk_score=i % 101, masked_input='Artificial performance sample') for i in range(1000)]
    Check.objects.bulk_create(checks)
    DetectedFeature.objects.bulk_create([DetectedFeature(check_record=c, code='URGENCY', title='Urgency') for c in checks])
    with CaptureQueriesContext(connection) as queries:
        started = perf_counter()
        response = client.get(ENDPOINT)
        api_time = perf_counter() - started
    query_count = len(queries)
    sql = '\n'.join(query['sql'] for query in queries)
    assert 'GROUP BY' in sql
    assert 'input_text' not in sql and 'masked_input' not in sql and 'result_json' not in sql
    started = perf_counter()
    page = client.get('/')
    page_time = perf_counter() - started
    assert response.status_code == page.status_code == 200
    assert response.json()['sms_count'] == 1000
    assert len(response.json()['recent_checks']) == 10
    assert response.json()['top_features'][0]['count'] == 1000
    assert 1 <= query_count <= 8
    with capsys.disabled():
        print(f'\nDashboard 1000 records: API={api_time:.6f}s, HTML={page_time:.6f}s, API queries={query_count}')
    assert api_time < 2 and page_time < 2


def test_js_safe_rendering_and_error_fallback():
    from django.conf import settings
    source = (settings.BASE_DIR / 'static/js/dashboard.js').read_text(encoding='utf-8')
    assert 'innerHTML' not in source
    for value in ['textContent', 'chart.destroy()', 'dashboard-loading', 'dashboard-error', 'dashboard-retry', 'Библиотека графиков недоступна']:
        assert value in source
