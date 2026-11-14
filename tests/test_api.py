"""API проверяется с настоящими сессиями/CSRF и подменой только ML-сервиса."""
from datetime import timedelta
import json
from unittest.mock import Mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import connection
from django.db import IntegrityError, transaction
from django.utils import timezone
import pytest
from rest_framework.test import APIClient

from apps.analysis import services, views
from apps.analysis.models import Check
from apps.analysis.services import AnalysisResult, URLAnalysis

pytestmark = pytest.mark.django_db
BASE = '/api/v1/'
PASSWORD = 'GoodPassword29'


def csrf(client):
    token = client.get(BASE + 'health/').json()['csrf_token']
    client.credentials(HTTP_X_CSRFTOKEN=token)


@pytest.fixture(autouse=True)
def isolated_cache():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def user():
    return get_user_model().objects.create_user('alice', 'alice@example.org', PASSWORD)


@pytest.fixture
def anonymous():
    client = APIClient(enforce_csrf_checks=True)
    csrf(client)
    return client


@pytest.fixture
def api(anonymous, user):
    assert anonymous.post(BASE + 'auth/login/', {'username': user.username, 'password': PASSWORD}, format='json').status_code == 200
    csrf(anonymous)  # login rotates the token
    return anonymous


@pytest.fixture(autouse=True)
def fake_services(monkeypatch):
    sms = Mock(return_value=AnalysisResult(language='en', category='SAFE', risk_level='LOW',
        risk_score=10, p_sms=0.1, explanation=['Вероятностная оценка.'],
        recommendation='Проверяйте отправителя.', model_versions={'sms': 'test', 'url': 'test'}))
    url = Mock(side_effect=lambda value: URLAnalysis(url=value, p_url=0.7,
        category='DANGEROUS', features={'is_https': 1}, explanation=['Оценка URL.']))
    monkeypatch.setattr(services, 'analyze_sms', sms)
    monkeypatch.setattr(services, 'analyze_url', url)
    return sms, url


def post_sms(api, text='Please bring the notebook tomorrow.'):
    return api.post(BASE + 'check/sms/', {'text': text}, format='json')


def create_check(user, **kwargs):
    values = dict(user=user, type='sms', category='SAFE', risk_level='LOW', risk_score=10,
                  masked_input='•••', result_json={})
    values.update(kwargs)
    check = Check.objects.create(**values)
    check.result_json = {'id': check.pk, 'type': check.type, 'category': check.category,
                         'risk_score': check.risk_score, 'created_at': check.created_at.isoformat()}
    check.save(update_fields=['result_json'])
    return check


def test_register_hashes_password(anonymous):
    response = anonymous.post(BASE + 'auth/register/',
        {'username': 'new_user', 'email': 'new@example.org', 'password': PASSWORD}, format='json')
    assert response.status_code == 201
    registered = get_user_model().objects.get(username='new_user')
    assert registered.check_password(PASSWORD)
    assert registered.password != PASSWORD
    assert 'password' not in response.json()


@pytest.mark.parametrize('field,value', [('username', 'alice'), ('email', 'ALICE@example.org')])
def test_duplicate_registration(anonymous, user, field, value):
    data = {'username': 'new_user', 'email': 'new@example.org', 'password': PASSWORD}
    data[field] = value
    response = anonymous.post(BASE + 'auth/register/', data, format='json')
    assert response.status_code == 400
    assert response.json()['error']['code'] == 'VALIDATION_ERROR'
    assert get_user_model().objects.count() == 1


@pytest.mark.parametrize('password', ['short1', 'onlyletters', '123456789'])
def test_weak_password(anonymous, password):
    assert anonymous.post(BASE + 'auth/register/', {'username': 'valid', 'email': 'x@example.org',
        'password': password}, format='json').status_code == 400


@pytest.mark.parametrize('username', ['ab', 'a' * 31, 'имя', "x' OR 1=1", '<script>'])
def test_invalid_username(anonymous, username):
    assert anonymous.post(BASE + 'auth/register/', {'username': username, 'email': 'x@example.org',
        'password': PASSWORD}, format='json').status_code == 400


def test_login_creates_session(api, user):
    assert api.session['_auth_user_id'] == str(user.pk)
    assert api.get(BASE + 'checks/').status_code == 200
    cookie = api.cookies['sessionid']
    assert cookie['httponly'] and cookie['samesite'] == 'Lax'


def test_bad_login_is_generic(anonymous, user):
    known = anonymous.post(BASE + 'auth/login/', {'username': 'alice', 'password': 'bad'}, format='json')
    unknown = anonymous.post(BASE + 'auth/login/', {'username': 'unknown', 'password': 'bad'}, format='json')
    assert known.status_code == unknown.status_code == 401
    assert known.json() == unknown.json()


def test_five_failures_lock_for_sixty_seconds(anonymous, user, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr('django.core.cache.backends.base.time.time', lambda: now[0])
    for _ in range(5):
        assert anonymous.post(BASE + 'auth/login/', {'username': 'alice', 'password': 'bad'}, format='json').status_code == 401
    assert anonymous.post(BASE + 'auth/login/', {'username': 'alice', 'password': PASSWORD}, format='json').status_code == 429
    now[0] += 59
    assert anonymous.post(BASE + 'auth/login/', {'username': 'alice', 'password': PASSWORD}, format='json').status_code == 429
    now[0] += 2
    assert anonymous.post(BASE + 'auth/login/', {'username': 'alice', 'password': PASSWORD}, format='json').status_code == 200


def test_successful_login_resets_failures(anonymous, user):
    for _ in range(4):
        anonymous.post(BASE + 'auth/login/', {'username': 'alice', 'password': 'bad'}, format='json')
    assert anonymous.post(BASE + 'auth/login/', {'username': 'alice', 'password': PASSWORD}, format='json').status_code == 200
    csrf(anonymous)
    anonymous.post(BASE + 'auth/logout/')
    for _ in range(5):
        assert anonymous.post(BASE + 'auth/login/', {'username': 'alice', 'password': 'bad'}, format='json').status_code == 401


def test_logout_destroys_session(api):
    assert api.post(BASE + 'auth/logout/').status_code == 204
    assert '_auth_user_id' not in api.session
    assert api.get(BASE + 'checks/').status_code == 401


@pytest.mark.parametrize('path,method', [('checks/', 'get'), ('checks/1/', 'get'),
    ('checks/1/', 'delete'), ('stats/summary/', 'get'), ('check/sms/', 'post'),
    ('check/url/', 'post'), ('auth/logout/', 'post')])
def test_unauthenticated_is_401(anonymous, path, method):
    assert getattr(anonymous, method)(BASE + path).status_code == 401


def test_sms_result_and_atomic_persistence(api, user, fake_services):
    response = post_sms(api)
    assert response.status_code == 201
    result = response.json()
    assert result['risk_score'] == 10 and result['p_sms'] == 0.1
    assert result['language'] == 'en' and result['model_versions']['sms'] == 'test'
    check = Check.objects.get(pk=result['id'])
    assert check.user == user and check.created_at.isoformat() == result['created_at']
    assert check.result_json['id'] == check.pk
    fake_services[0].assert_called_once_with('Please bring the notebook tomorrow.')


@pytest.mark.parametrize('text', ['', '  \n\t', 'x' * 1001, 42, True, [], None, '\x00\x01'])
def test_invalid_sms(api, text, fake_services):
    response = post_sms(api, text)
    assert response.status_code == 400
    fake_services[0].assert_not_called()
    assert not Check.objects.exists()


def test_control_characters_without_nlp_normalization(api, fake_services):
    assert post_sms(api, 'URGENT\x00 CODE\tToday\nKeep Case!').status_code == 201
    fake_services[0].assert_called_once_with('URGENT CODE Today Keep Case!')


def test_url_result_has_actual_probability_and_no_invented_risk(api, fake_services):
    response = api.post(BASE + 'check/url/', {'url': 'https://example.org/path?code=123'}, format='json')
    assert response.status_code == 201
    data = response.json()
    assert data['category'] == 'DANGEROUS' and data['p_url'] == 0.7
    assert data['risk_score'] is None and data['p_sms'] is None
    check = Check.objects.get(pk=data['id'])
    assert check.p_url == 0.7 and check.risk_score is None
    assert 'example.org' not in json.dumps(check.result_json)


@pytest.mark.parametrize('url', ['ftp://example.org', 'http://999.1.1.1', 'https://example.org:bad', '', 123, 'x' * 2049])
def test_invalid_url(api, url, fake_services):
    assert api.post(BASE + 'check/url/', {'url': url}, format='json').status_code == 400
    fake_services[1].assert_not_called()


def test_history_owner_and_detail(api, user):
    own = create_check(user)
    other = get_user_model().objects.create_user('other')
    create_check(other)
    response = api.get(BASE + 'checks/').json()
    assert response['count'] == 1 and response['results'][0]['id'] == own.pk
    assert api.get(BASE + f'checks/{own.pk}/').json()['masked_input'] == '•••'


@pytest.mark.parametrize('method', ['get', 'delete'])
def test_foreign_record_404_even_for_staff(api, user, method):
    user.is_staff = True
    user.save()
    other = get_user_model().objects.create_user('other')
    foreign = create_check(other)
    assert getattr(api, method)(BASE + f'checks/{foreign.pk}/').status_code == 404
    assert Check.objects.filter(pk=foreign.pk).exists()


def test_delete_own(api, user):
    own = create_check(user)
    assert api.delete(BASE + f'checks/{own.pk}/').status_code == 204
    assert not Check.objects.exists()


def test_filters_pagination_and_sorting(api, user):
    checks = [create_check(user) for _ in range(22)]
    page1 = api.get(BASE + 'checks/').json()
    assert page1['count'] == 22 and len(page1['results']) == 20
    assert page1['results'][0]['id'] == checks[-1].pk
    assert len(api.get(BASE + 'checks/?page=2').json()['results']) == 2
    old = timezone.now() - timedelta(days=3)
    Check.objects.filter(pk=checks[0].pk).update(created_at=old, type='url', category='DANGEROUS', risk_level=None)
    assert api.get(BASE + 'checks/?type=url&category=DANGEROUS').json()['count'] == 1
    today = timezone.localdate().isoformat()
    assert api.get(BASE + f'checks/?date_from={today}&date_to={today}&risk_level=LOW').json()['count'] == 21


@pytest.mark.parametrize('query', ['type=wrong', 'risk_level=BAD', 'category=BAD',
    'date_from=wrong', 'date_from=2026-12-31&date_to=2026-01-01'])
def test_invalid_filters(api, query):
    assert api.get(BASE + 'checks/?' + query).status_code == 400


def test_shared_throttle_thirty_then_429(api):
    for n in range(30):
        response = post_sms(api) if n % 2 == 0 else api.post(BASE + 'check/url/', {'url': 'https://example.org'}, format='json')
        assert response.status_code == 201
    response = post_sms(api)
    assert response.status_code == 429
    assert response.json()['error']['code'] == 'RATE_LIMITED'
    assert Check.objects.count() == 30


@pytest.mark.parametrize('error', [RuntimeError('secret C:/private/model'), FileNotFoundError('private file')])
def test_model_failure_returns_503(api, fake_services, error, caplog):
    fake_services[0].side_effect = error
    response = post_sms(api, 'Private message 298745')
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'MODEL_UNAVAILABLE'
    assert 'private' not in response.content.decode().lower()
    assert '298745' not in caplog.text and 'C:/private' not in caplog.text
    assert not Check.objects.exists()


@pytest.mark.parametrize('path', ['auth/register/', 'auth/login/'])
def test_anonymous_csrf_required(path):
    client = APIClient(enforce_csrf_checks=True)
    response = client.post(BASE + path, {}, format='json')
    assert response.status_code == 403
    assert response.json()['error']['code'] == 'FORBIDDEN'


@pytest.mark.parametrize('path,method', [('check/sms/', 'post'), ('check/url/', 'post'),
    ('auth/logout/', 'post'), ('checks/1/', 'delete')])
def test_authenticated_csrf_required(api, path, method):
    api.credentials()
    assert getattr(api, method)(BASE + path, {}, format='json').status_code == 403


def test_wrong_csrf_token_and_origin_rejected(api):
    api.credentials(HTTP_X_CSRFTOKEN='a' * 64)
    assert post_sms(api).status_code == 403
    csrf(api)
    assert api.post(BASE + 'auth/logout/', HTTP_ORIGIN='https://evil.test').status_code == 403


def test_sqli_xss_is_data_and_not_stored_as_raw_sms(api, fake_services, caplog):
    text = "<script>alert('secret2974')</script>'; DROP TABLE auth_user; --"
    assert post_sms(api, text).status_code == 201
    fake_services[0].assert_called_once_with(text)
    check = Check.objects.get()
    # Task 16 preserves safe text portions. This payload has no phone/card/money.
    assert check.masked_input == text and check.input_text is None
    assert text not in json.dumps(check.result_json)
    assert 'secret2974' not in caplog.text
    assert get_user_model().objects.count() == 1


def test_sms_and_embedded_url_absent_from_database_and_logs(api, fake_services, caplog):
    url = 'https://example.org/private?otp=823741'
    text = 'Send my private code here ' + url
    fake_services[0].return_value = AnalysisResult(category='SUSPICIOUS', risk_level='MEDIUM',
        risk_score=50, p_sms=0.5, urls=[URLAnalysis(url, 0.7, 'DANGEROUS', {})],
        explanation=[text])
    assert post_sms(api, text).status_code == 201
    with connection.cursor() as cursor:
        cursor.execute('SELECT * FROM analysis_check')
        stored = repr(cursor.fetchall())
    assert text not in stored and url not in stored and '823741' not in stored
    assert text not in caplog.text and '823741' not in caplog.text
    detail = api.get(BASE + f'checks/{Check.objects.get().pk}/').json()
    assert '823741' not in json.dumps(detail)


def test_stats_scope_and_nullable_risk(api, user):
    create_check(user, risk_score=20)
    create_check(user, type='url', category='DANGEROUS', risk_score=None, risk_level=None)
    other = get_user_model().objects.create_user('other')
    create_check(other, risk_score=80)
    result = api.get(BASE + 'stats/summary/').json()
    expected = {'sms_count': 1, 'url_count': 1, 'average_risk_score': 20.0,
                'categories': {'SAFE': 1, 'DANGEROUS': 1}}
    assert {key: result[key] for key in expected} == expected
    user.is_staff = True
    user.save()
    result = api.get(BASE + 'stats/summary/').json()
    assert result['sms_count'] == 2 and result['average_risk_score'] == 50
    assert api.get(BASE + 'checks/').json()['count'] == 2


def test_health_without_model_loading(anonymous, monkeypatch, tmp_path):
    monkeypatch.setattr(views, 'ARTIFACTS', tmp_path)
    (tmp_path / 'sms_model.joblib').write_bytes(b'present but not loaded')
    response = anonymous.get(BASE + 'health/')
    assert response.status_code == 200 and response.json()['status'] == 'ok'
    assert response.json()['models']['sms_model.joblib'] == 'available'
    assert response.json()['models']['url_model.joblib'] == 'missing'
    assert response.cookies['csrftoken']['httponly']


def test_secure_cookies_for_https(anonymous, user, settings):
    settings.SESSION_COOKIE_SECURE = True
    settings.CSRF_COOKIE_SECURE = True
    response = anonymous.get(BASE + 'health/', secure=True)
    anonymous.credentials(HTTP_X_CSRFTOKEN=response.json()['csrf_token'])
    response = anonymous.post(BASE + 'auth/login/', {'username': 'alice', 'password': PASSWORD}, format='json', secure=True, HTTP_REFERER='https://testserver/')
    assert response.status_code == 200
    assert response.cookies['sessionid']['secure'] and response.cookies['csrftoken']['secure']


def test_email_unique_in_database(user):
    with pytest.raises(IntegrityError), transaction.atomic():
        get_user_model().objects.create_user('different', 'ALICE@example.org')


def test_save_is_atomic_on_json_update_failure(api, monkeypatch):
    original = Check.save

    def fail_update(self, *args, **kwargs):
        if kwargs.get('update_fields') == ['result_json']:
            raise RuntimeError('internal secret')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Check, 'save', fail_update)
    response = post_sms(api)
    assert response.status_code == 500
    assert response.json()['error']['code'] == 'INTERNAL_ERROR'
    assert 'internal secret' not in response.content.decode()
    assert not Check.objects.exists()


def test_maximum_sms_length_allowed(api, fake_services):
    assert post_sms(api, 'x' * 1000).status_code == 201
    assert len(fake_services[0].call_args.args[0]) == 1000


def test_login_does_not_log_credentials(anonymous, user, caplog):
    password = 'PrivateWrongPassword2834'
    anonymous.post(BASE + 'auth/login/', {'username': 'alice', 'password': password}, format='json')
    assert password not in caplog.text


def test_no_user_data_in_empty_statistics(api):
    expected = {
        'sms_count': 0, 'url_count': 0, 'average_risk_score': None, 'categories': {}}
    result = api.get(BASE + 'stats/summary/').json()
    assert {key: result[key] for key in expected} == expected
