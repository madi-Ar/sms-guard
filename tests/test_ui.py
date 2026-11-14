"""Страницы UI и интеграция с существующими сессиями/CSRF."""
from pathlib import Path
import re
from unittest.mock import Mock

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import Client
from django.urls import reverse
import pytest

from apps.analysis import services
from apps.analysis.services import AnalysisResult, URLAnalysis
from apps.analysis.views import save_check

pytestmark = pytest.mark.django_db
PASSWORD = 'UiDemoPassword294'


@pytest.fixture(autouse=True)
def clear_cache():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def user():
    return get_user_model().objects.create_user('ui_student', 'ui@example.test', PASSWORD)


@pytest.fixture
def signed_in(client, user):
    client.force_login(user)
    return client


def token(response):
    return re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', response.content.decode()).group(1)


@pytest.mark.parametrize('path', ['/check/sms/', '/check/url/', '/checks/1/'])
def test_anonymous_redirects_to_login(client, path):
    response = client.get(path)
    assert response.status_code == 302
    assert response.url == '/login/?next=' + path


@pytest.mark.parametrize('path', ['/login/', '/register/'])
def test_auth_pages_inherit_base_and_set_csrf(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert 'accounts/auth.html' in [t.name for t in response.templates]
    assert 'base.html' in [t.name for t in response.templates]
    assert token(response)
    assert response.cookies['csrftoken']['httponly']
    assert b'autocomplete="username"' in response.content
    assert b'type="password"' in response.content


@pytest.mark.parametrize('path', ['/login/', '/register/'])
def test_logged_in_auth_pages_redirect(signed_in, path):
    assert signed_in.get(path).url == '/check/sms/'


@pytest.mark.parametrize('path,kind,limit,endpoint', [
    ('/check/sms/', 'sms', '1000', '/api/v1/check/sms/'),
    ('/check/url/', 'url', '2048', '/api/v1/check/url/'),
])
def test_check_form_contract(signed_in, path, kind, limit, endpoint):
    response = signed_in.get(path)
    assert response.status_code == 200
    assert 'analysis/check.html' in [t.name for t in response.templates]
    assert 'base.html' in [t.name for t in response.templates]
    html = response.content.decode()
    for expected in (
        'id="check-input"', 'for="check-input"', f'maxlength="{limit}"',
        f'data-kind="{kind}"', f'data-endpoint="{endpoint}"',
        'type="submit"', 'id="clear-button"', 'id="result"',
        'aria-live="polite"', 'js/analysis.js', 'js/ui.js', 'css/style.css',
        'spinner-border', 'id="logout-button"',
    ):
        assert expected in html
    assert token(response)


def test_sms_examples_and_counter(signed_in):
    html = signed_in.get('/check/sms/').content.decode()
    assert html.count('data-example=') == 3
    assert '0/1000' in html
    assert 'Безопасное' in html and 'Подозрительное' in html and 'Мошенническое' in html
    assert 'type="button" data-example=' in html


def test_url_accepts_schemeless_text(signed_in):
    html = signed_in.get('/check/url/').content.decode()
    assert 'name="url" type="text"' in html
    assert 'type="url"' not in html
    assert 'Ссылка анализируется как текст. Сайт не открывается.' in html


def test_result_page_and_api_owner(signed_in, user, monkeypatch):
    mock = Mock(side_effect=AssertionError('Opening result must not repeat analysis'))
    monkeypatch.setattr(services, 'analyze_sms', mock)
    result = AnalysisResult(category='SAFE', risk_level='LOW', risk_score=10,
                            p_sms=0.1, language='en').to_dict()
    check = save_check(user, 'sms', 'private message', result)
    response = signed_in.get(reverse('check-result', args=[check.pk]))
    assert response.status_code == 200
    assert 'analysis/result.html' in [t.name for t in response.templates]
    assert f'data-endpoint="/api/v1/checks/{check.pk}/"' in response.content.decode()
    payload = signed_in.get(f'/api/v1/checks/{check.pk}/').json()
    assert payload['masked_input'] == check.masked_input
    assert 'private message' not in response.content.decode()
    mock.assert_not_called()


@pytest.mark.parametrize('staff', [False, True])
def test_foreign_result_is_404(signed_in, user, staff):
    user.is_staff = staff
    user.save()
    other = get_user_model().objects.create_user('other_ui')
    check = save_check(other, 'sms', 'private', AnalysisResult(category='SAFE').to_dict())
    assert signed_in.get(f'/checks/{check.pk}/').status_code == 404
    assert signed_in.get(f'/api/v1/checks/{check.pk}/').status_code == 404


def test_nonexistent_result_is_404(signed_in):
    response = signed_in.get('/checks/99999999/')
    assert response.status_code == 404
    assert 'Запись не найдена' in response.content.decode()


def test_username_is_autoescaped(signed_in, user):
    user.username = '<script>alert(1)</script>'
    user.save()
    html = signed_in.get('/check/sms/').content.decode()
    assert '<script>alert(1)</script>' not in html
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html


def test_auth_csrf_login_rotation_and_logout(user):
    browser = Client(enforce_csrf_checks=True)
    page = browser.get('/login/')
    login_data = {'username': user.username, 'password': PASSWORD}
    assert browser.post('/api/v1/auth/login/', login_data, content_type='application/json').status_code == 403
    response = browser.post('/api/v1/auth/login/', login_data, content_type='application/json',
                            HTTP_X_CSRFTOKEN=token(page))
    assert response.status_code == 200
    page = browser.get('/check/sms/')
    assert page.status_code == 200
    assert browser.post('/api/v1/auth/logout/', {}, content_type='application/json').status_code == 403
    assert browser.post('/api/v1/auth/logout/', {}, content_type='application/json',
                        HTTP_X_CSRFTOKEN=token(page)).status_code == 204
    assert browser.get('/check/sms/').status_code == 302


def test_registration_uses_existing_api():
    browser = Client(enforce_csrf_checks=True)
    page = browser.get('/register/')
    response = browser.post('/api/v1/auth/register/', {
        'username': 'new_ui', 'email': 'new_ui@example.test', 'password': PASSWORD,
    }, content_type='application/json', HTTP_X_CSRFTOKEN=token(page))
    assert response.status_code == 201
    user = get_user_model().objects.get(username='new_ui')
    assert user.check_password(PASSWORD)
    assert browser.get('/check/sms/').status_code == 302


@pytest.mark.parametrize('kind,body', [
    ('sms', {'text': '<script>alert(1)</script> "hello" & https://example.com/?a=1&b=2'}),
    ('url', {'url': 'example.com/?a=1&b=2'}),
])
def test_ui_token_works_with_existing_check_api(user, monkeypatch, kind, body):
    monkeypatch.setattr(services, 'analyze_sms', lambda value: AnalysisResult(
        category='SAFE', risk_level='LOW', risk_score=10, p_sms=.1, language='en'))
    monkeypatch.setattr(services, 'analyze_url', lambda value: URLAnalysis(
        url=value, p_url=.2, category='SAFE', features={}))
    browser = Client(enforce_csrf_checks=True)
    browser.force_login(user)
    page = browser.get('/check/' + kind + '/')
    endpoint = '/api/v1/check/' + kind + '/'
    assert browser.post(endpoint, body, content_type='application/json').status_code == 403
    response = browser.post(endpoint, body, content_type='application/json',
                            HTTP_X_CSRFTOKEN=token(page))
    assert response.status_code == 201
    assert response.json()['type'] == kind


def test_javascript_safe_sinks_and_session_transport():
    root = Path(settings.BASE_DIR)
    ui = (root / 'static/js/ui.js').read_text(encoding='utf-8')
    analysis = (root / 'static/js/analysis.js').read_text(encoding='utf-8')
    for unsafe in ('innerHTML', 'insertAdjacentHTML', 'localStorage', 'sessionStorage', 'document.write'):
        assert unsafe not in ui + analysis
    assert 'textContent' in analysis and 'createElement' in analysis
    assert 'credentials: "same-origin"' in ui and '"X-CSRFToken"' in ui
    assert 'finally' in ui and 'finally' in analysis
    assert 'api(form.dataset.endpoint, "POST"' in analysis
    assert 'event.preventDefault()' in analysis
    assert 'busy(form, false)' in analysis
    assert 'data.p_url_max === null' in analysis
    assert 'typeof data.risk_score === "number"' in analysis


def test_ui_sources_keep_russian_utf8_text():
    root = Path(settings.BASE_DIR)
    for name in (
        "templates/base.html", "templates/accounts/auth.html",
        "templates/analysis/check.html", "templates/analysis/result.html",
        "templates/analysis/not_found.html", "templates/dashboard/index.html",
        "static/js/ui.js", "static/js/analysis.js",
    ):
        value = (root / name).read_text(encoding="utf-8")
        assert re.search(r"[А-Яа-яЁё]", value), name
        assert "???" not in value, name
