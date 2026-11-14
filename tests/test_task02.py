"""Проверяемые критерии задания 2."""
from django.urls import reverse
import pytest


@pytest.fixture
def home_client(client, django_user_model, db):
    # The task 17 Dashboard replaces the public task 2 placeholder.
    user = django_user_model.objects.create_user('home_test')
    client.force_login(user)
    return client


def test_home_page_works(home_client):
    response = home_client.get(reverse('dashboard-index'))
    assert response.status_code == 200
    assert 'SMS Guard' in response.content.decode('utf-8')


def test_home_has_bootstrap_and_navigation(home_client):
    response = home_client.get('/')
    html = response.content.decode('utf-8')
    assert 'bootstrap@5.3.3' in html
    assert 'href="/api/v1/health/"' in html


def test_health_endpoint_returns_expected_json(client):
    response = client.get('/api/v1/health/')
    assert response.status_code == 200
    assert response['Content-Type'].startswith('application/json')
    assert response.json()['status'] == 'ok'


def test_health_disallows_post(client):
    assert client.post('/api/v1/health/').status_code == 405
