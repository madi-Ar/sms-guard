"""История, приватность и профиль. Все записи находятся в тестовой БД."""
import json
from io import StringIO
from unittest.mock import patch
import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.utils import timezone
from apps.accounts.models import UserProfile
from apps.analysis.models import Check, DetectedFeature, UrlResult
from apps.analysis.persistence import mask_input, save_check

pytestmark = pytest.mark.django_db


@pytest.fixture
def owner(client):
    user = get_user_model().objects.create_user(username='history_owner', password='OriginalSecure48!')
    client.force_login(user)
    return user


def result():
    return {'type': 'sms', 'language': 'en', 'category': 'SUSPICIOUS', 'risk_level': 'MEDIUM',
        'risk_score': 50, 'p_sms': .4, 'p_url_max': .5,
        'model_versions': {'sms': '1.0.0', 'url': '1.0.0'},
        'features': {'social_engineering_groups': ['URGENCY']},
        'urls': [{'url': 'https://example.org/?secret=12345', 'p_url': .5, 'category': 'SUSPICIOUS', 'features': {}}],
        'explanation': ['Urgent message'], 'recommendation': 'Check the sender'}


def saved(user, text='Call +7 (700) 123-45-67 and send 5000 тг'):
    return save_check(user, 'sms', text, result())


def test_profile_created_default_false(owner):
    assert UserProfile.objects.get(user=owner).store_sms_text is False


@pytest.mark.parametrize('consent', [True, False])
def test_profile_consent_toggle(client, owner, consent):
    response = client.post('/profile/', {'action': 'preferences', 'preferred_language': 'kk', **({'store_sms_text': 'on'} if consent else {})})
    assert response.status_code == 302
    owner.profile.refresh_from_db()
    assert owner.profile.store_sms_text is consent
    assert owner.profile.preferred_language == 'kk'


def test_consent_only_new_sms(owner):
    first = saved(owner)
    owner.profile.store_sms_text = True
    owner.profile.save()
    second = saved(owner)
    first.refresh_from_db()
    assert first.input_text is None
    assert second.input_text is not None
    owner.profile.store_sms_text = False
    owner.profile.save()
    assert saved(owner).input_text is None


def test_json_does_not_store_source(owner):
    source = 'Transfer 5000 тг to +7 700 123 45 67'
    data = result()
    data['explanation'] = [source]
    check = save_check(owner, 'sms', source, data)
    assert source not in json.dumps(check.result_json, ensure_ascii=False)
    assert check.input_text is None
    assert '5000' not in check.masked_input


@pytest.mark.parametrize('source', ['Call +7 (700) 123-45-67 now', 'Call 07700 900123 now', 'Card 1234 5678 9012 3456 now', 'Card 1234567890123 now', 'Card 1234567890123456789 now', 'Send 5000 тг now', 'Send $120.50 now', 'Send 1 000 руб now', 'Send £20 now'])
def test_sensitive_mask(source):
    masked = mask_input(source)
    assert '***' in masked and masked.endswith('now')
    assert not any(c.isdigit() for c in masked)


def test_safe_portion_kept():
    assert mask_input('Meet at the library!') == 'Meet at the library!'


def test_related_records(owner):
    check = saved(owner)
    assert check.check_type == 'SMS' and check.input_masked == check.masked_input
    assert check.url_results.get().domain == 'example.org'
    assert check.url_results.get().url == '***'
    assert check.detected_features.get().code == 'URGENCY'
    assert check.detected_features.get().weight is None


@pytest.mark.parametrize('target', ['apps.analysis.persistence.UrlResult.objects.create', 'apps.analysis.persistence.DetectedFeature.objects.create'])
def test_atomic_rollback(owner, target):
    with patch(target, side_effect=RuntimeError('simulated failure')):
        with pytest.raises(RuntimeError):
            saved(owner)
    assert Check.objects.count() == UrlResult.objects.count() == DetectedFeature.objects.count() == 0


def test_history_own_only(client, owner):
    mine = saved(owner)
    other = get_user_model().objects.create_user(username='another')
    saved(other)
    response = client.get('/history/')
    assert list(response.context['page_obj']) == [mine]


@pytest.mark.parametrize('query,expected', [({'type': 'sms'}, 1), ({'type': 'url'}, 0), ({'category': 'SAFE'}, 0), ({'risk_level': 'MEDIUM'}, 1), ({'category': 'SUSPICIOUS', 'risk_level': 'MEDIUM', 'type': 'sms'}, 1), ({'date_from': '2000-01-01', 'date_to': '2100-01-01'}, 1), ({'date_to': '2000-01-01'}, 0)])
def test_history_filters(client, owner, query, expected):
    saved(owner)
    response = client.get('/history/', query)
    assert response.status_code == 200
    assert response.context['page_obj'].paginator.count == expected


def test_pagination_filters_order(client, owner):
    checks = [saved(owner) for _ in range(21)]
    response = client.get('/history/?type=sms&category=SUSPICIOUS')
    assert len(response.context['page_obj']) == 20
    assert response.context['page_obj'][0] == checks[-1]
    assert 'type=sms' in response.context['filter_query']
    assert 'category=SUSPICIOUS' in response.context['filter_query']
    assert len(client.get('/history/?type=sms&page=2').context['page_obj']) == 1


@pytest.mark.parametrize('path', ['/history/', '/profile/'])
def test_login_required(client, path):
    assert client.get(path).status_code == 302


def test_own_detail_and_api_without_models(client, owner):
    check = saved(owner)
    with patch('apps.analysis.services.analyze_sms', side_effect=AssertionError):
        assert client.get(f'/checks/{check.pk}/').status_code == 200
        data = client.get(f'/api/v1/checks/{check.pk}/').json()
    assert data['masked_input'] == check.masked_input
    assert 'input_text' not in data


def test_consent_detail_api(client, owner):
    owner.profile.store_sms_text = True
    owner.profile.save()
    check = saved(owner)
    assert client.get(f'/api/v1/checks/{check.pk}/').json()['input_text'] == check.input_text


def test_foreign_detail_delete_even_staff(client, owner):
    other = get_user_model().objects.create_user(username='other')
    check = saved(other)
    owner.is_staff = True
    owner.save()
    assert client.get(f'/checks/{check.pk}/').status_code == 404
    assert client.post(f'/checks/{check.pk}/delete/', {'confirm': 'yes'}).status_code == 404
    assert client.delete(f'/api/v1/checks/{check.pk}/').status_code == 404
    assert Check.objects.filter(pk=check.pk).exists()


def test_delete_requires_post_confirmation_and_cascades(client, owner):
    check = saved(owner)
    path = f'/checks/{check.pk}/delete/'
    assert client.get(path).status_code == 405
    client.post(path)
    assert Check.objects.filter(pk=check.pk).exists()
    assert client.post(path, {'confirm': 'yes'}).url == '/history/'
    assert not Check.objects.exists() and not UrlResult.objects.exists() and not DetectedFeature.objects.exists()


def test_delete_history_only_own_confirmed(client, owner):
    saved(owner)
    other = get_user_model().objects.create_user(username='other')
    theirs = saved(other)
    client.post('/profile/', {'action': 'delete_history'})
    assert Check.objects.count() == 2
    client.post('/profile/', {'action': 'delete_history', 'confirm': 'yes'})
    assert list(Check.objects.all()) == [theirs]
    assert UrlResult.objects.count() == DetectedFeature.objects.count() == 1


def test_password_change_session(client, owner):
    response = client.post('/profile/', {'action': 'password', 'old_password': 'OriginalSecure48!', 'new_password1': 'ReplacementSecure73!', 'new_password2': 'ReplacementSecure73!'})
    assert response.status_code == 302
    owner.refresh_from_db()
    assert owner.check_password('ReplacementSecure73!')
    assert client.get('/profile/').status_code == 200


@pytest.mark.parametrize('old,new,confirm', [('wrong', 'ReplacementSecure73!', 'ReplacementSecure73!'), ('OriginalSecure48!', '123', '123'), ('OriginalSecure48!', 'ReplacementSecure73!', 'different')])
def test_invalid_password(client, owner, old, new, confirm):
    client.post('/profile/', {'action': 'password', 'old_password': old, 'new_password1': new, 'new_password2': confirm})
    owner.refresh_from_db()
    assert owner.check_password('OriginalSecure48!')


def test_csrf(owner):
    from django.test import Client
    client = Client(enforce_csrf_checks=True)
    client.force_login(owner)
    check = saved(owner)
    assert client.post('/profile/', {'action': 'delete_history', 'confirm': 'yes'}).status_code == 403
    assert client.post(f'/checks/{check.pk}/delete/', {'confirm': 'yes'}).status_code == 403


def test_seed_repeat():
    import secrets
    output = StringIO()
    call_command('seed_demo', password=secrets.token_urlsafe(24), stdout=output)
    call_command('seed_demo', stdout=output)
    assert Check.objects.count() == 30
    assert Check.objects.filter(demo_key__isnull=False).count() == 30
    assert set(Check.objects.values_list('category', flat=True)) == {'SAFE', 'SUSPICIOUS', 'FRAUDULENT'}
    assert UrlResult.objects.count() == 30 and DetectedFeature.objects.count() == 20
    assert all(c.input_text is None and c.result_json['is_demo'] for c in Check.objects.all())


def test_history_empty_and_invalid_dates(client, owner):
    assert 'Проверок не найдено' in client.get('/history/').content.decode()
    assert client.get('/history/?date_from=2030-01-01&date_to=2020-01-01').status_code == 400


def test_history_escapes_html_and_truncates(client, owner):
    source = '<script>alert(1)</script>' + 'x' * 100
    saved(owner, source)
    content = client.get('/history/').content.decode()
    assert source not in content
    assert '&lt;script&gt;' in content
    assert 'x' * 61 not in content


def test_delete_history_rollback(client, owner):
    from django.db.models.signals import pre_delete
    saved(owner)
    def fail(sender, **kwargs):
        raise RuntimeError('simulated cascade failure')
    pre_delete.connect(fail, sender=UrlResult)
    try:
        with pytest.raises(RuntimeError):
            client.post('/profile/', {'action': 'delete_history', 'confirm': 'yes'})
    finally:
        pre_delete.disconnect(fail, sender=UrlResult)
    assert Check.objects.count() == UrlResult.objects.count() == DetectedFeature.objects.count() == 1


def test_registry_matches_metadata():
    from apps.analysis.persistence import sync_model_versions
    from apps.analysis.models import ModelVersion
    from ml.predictor import ARTIFACTS
    sync_model_versions()
    sync_model_versions()
    assert ModelVersion.objects.count() == 2
    for kind in ('sms', 'url'):
        data = json.loads((ARTIFACTS / f'{kind}_model_meta.json').read_text(encoding='utf-8'))
        model = ModelVersion.objects.get(model_type=kind)
        assert model.version == data['version'] and model.algorithm == data['algorithm']
        assert model.metrics_json == data['test_metrics']
        assert model.threshold == data.get('threshold', data.get('t2'))
