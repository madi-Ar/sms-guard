"""Приватное, атомарное сохранение уже рассчитанного результата."""
from copy import deepcopy
import json
import re
from urllib.parse import urlsplit

from django.db import transaction
from django.utils.dateparse import parse_datetime

from apps.accounts.models import UserProfile
from ml.predictor import ARTIFACTS
from ml.risk import GROUP_NAMES
from .models import Check, DetectedFeature, ModelVersion, UrlResult

URL = re.compile(r'(?i)\b(?:https?://|www\.)\S+')
MONEY = re.compile(r'(?i)(?:[$£€₸]\s*\d(?:[\d\s.,]*\d)?|\d(?:[\d\s.,]*\d)?\s*(?:₸|тг\b|тенге\b|руб\w*|₽|доллар\w*|[$£€]))')
CARD = re.compile(r'(?<!\d)\d(?:[ -]?\d){12,18}(?![ -]?\d)')
PHONE = re.compile(r'(?<!\w)\+?\d(?:[\s().-]*\d){6,14}(?!\w)')


def mask_input(value):
    for pattern in (URL, MONEY, CARD, PHONE):
        value = pattern.sub('***', value)
    return value


def private_result(result, source):
    allowed = {'type', 'language', 'category', 'risk_level', 'risk_score', 'p_sms',
               'p_url', 'p_url_max', 'urls', 'url', 'features', 'explanation',
               'recommendation', 'disclaimer', 'model_versions'}
    data = deepcopy({k: v for k, v in result.items() if k in allowed})
    secrets = [source, data.get('url', '')]
    secrets.extend(item.get('url', '') for item in data.get('urls', []))

    def clean(value, key=''):
        if key in {'type', 'language', 'category', 'risk_level', 'model_versions'}:
            return value
        if isinstance(value, dict):
            return {k: clean(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v, key) for v in value]
        if isinstance(value, str):
            if key == 'url':
                return '***'
            for secret in sorted(set(secrets), key=len, reverse=True):
                if secret:
                    value = value.replace(secret, '***')
            return mask_input(value)
        return value
    return clean(data)


@transaction.atomic
def sync_model_versions():
    """Читаем только метаданные, артефакты моделей не загружаются."""
    versions = {}
    for kind in ('sms', 'url'):
        path = ARTIFACTS / f'{kind}_model_meta.json'
        if not path.exists():
            continue
        meta = json.loads(path.read_text(encoding='utf-8'))
        version = meta['version']
        ModelVersion.objects.filter(model_type=kind).exclude(version=version).update(is_active=False)
        ModelVersion.objects.update_or_create(model_type=kind, version=version, defaults={
            'algorithm': meta['algorithm'], 'metrics_json': meta.get('test_metrics', {}),
            'threshold': meta.get('threshold', meta.get('t2')),
            'trained_at': parse_datetime(meta['trained_at']) if meta.get('trained_at') else None,
            'is_active': True})
        versions[kind] = version
    return versions


@transaction.atomic
def save_check(user, check_type, source, result):
    safe = private_result(result, source)
    profile, _ = UserProfile.objects.get_or_create(user=user)
    versions = result.get('model_versions') or {}
    if check_type == 'url' and not versions.get('url'):
        path = ARTIFACTS / 'url_model_meta.json'
        if path.exists():
            versions = {'url': json.loads(path.read_text(encoding='utf-8')).get('version')}
    masked_source = source
    for item in result.get('urls', []):
        if item.get('url'):
            masked_source = masked_source.replace(item['url'], '***')
    check = Check.objects.create(
        user=user, type=check_type, category=result['category'],
        risk_level=result.get('risk_level'), risk_score=result.get('risk_score'),
        p_sms=result.get('p_sms'), p_url=result.get('p_url'), p_url_max=result.get('p_url_max'),
        masked_input=mask_input(masked_source) if check_type == 'sms' else '***', result_json=safe,
        input_text=source if check_type == 'sms' and profile.store_sms_text else None,
        language=safe.get('language'), explanation=safe.get('explanation'),
        recommendation=safe.get('recommendation'),
        sms_model_version=versions.get('sms'), url_model_version=versions.get('url'))
    urls = result.get('urls', []) if check_type == 'sms' else [result]
    for item in urls:
        address = item.get('url', '')
        try:
            domain = urlsplit(address if '://' in address else 'https://' + address).hostname
        except ValueError:
            domain = None
        clean = private_result(item, source)
        UrlResult.objects.create(check_record=check, url='***', domain=domain,
            p_url=item.get('p_url'), category=item.get('category'), features_json=clean.get('features', {}))
    for code in safe.get('features', {}).get('social_engineering_groups', []):
        DetectedFeature.objects.create(check_record=check, code=code, title=GROUP_NAMES.get(code, code), weight=None)
    check.result_json.update(id=check.pk, created_at=check.created_at.isoformat())
    check.save(update_fields=['result_json'])
    return check
