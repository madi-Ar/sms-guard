"""Агрегаты Dashboard. В память загружаются только сводки и десять строк."""
from datetime import datetime, time, timedelta
from math import isfinite

from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.urls import reverse
from django.utils import timezone
from rest_framework.exceptions import ValidationError

from apps.analysis.models import Check, DetectedFeature, ModelVersion
from apps.analysis.persistence import sync_model_versions
from ml.risk import GROUP_NAMES

PERIODS = ('7', '30', '90')
GROUPS = (
    ('SAFE', 'Безопасные', ('SAFE',)),
    ('SUSPICIOUS', 'Подозрительные', ('SUSPICIOUS',)),
    ('DANGEROUS', 'Опасные', ('FRAUDULENT', 'DANGEROUS')),
)
METRICS = ('accuracy', 'precision', 'recall', 'f1', 'macro_f1', 'roc_auc', 'pr_auc', 'f2')


def parse_period(value='30'):
    if value not in PERIODS:
        raise ValidationError('Период должен быть 7, 30 или 90 дней.')
    return int(value)


def active_models():
    query = ModelVersion.objects.filter(is_active=True)
    if not {'sms', 'url'}.issubset(set(query.values_list('model_type', flat=True))):
        sync_model_versions()
    result = []
    for model in query.order_by('model_type', 'version'):
        metrics = {key: value for key, value in model.metrics_json.items()
                   if key in METRICS and isinstance(value, (int, float)) and isfinite(value)}
        result.append({'model_type': model.model_type, 'version': model.version,
                       'algorithm': model.algorithm, 'metrics': metrics,
                       'trained_at': model.trained_at.isoformat() if model.trained_at else None})
    return result


def summary(user, period):
    tz = timezone.get_current_timezone()
    today = timezone.localdate()
    first = today - timedelta(days=period - 1)
    start = timezone.make_aware(datetime.combine(first, time.min), tz)
    end = timezone.make_aware(datetime.combine(today + timedelta(days=1), time.min), tz)
    query = Check.objects.filter(created_at__gte=start, created_at__lt=end)
    if not user.is_staff:
        query = query.filter(user=user)
    counts = query.aggregate(
        sms_count=Count('id', filter=Q(type='sms')),
        url_count=Count('id', filter=Q(type='url')),
        average_risk_score=Avg('risk_score'))
    categories = dict(query.order_by().values_list('category').annotate(count=Count('id')))
    distribution = [{'category': code, 'label': label,
                     'count': sum(categories.get(value, 0) for value in members)}
                    for code, label, members in GROUPS]
    grouped_days = query.order_by().annotate(day=TruncDate('created_at', tzinfo=tz)).values('day', 'category').annotate(count=Count('id'))
    daily_counts = {(row['day'], row['category']): row['count'] for row in grouped_days}
    daily = []
    for offset in range(period):
        day = first + timedelta(days=offset)
        values = {code: sum(daily_counts.get((day, category), 0) for category in members)
                  for code, _, members in GROUPS}
        daily.append({'date': day.isoformat(), **values})
    features = DetectedFeature.objects.filter(check_record__in=query).order_by().values('code').annotate(count=Count('check_record_id', distinct=True)).order_by('-count', 'code')[:5]
    top_features = [{'code': row['code'], 'title': GROUP_NAMES.get(row['code'], row['code']), 'count': row['count']} for row in features]
    recent = []
    rows = query.values('id', 'user_id', 'created_at', 'type', 'category', 'risk_level', 'risk_score')[:10]
    for row in rows:
        own = row.pop('user_id') == user.pk
        row['created_at'] = row['created_at'].isoformat()
        row['detail_url'] = reverse('check-result', args=[row['id']]) if own else None
        recent.append(row)
    return {
        **counts, 'categories': categories,
        'safe_count': distribution[0]['count'], 'suspicious_count': distribution[1]['count'],
        'dangerous_count': distribution[2]['count'],
        'period': period, 'date_from': first.isoformat(), 'date_to': today.isoformat(),
        'timezone': str(tz), 'scope': 'all_users' if user.is_staff else 'current_user',
        'category_distribution': distribution, 'daily': daily,
        'top_features': top_features, 'recent_checks': recent,
        'active_models': active_models() if user.is_staff else [],
    }
