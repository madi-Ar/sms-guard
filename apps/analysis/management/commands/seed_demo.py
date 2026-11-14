"""Только искусственные демонстрационные записи, без вызова ML."""
import os
from datetime import timedelta
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone
from apps.analysis.models import Check
from apps.analysis.persistence import save_check, sync_model_versions


class Command(BaseCommand):
    help = 'Создать 30 явно искусственных проверок пользователя sms_guard_demo.'

    def add_arguments(self, parser):
        parser.add_argument('--password', help='Предпочтительнее переменная SMS_GUARD_DEMO_PASSWORD.')

    @transaction.atomic
    def handle(self, *args, **options):
        User = get_user_model()
        user = User.objects.filter(username='sms_guard_demo').first()
        if user is not None and not Check.objects.filter(user=user, demo_key__isnull=False).exists():
            raise CommandError('Имя sms_guard_demo уже занято. Существующий аккаунт не изменён.')
        if user is None:
            password = options.get('password') or os.environ.get('SMS_GUARD_DEMO_PASSWORD')
            if not password:
                raise CommandError('Задайте SMS_GUARD_DEMO_PASSWORD или --password.')
            try:
                validate_password(password, User(username='sms_guard_demo'))
            except ValidationError:
                raise CommandError('Пароль не соответствует стандартным требованиям Django.') from None
            user = User.objects.create_user(username='sms_guard_demo', password=password)
        sync_model_versions()
        now = timezone.now()
        for number in range(30):
            key = f'task16-demo-{user.pk}-{number}'
            if Check.objects.filter(demo_key=key).exists():
                continue
            category = ('SAFE', 'SUSPICIOUS', 'FRAUDULENT')[number % 3]
            level = ('LOW', 'MEDIUM', 'HIGH')[number % 3]
            kind = 'sms' if number % 2 == 0 else 'url'
            address = f'https://demo.example/message/{number}'
            source = (f'Искусственный пример {number + 1}. Встреча в библиотеке. {address}' if kind == 'sms' else address)
            result = {'type': kind, 'language': 'ru', 'category': category,
                'risk_level': level, 'risk_score': (10, 50, 90)[number % 3],
                'p_sms': None, 'p_url': None, 'p_url_max': None,
                'features': {'social_engineering_groups': ['URGENCY'] if number % 3 else []},
                'explanation': ['Искусственный демонстрационный результат, без запуска ML.'],
                'recommendation': 'Используйте только для демонстрации истории.',
                'urls': [{'url': address, 'p_url': None, 'category': category, 'features': {}}]}
            if kind == 'url':
                result['url'] = address
            check = save_check(user, kind, source, result)
            created = now - timedelta(days=number)
            check.demo_key = key
            check.result_json.update(is_demo=True, created_at=created.isoformat())
            check.save(update_fields=['demo_key', 'result_json'])
            Check.objects.filter(pk=check.pk).update(created_at=created)
        self.stdout.write(self.style.SUCCESS('Демо готово. Пользователь sms_guard_demo, 30 искусственных проверок.'))
