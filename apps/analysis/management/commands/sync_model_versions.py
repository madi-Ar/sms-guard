from django.core.management.base import BaseCommand
from apps.analysis.persistence import sync_model_versions


class Command(BaseCommand):
    help = 'Обновить справочник ModelVersion из существующих метаданных без обучения.'

    def handle(self, *args, **options):
        versions = sync_model_versions()
        self.stdout.write(f'Загружено версий моделей: {len(versions)}')
