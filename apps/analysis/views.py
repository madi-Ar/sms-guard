"""Служебный endpoint без доступа к конфиденциальным данным."""
from django.http import JsonResponse
from django.views.decorators.http import require_GET


@require_GET
def health(request):
    """Проверка доступности приложения на этапе задания 2."""
    return JsonResponse({'status': 'ok'})
