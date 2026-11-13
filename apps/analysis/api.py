"""Общие настройки API: сессии, безопасные ошибки и лимит проверок."""
import logging

from rest_framework.authentication import SessionAuthentication
from rest_framework.exceptions import APIException, AuthenticationFailed
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle

logger = logging.getLogger(__name__)


class APISessionAuthentication(SessionAuthentication):
    def authenticate_header(self, request):
        # DRF иначе превращает отсутствие сессии в 403.
        return 'Session'


class CheckThrottle(UserRateThrottle):
    scope = 'checks'
    rate = '30/min'


class ModelUnavailable(APIException):
    status_code = 503
    default_detail = 'Модели анализа временно недоступны.'
    default_code = 'model_unavailable'


def api_exception_handler(exc, context):
    from rest_framework.views import exception_handler
    response = exception_handler(exc, context)
    if response is None:
        # Не выводим исключение/traceback: они могут содержать пользовательский ввод.
        logger.error('API status=500 error=%s', type(exc).__name__)
        return Response({'error': {'code': 'INTERNAL_ERROR',
                                  'message': 'Внутренняя ошибка сервера.'}}, status=500)
    codes = {400: 'VALIDATION_ERROR', 401: 'UNAUTHENTICATED', 403: 'FORBIDDEN',
             404: 'NOT_FOUND', 405: 'METHOD_NOT_ALLOWED', 429: 'RATE_LIMITED',
             503: 'MODEL_UNAVAILABLE'}
    messages = {400: 'Некорректный ввод.', 401: 'Требуется вход в систему.',
                403: 'Действие запрещено. Проверьте CSRF-токен.',
                404: 'Запись не найдена.', 405: 'Метод не поддерживается.',
                429: 'Лимит запросов превышен. Повторите позже.',
                503: 'Модели анализа временно недоступны.'}
    # Сообщения сериализаторов статичны; не включаем отклонённые значения.
    def first_message(value):
        if isinstance(value, dict):
            return first_message(next(iter(value.values()))) if value else ''
        if isinstance(value, (list, tuple)):
            return first_message(value[0]) if value else ''
        return str(value)
    status = response.status_code
    message = first_message(response.data) if status == 400 else messages.get(status, 'Ошибка запроса.')
    if isinstance(exc, AuthenticationFailed):
        message = 'Неверные учётные данные.'
    response.data = {'error': {'code': codes.get(status, 'REQUEST_ERROR'), 'message': message}}
    return response
