"""Сессионный вход с CSRF, без записи учётных данных в журналы."""
import hashlib

from django.contrib.auth import authenticate, get_user_model, login, logout
from django.core.cache import cache
from django.db import IntegrityError, transaction
from rest_framework.exceptions import AuthenticationFailed, Throttled, ValidationError
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.analysis.api import APISessionAuthentication
from .serializers import LoginInput, RegisterInput


class PublicAuthView(APIView):
    permission_classes = [AllowAny]

    def initial(self, request, *args, **kwargs):
        super().initial(request, *args, **kwargs)
        # SessionAuthentication проверяет CSRF только у вошедших пользователей.
        # Для регистрации/входа явно проверяем и анонимный запрос.
        APISessionAuthentication().enforce_csrf(request)


class RegisterView(PublicAuthView):
    def post(self, request):
        serializer = RegisterInput(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            with transaction.atomic():
                user = get_user_model().objects.create_user(**serializer.validated_data)
        except IntegrityError:
            raise ValidationError('Пользователь уже зарегистрирован.') from None
        return Response({'id': user.pk, 'username': user.username, 'email': user.email}, status=201)


class LoginView(PublicAuthView):
    def post(self, request):
        serializer = LoginInput(data=request.data)
        serializer.is_valid(raise_exception=True)
        credentials = serializer.validated_data
        # Общая блокировка имени между клиентами; пароль в ключ не попадает.
        digest = hashlib.sha256(credentials['username'].casefold().encode()).hexdigest()
        key = 'login-failures:' + digest
        if cache.get(key, 0) >= 5:
            raise Throttled(wait=60)
        user = authenticate(request=request, **credentials)
        if user is None:
            cache.add(key, 0, timeout=None)
            failures = cache.incr(key)
            if failures >= 5:
                cache.set(key, failures, timeout=60)
            raise AuthenticationFailed('Неверные учётные данные.')
        cache.delete(key)
        login(request, user)
        return Response({'id': user.pk, 'username': user.username})


class LogoutView(APIView):
    def post(self, request):
        logout(request)
        return Response(status=204)
