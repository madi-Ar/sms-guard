"""Сессионный вход с CSRF, без записи учётных данных в журналы."""
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods

import hashlib

from django.contrib.auth import authenticate, get_user_model, login, logout
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.core.cache import cache
from django.db import IntegrityError, transaction
from rest_framework.exceptions import AuthenticationFailed, Throttled, ValidationError
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.analysis.api import APISessionAuthentication
from .serializers import LoginInput, RegisterInput
from apps.analysis.models import Check
from .forms import ProfileForm
from .models import UserProfile


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


@require_GET
def auth_page(request, mode):
    if request.user.is_authenticated:
        return redirect('check-sms')
    return render(request, 'accounts/auth.html', {'mode': mode})


@login_required(login_url='/login/')
@require_http_methods(['GET', 'POST'])
def profile_page(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    preferences = ProfileForm(instance=profile)
    password = PasswordChangeForm(request.user)
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'preferences':
            preferences = ProfileForm(request.POST, instance=profile)
            if preferences.is_valid():
                preferences.save()
                messages.success(request, 'Настройки сохранены. Согласие действует на новые проверки.')
                return redirect('profile')
        elif action == 'password':
            password = PasswordChangeForm(request.user, request.POST)
            if password.is_valid():
                user = password.save()
                update_session_auth_hash(request, user)
                messages.success(request, 'Пароль изменён.')
                return redirect('profile')
        elif action == 'delete_history' and request.POST.get('confirm') == 'yes':
            with transaction.atomic():
                Check.objects.filter(user=request.user).delete()
            messages.success(request, 'Вся ваша история удалена.')
            return redirect('profile')
        else:
            messages.error(request, 'Подтвердите удаление отдельным флажком.')
    return render(request, 'accounts/profile.html', {'preferences': preferences, 'password_form': password})
