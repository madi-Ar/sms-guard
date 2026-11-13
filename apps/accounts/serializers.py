"""Регистрация стандартного Django User."""
import re
from django.contrib.auth import get_user_model
from rest_framework import serializers
from apps.analysis.serializers import StrictString

User = get_user_model()


class RegisterInput(serializers.Serializer):
    username = StrictString(min_length=3, max_length=30, trim_whitespace=False)
    email = serializers.EmailField(max_length=254)
    password = StrictString(min_length=8, trim_whitespace=False, write_only=True)

    def validate_username(self, value):
        if not re.fullmatch(r'[A-Za-z0-9_]{3,30}', value):
            raise serializers.ValidationError('Username: 3–30 ASCII-букв, цифр или _.')
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError('Этот username уже занят.')
        return value

    def validate_email(self, value):
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError('Этот email уже используется.')
        return value.lower()

    def validate_password(self, value):
        if not any(c.isalpha() for c in value) or not any(c.isdigit() for c in value):
            raise serializers.ValidationError('Пароль должен содержать буквы и цифры.')
        return value


class LoginInput(serializers.Serializer):
    username = StrictString(max_length=150, trim_whitespace=False)
    password = StrictString(trim_whitespace=False, write_only=True)
