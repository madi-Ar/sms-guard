"""Проверка входа без предварительной NLP-нормализации."""
import unicodedata

from rest_framework import serializers
from ml.url_features import validate_url
from .models import Check


class StrictString(serializers.CharField):
    def to_internal_value(self, data):
        if not isinstance(data, str):
            raise serializers.ValidationError('Ожидается строка.')
        return super().to_internal_value(data)


class SMSString(StrictString):
    def to_internal_value(self, data):
        value = super().to_internal_value(data)
        if len(value) > 1000:
            raise serializers.ValidationError('SMS не должно превышать 1000 символов.')
        # Перенос/табуляция становятся разделителями, прочие Cc удаляются.
        value = ''.join(' ' if c in '\r\n\t' else c for c in value
                        if c in '\r\n\t' or unicodedata.category(c) != 'Cc')
        if not value.strip():
            raise serializers.ValidationError('SMS должно содержать текст.')
        return value


class SMSInput(serializers.Serializer):
    text = SMSString(max_length=1000, trim_whitespace=False)


class URLInput(serializers.Serializer):
    url = StrictString(max_length=2048, trim_whitespace=False)

    def validate_url(self, value):
        try:
            validate_url(value)
        except ValueError as exc:
            raise serializers.ValidationError(str(exc)) from None
        return value


class HistoryFilters(serializers.Serializer):
    date_from = serializers.DateField(required=False)
    date_to = serializers.DateField(required=False)
    type = serializers.ChoiceField(choices=['sms', 'url'], required=False)
    category = serializers.ChoiceField(choices=Check.CATEGORIES, required=False)
    risk_level = serializers.ChoiceField(choices=Check.LEVELS, required=False)

    def validate(self, attrs):
        if 'date_from' in attrs and 'date_to' in attrs and attrs['date_from'] > attrs['date_to']:
            raise serializers.ValidationError('date_from не может быть позже date_to.')
        return attrs
