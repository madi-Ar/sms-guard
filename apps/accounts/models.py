"""Настройки стандартного пользователя Django."""
from django.conf import settings
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profile')
    store_sms_text = models.BooleanField(default=False)
    preferred_language = models.CharField(max_length=2, default='ru', choices=[('ru', 'Русский'), ('kk', 'Қазақша'), ('en', 'English')])


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_profile(sender, instance, created, raw=False, **kwargs):
    if created and not raw:
        UserProfile.objects.get_or_create(user=instance)
