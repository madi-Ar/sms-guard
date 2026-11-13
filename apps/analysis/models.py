"""Минимальная история API без исходного текста сообщений."""
from django.conf import settings
from django.db import models


class Check(models.Model):
    TYPES = (('sms', 'SMS'), ('url', 'URL'))
    CATEGORIES = ('SAFE', 'SUSPICIOUS', 'FRAUDULENT', 'DANGEROUS')
    LEVELS = ('LOW', 'MEDIUM', 'HIGH')

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    type = models.CharField(max_length=3, choices=TYPES)
    created_at = models.DateTimeField(auto_now_add=True)
    category = models.CharField(max_length=16)
    risk_level = models.CharField(max_length=6, null=True)
    risk_score = models.PositiveSmallIntegerField(null=True)
    p_sms = models.FloatField(null=True)
    p_url = models.FloatField(null=True)
    p_url_max = models.FloatField(null=True)
    masked_input = models.TextField()
    result_json = models.JSONField(default=dict)

    class Meta:
        ordering = ['-created_at', '-id']
