"""История анализа и связанные результаты с настройкой хранения SMS."""
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
    input_text = models.TextField(null=True, blank=True)
    language = models.CharField(max_length=16, null=True, blank=True)
    explanation = models.JSONField(null=True, blank=True)
    recommendation = models.TextField(null=True, blank=True)
    sms_model_version = models.CharField(max_length=80, null=True, blank=True)
    url_model_version = models.CharField(max_length=80, null=True, blank=True)
    demo_key = models.CharField(max_length=80, null=True, blank=True, unique=True)

    @property
    def check_type(self):
        return self.type.upper()

    @check_type.setter
    def check_type(self, value):
        self.type = value.lower()

    @property
    def input_masked(self):
        return self.masked_input

    @input_masked.setter
    def input_masked(self, value):
        self.masked_input = value

    @property
    def display_input(self):
        return self.input_text if self.input_text is not None else self.input_masked

    class Meta:
        ordering = ['-created_at', '-id']
        indexes = [models.Index(fields=['user', 'created_at']), models.Index(fields=['category'])]


class UrlResult(models.Model):
    check_record = models.ForeignKey(Check, db_column='check_id', on_delete=models.CASCADE, related_name='url_results')
    url = models.TextField()
    domain = models.CharField(max_length=253, null=True, blank=True)
    p_url = models.FloatField(null=True)
    category = models.CharField(max_length=16, null=True)
    features_json = models.JSONField(default=dict)


class DetectedFeature(models.Model):
    check_record = models.ForeignKey(Check, db_column='check_id', on_delete=models.CASCADE, related_name='detected_features')
    code = models.CharField(max_length=80)
    title = models.CharField(max_length=200)
    weight = models.FloatField(null=True)


class ModelVersion(models.Model):
    model_type = models.CharField(max_length=3, choices=Check.TYPES)
    version = models.CharField(max_length=80)
    algorithm = models.CharField(max_length=120)
    metrics_json = models.JSONField(default=dict)
    threshold = models.FloatField(null=True)
    trained_at = models.DateTimeField(null=True)
    is_active = models.BooleanField(default=False)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['model_type', 'version'], name='unique_model_version')]
