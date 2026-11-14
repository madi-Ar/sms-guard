from django.conf import settings
from django.db import migrations


def backfill(apps, schema_editor):
    alias = schema_editor.connection.alias
    User = apps.get_model(settings.AUTH_USER_MODEL)
    Profile = apps.get_model('accounts', 'UserProfile')
    Check = apps.get_model('analysis', 'Check')
    UrlResult = apps.get_model('analysis', 'UrlResult')
    Feature = apps.get_model('analysis', 'DetectedFeature')
    for user in User.objects.using(alias).iterator():
        Profile.objects.using(alias).get_or_create(user_id=user.pk)
    for check in Check.objects.using(alias).iterator():
        data = check.result_json or {}
        versions = data.get('model_versions') or {}
        check.language = data.get('language')
        check.explanation = data.get('explanation')
        check.recommendation = data.get('recommendation')
        check.sms_model_version = versions.get('sms')
        check.url_model_version = versions.get('url')
        check.save(using=alias, update_fields=['language', 'explanation', 'recommendation', 'sms_model_version', 'url_model_version'])
        urls = data.get('urls', []) if check.type == 'sms' else ([data] if data.get('url') else [])
        for item in urls:
            UrlResult.objects.using(alias).create(check_record_id=check.pk,
                url=item.get('url', ''), domain=None, p_url=item.get('p_url'),
                category=item.get('category'), features_json=item.get('features', {}))
        for code in data.get('features', {}).get('social_engineering_groups', []):
            Feature.objects.using(alias).create(check_record_id=check.pk, code=code, title=code, weight=None)


class Migration(migrations.Migration):
    dependencies = [('accounts', '0002_initial'), ('analysis', '0002_detectedfeature_modelversion_urlresult_and_more')]
    operations = [migrations.RunPython(backfill, migrations.RunPython.noop)]
