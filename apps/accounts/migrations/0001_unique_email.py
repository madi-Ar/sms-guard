"""Уникальность непустого email стандартного User, включая конкурентную запись."""
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [('auth', '0012_alter_user_first_name_max_length')]
    operations = [migrations.RunSQL(
        sql="CREATE UNIQUE INDEX accounts_user_email_unique ON auth_user (LOWER(email)) WHERE email <> ''",
        reverse_sql='DROP INDEX accounts_user_email_unique',
    )]
