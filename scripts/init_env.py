"""Создаёт локальный .env с уникальным SECRET_KEY; повторный запуск безопасен."""
from pathlib import Path
import secrets
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
example = BASE_DIR / '.env.example'
target = BASE_DIR / '.env'

if not example.is_file():
    sys.exit('Ошибка: не найден .env.example')

if target.exists():
    content = target.read_text(encoding='utf-8')
    for line in content.splitlines():
        if line.startswith('DJANGO_SECRET_KEY=') and line.partition('=')[2].strip():
            print('Файл .env уже содержит SECRET_KEY. Ничего не изменено.')
            sys.exit(0)
else:
    content = example.read_text(encoding='utf-8')

lines = content.splitlines()
key = secrets.token_urlsafe(48)
new_lines = [f'DJANGO_SECRET_KEY={key}' if line.startswith('DJANGO_SECRET_KEY=') else line for line in lines]
if not any(line.startswith('DJANGO_SECRET_KEY=') for line in lines):
    new_lines.insert(0, f'DJANGO_SECRET_KEY={key}')
target.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
print('Файл .env подготовлен с индивидуальным секретным ключом. Не публикуйте его.')
