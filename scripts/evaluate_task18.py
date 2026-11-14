"""Аудит сохранённых моделей и фиксированных данных, без обучения и сети.

Запуск из корня проекта: python -m scripts.evaluate_task18
Результат записывается только в docs/18_model_results.json.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             fbeta_score, precision_recall_fscore_support, roc_auc_score)

from apps.analysis.services import analyze_sms, analyze_url, extract_urls
from ml.predictor import get_predictor
from ml.rules import find_social_engineering_groups
from ml.url_dataset import assert_no_group_overlap, domain_group

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'ml/artifacts'
DATA = ROOT / 'data/processed'
OUTPUT = ROOT / 'docs/18_model_results.json'
DEMO = [
    ('ru', 'Привет! Встречаемся завтра в 18:00 у входа в библиотеку.', ['SAFE']),
    ('en', 'Your order #4821 has been shipped and will arrive on Monday.', ['SAFE']),
    ('kk', 'Сәлем! Ертең сағат 10:00-де кездесеміз бе?', ['SAFE']),
    ('ru', 'Ваш код подтверждения: 4821. Никому не сообщайте этот код.', ['SAFE', 'SUSPICIOUS']),
    ('ru', 'Ваша карта заблокирована! Срочно подтвердите данные по ссылке: http://secure-bank-verify.example/login', ['FRAUDULENT']),
    ('en', 'URGENT: Your account will be suspended in 24 hours. Verify now at http://account-update.example/account-update', ['FRAUDULENT']),
    ('ru', 'Мама, у меня проблемы, срочно переведи 50 000 на этот номер, потом всё объясню. Никому не говори.', ['FRAUDULENT']),
    ('en', 'Congratulations! You won a $1000 gift card. Claim your prize: http://short.example/prize-claim-test', ['FRAUDULENT']),
    ('ru', 'Сотрудник службы безопасности банка: назовите код из SMS для отмены подозрительной операции.', ['FRAUDULENT']),
    ('kk', 'Картаңыз бұғатталды. Жедел түрде мына сілтеме арқылы растаңыз: http://bank-kz-verify.test', ['SUSPICIOUS', 'FRAUDULENT']),
    ('ru', 'Ваша посылка ожидает оплаты таможенного сбора 350 ₸: http://post-delivery-pay.example', ['SUSPICIOUS', 'FRAUDULENT']),
    ('URL', 'https://www.example.com/about', ['SAFE']),
    ('URL', 'http://xn--exmple-cua.test/secure/login?verify=1', ['DANGEROUS']),
]


def metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    binary = precision_recall_fscore_support(labels, predicted, average='binary', zero_division=0)
    macro = precision_recall_fscore_support(labels, predicted, average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(labels, predicted)),
            'precision': float(binary[0]), 'recall': float(binary[1]), 'f1': float(binary[2]),
            'macro_precision': float(macro[0]), 'macro_recall': float(macro[1]), 'macro_f1': float(macro[2]),
            'roc_auc': float(roc_auc_score(labels, probabilities)),
            'pr_auc': float(average_precision_score(labels, probabilities)),
            'f2': float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
            'confusion_matrix': confusion_matrix(labels, predicted, labels=[0, 1]).tolist()}


def error_examples(frame, probabilities, threshold, model):
    predicted = (probabilities >= threshold).astype(int)
    examples = []
    for kind, truth, guess in [('FP', 0, 1), ('FN', 1, 0)]:
        positions = np.flatnonzero((frame.label.to_numpy() == truth) & (predicted == guess))[:5]
        for position in positions:
            row = frame.iloc[position]
            # Never expose actual corpus hostnames as potentially clickable links.
            if model == 'SMS':
                value = row.text_raw
                for address in extract_urls(value):
                    value = value.replace(address, '[address omitted]')
                value = re.sub(r'(?i)(?:https?://|www\.)\S+|[\w.+-]+@[\w.-]+', '[address omitted]', value)
                value = re.sub(r'\d', '*', value)
            else:
                parsed = urlsplit(row.url)
                value = parsed.scheme + '://redacted.example'
                value += '/[path omitted]' if parsed.path else ''
                value += '?[query omitted]' if parsed.query else ''
            examples.append({'model': model, 'kind': kind, 'test_position': int(position),
                             'source_id': int(row.id if model == 'SMS' else row.source_id),
                             'language': 'en' if model == 'SMS' else None,
                             'input': str(value), 'true': truth, 'predicted': guess,
                             'probability': float(probabilities[position]),
                             'url_features': {k: float(row[k]) for k in ('is_https', 'path_length', 'query_param_count')} if model == 'URL' else None})
    return examples


def latency(model, values):
    model.predict_proba(values)
    times = []
    for _ in range(100):
        started = perf_counter()
        model.predict_proba(values)
        times.append((perf_counter() - started) * 1000)
    return {'requests': len(times), 'mean_ms': float(np.mean(times)), 'max_ms': float(max(times))}


def main():
    protected = [p for folder in ('data', 'ml/artifacts') for p in (ROOT / folder).rglob('*') if p.is_file()]
    protected += [ROOT / 'risk_config.yaml', ROOT / 'ml/rules.py']
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in protected}
    sms_meta = json.loads((ARTIFACTS / 'sms_model_meta.json').read_text(encoding='utf-8'))
    url_meta = json.loads((ARTIFACTS / 'url_model_meta.json').read_text(encoding='utf-8'))
    sms = joblib.load(ARTIFACTS / 'sms_model.joblib')
    url = joblib.load(ARTIFACTS / 'url_model.joblib')
    messages = pd.read_csv(DATA / 'sms_clean.csv', keep_default_na=False)
    sms_split = json.loads((DATA / 'sms_split.json').read_text(encoding='utf-8'))
    assert sorted(i for part in sms_split.values() for i in part) == list(range(len(messages)))
    test_sms = messages.iloc[sms_split['test']].reset_index(drop=True)
    sms_scores = sms.predict_proba(test_sms.text_raw)[:, list(sms.classes_).index(1)]
    frame = pd.read_parquet(DATA / 'url_features.parquet')
    url_split = json.loads((DATA / 'url_split.json').read_text(encoding='utf-8'))
    assert_no_group_overlap(frame, url_split)
    assert frame['group'].tolist() == frame.url.map(domain_group).tolist()
    assert all(frame.iloc[indices].split.eq(name).all() for name, indices in url_split.items())
    assert sha256((DATA / 'url_split.json').read_bytes()).hexdigest() == url_meta['dataset']['split_sha256']
    assert sha256((DATA / 'url_features.parquet').read_bytes()).hexdigest() == url_meta['dataset']['feature_parquet_sha256']
    test_url = frame.iloc[url_split['test']].reset_index(drop=True)
    matrix = test_url[url_meta['feature_names']]
    url_scores = url.predict_proba(matrix)[:, list(url.classes_).index(1)]
    custom = pd.read_csv(ROOT / 'data/raw/custom_ru_kk.csv', keep_default_na=False)
    languages = {}
    for language in ('ru', 'kk'):
        subset = custom.loc[custom.language.eq(language)]
        scores = sms.predict_proba(subset.text)[:, list(sms.classes_).index(1)]
        combined = [analyze_sms(text) for text in subset.text]
        languages[language] = {'rows': len(subset), 'classifier': metrics(subset.label, scores, sms_meta['threshold']),
            'joint_recall_non_safe': float(np.mean([r.category != 'SAFE' for r, label in zip(combined, subset.label) if label == 1])),
            'joint_recall_fraudulent': float(np.mean([r.category == 'FRAUDULENT' for r, label in zip(combined, subset.label) if label == 1]))}
    demo = []
    for number, (language, text, expected) in enumerate(DEMO, start=1):
        result = analyze_url(text) if language == 'URL' else analyze_sms(text)
        demo.append({'number': number, 'language': language, 'input': text, 'expected': expected,
                     'matches': result.category in expected, 'result': result.to_dict()})
    diagnostics = {
        'python_url': analyze_url('https://www.python.org/downloads/').to_dict(),
        'bare_reserved_example_urls': extract_urls('short.example/path'),
        'code_request_phrases': [{'text': text, 'groups': find_social_engineering_groups(text)} for text in [
            'Назовите код из SMS', 'Продиктуйте код из сообщения', 'Сообщите код подтверждения', 'Никому не сообщайте код подтверждения']],
    }
    result = {'generated_at': datetime.now(timezone.utc).isoformat(),
        'sms': {'rows': len(test_sms), 'threshold': sms_meta['threshold'],
                'historical_pre_refit': sms_meta['test_metrics'], 'saved_model': metrics(test_sms.label, sms_scores, sms_meta['threshold']),
                'latency': latency(sms, [test_sms.text_raw.iloc[0]])},
        'url': {'rows': len(test_url), 't1': url_meta['t1'], 't2': url_meta['t2'],
                'metrics': metrics(test_url.label, url_scores, url_meta['t2']),
                'saved_metrics': url_meta['test_metrics'], 'latency_numeric_features': latency(url, matrix.iloc[:1]),
                'domain_overlap': False, 'dataset_hashes_match': True},
        'languages': languages, 'errors': error_examples(test_sms, sms_scores, sms_meta['threshold'], 'SMS') + error_examples(test_url, url_scores, url_meta['t2'], 'URL'),
        'demo': demo, 'diagnostics': diagnostics, 'protected_sha256': hashes}
    assert all(sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {OUTPUT.relative_to(ROOT)}. No training, threshold changes or network requests.')


if __name__ == '__main__':
    main()
