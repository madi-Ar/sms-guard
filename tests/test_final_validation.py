"""Проверки задания №18. Production-модели не обучаются и не перезаписываются."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import Mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import Client
import joblib
import numpy as np
import pandas as pd
import pytest

from apps.analysis import services, views
from apps.analysis.api import CheckThrottle
from apps.analysis.models import Check
from apps.analysis.services import AnalysisResult
from ml import baseline, calibrate_risk, compare_sms, features, predictor, preprocessing, url_model


def evidence(name, value):
    target = os.getenv('TASK18_EVIDENCE_PATH')
    if target:
        path = Path(target)
        data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        data[name] = value
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


@pytest.fixture
def signed(client, db):
    cache.clear()
    user = get_user_model().objects.create_user('final_test_user')
    client.force_login(user)
    yield client, user
    cache.clear()


def fake_analysis(monkeypatch):
    analyze = Mock(return_value=AnalysisResult(language='en', category='SAFE', risk_level='LOW', risk_score=10,
        p_sms=.1, explanation=['Учебная проверка.'], recommendation='Проверьте отправителя.'))
    monkeypatch.setattr(services, 'analyze_sms', analyze)
    return analyze


def test_sql_injection_input_and_filters(signed, monkeypatch):
    client, user = signed
    payload = "' OR 1=1 --"
    analyze = fake_analysis(monkeypatch)
    other = get_user_model().objects.create_user('final_other')
    foreign = Check.objects.create(user=other, type='sms', category='SAFE', masked_input='PRIVATE_OTHER')
    response = client.post('/api/v1/check/sms/', {'text': payload})
    assert response.status_code == 201
    analyze.assert_called_once_with(payload)
    for field in ('category', 'type', 'date_from', 'date_to', 'risk_level'):
        assert client.get('/api/v1/checks/', {field: payload}).status_code == 400
        assert client.get('/history/', {field: payload}).status_code == 400
    assert client.get('/api/v1/checks/', {'user': other.pk}).json()['count'] == 1
    assert client.get(f'/api/v1/checks/{foreign.pk}/').status_code == 404
    assert get_user_model().objects.count() == 2


def test_exact_xss_payload_history_result_dashboard(signed, monkeypatch):
    client, user = signed
    payload = '<script>alert(1)</script>'
    fake_analysis(monkeypatch)
    response = client.post('/api/v1/check/sms/', {'text': payload})
    assert response.status_code == 201
    pk = response.json()['id']
    history = client.get('/history/').content.decode()
    assert payload not in history and '&lt;script&gt;alert(1)&lt;/script&gt;' in history
    assert payload not in client.get(f'/checks/{pk}/').content.decode()
    assert payload not in client.get('/api/v1/stats/summary/').content.decode()
    source = (Path(__file__).resolve().parents[1] / 'static/js/analysis.js').read_text(encoding='utf-8')
    assert 'textContent' in source and 'innerHTML' not in source


def test_delete_history_ignores_foreign_user_parameter(signed):
    client, user = signed
    other = get_user_model().objects.create_user('history_other')
    own = Check.objects.create(user=user, type='sms', category='SAFE', masked_input='mine')
    foreign = Check.objects.create(user=other, type='sms', category='SAFE', masked_input='theirs')
    client.post('/profile/?user=' + str(other.pk), {'action': 'delete_history', 'confirm': 'yes', 'user_id': other.pk})
    assert not Check.objects.filter(pk=own.pk).exists()
    assert Check.objects.filter(pk=foreign.pk).exists()


@pytest.mark.parametrize('path', ['/api/v1/check/sms/', '/api/v1/check/url/', '/api/v1/auth/logout/', '/profile/'])
def test_state_changes_reject_missing_csrf(signed, path):
    _, user = signed
    strict = Client(enforce_csrf_checks=True)
    strict.force_login(user)
    assert strict.post(path, {'text': 'Test SMS', 'url': 'https://example.com', 'action': 'preferences'}).status_code == 403


@pytest.mark.parametrize('payload', [b'\xff\xfe\x00', b'{"text":"\xff"}', b'{"text": [0, 255]}', b'{"text": null}'])
def test_binary_or_invalid_json_is_400(signed, payload):
    client, _ = signed
    assert client.post('/api/v1/check/sms/', data=payload, content_type='application/json').status_code == 400


@pytest.mark.parametrize('name', ['sms_model.joblib', 'url_model.joblib', 'sms_model_meta.json', 'url_model_meta.json', 'corrupt_sms_json', 'corrupt_url_json'])
def test_missing_and_corrupt_artifacts_return_503(signed, tmp_path, monkeypatch, name, caplog):
    client, _ = signed
    for path in predictor.ARTIFACTS.glob('*model*'):
        shutil.copy2(path, tmp_path / path.name)
    if name.startswith('corrupt'):
        kind = 'sms' if name == 'corrupt_sms_json' else 'url'
        (tmp_path / f'{kind}_model_meta.json').write_text('{invalid metadata', encoding='utf-8')
    else:
        (tmp_path / name).unlink()
    monkeypatch.setattr(predictor, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(views, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(predictor.ModelPredictor, '_shared_bundle', None)
    monkeypatch.setattr(predictor.ModelPredictor, '_shared_error', None)
    monkeypatch.setattr(services, 'get_predictor', lambda: predictor.ModelPredictor())
    assert client.get('/api/v1/health/').status_code == 200
    response = client.post('/api/v1/check/sms/', {'text': 'PRIVATE_FAILURE_MARKER'})
    assert response.status_code == 503 and response.json()['error']['code'] == 'MODEL_UNAVAILABLE'
    assert Check.objects.count() == 0
    assert 'PRIVATE_FAILURE_MARKER' not in caplog.text


def test_runtime_latency_one_hundred_real_requests(signed, monkeypatch, capsys):
    client, _ = signed
    text = 'Review https://alpha.example/a https://beta.test/b https://example.com/c before our meeting.'
    started = perf_counter()
    assert len(services.analyze_sms(text).urls) == 3
    warmup_ms = (perf_counter() - started) * 1000
    clock = [1000.]
    monkeypatch.setattr(CheckThrottle, 'timer', staticmethod(lambda: clock[0]))
    times = []
    for _ in range(100):
        # Simulated 3-second spacing respects the unchanged 30/min limiter.
        clock[0] += 3
        started = perf_counter()
        response = client.post('/api/v1/check/sms/', {'text': text})
        times.append((perf_counter() - started) * 1000)
        assert response.status_code == 201
        assert len(response.json()['urls']) == 3
    result = {'requests': len(times), 'mean_ms': float(np.mean(times)), 'max_ms': max(times),
              'warmup_ms': warmup_ms, 'limit_ms': 500, 'passes': max(times) <= 500,
              'real_models': True, 'throttle_clock_spacing_seconds': 3}
    evidence('sms_100_requests', result)
    with capsys.disabled(): print('\nTask18 SMS latency', json.dumps(result))
    # Performance failure is reported as FAIL in the report, not hidden by altering a bound.
    assert Check.objects.count() == 100


def test_dashboard_series_one_thousand_rows(signed, capsys):
    client, user = signed
    Check.objects.bulk_create([Check(user=user, type='sms', category='SAFE', risk_score=10,
                                   masked_input='Artificial load record') for _ in range(1000)])
    result = {}
    for name, path in [('api', '/api/v1/stats/summary/'), ('html', '/')]:
        times = []
        for _ in range(10):
            started = perf_counter()
            response = client.get(path)
            times.append((perf_counter() - started) * 1000)
            assert response.status_code == 200
        result[name] = {'requests': len(times), 'mean_ms': float(np.mean(times)), 'max_ms': max(times), 'passes': max(times) <= 2000}
    evidence('dashboard_1000_rows', result)
    with capsys.disabled(): print('\nTask18 Dashboard latency', json.dumps(result))


def test_forty_requests_enforce_thirty_per_minute(signed, monkeypatch, capsys):
    client, _ = signed
    fake_analysis(monkeypatch)
    monkeypatch.setattr(CheckThrottle, 'timer', staticmethod(lambda: 1000.))
    times, statuses = [], []
    for _ in range(40):
        started = perf_counter()
        statuses.append(client.post('/api/v1/check/sms/', {'text': 'Rate limit test'}).status_code)
        times.append((perf_counter() - started) * 1000)
    assert statuses == [201] * 30 + [429] * 10 and Check.objects.count() == 30
    result = {'requests': 40, 'accepted': 30, 'rejected_429': 10, 'mean_ms': float(np.mean(times)), 'max_ms': max(times)}
    evidence('rate_limit_40_requests', result)
    with capsys.disabled(): print('\nTask18 rate limit', json.dumps(result))


def test_five_real_model_flows_save_history_detail_dashboard(signed):
    client, _ = signed
    scenarios = [
        ('A', 'sms', 'Please bring the blue notebook tomorrow.'),
        ('B', 'sms', 'Мама, у меня проблемы, срочно переведи 50 000 на этот номер, потом всё объясню. Никому не говори.'),
        ('C', 'sms', 'Ваша карта заблокирована! Срочно подтвердите данные по ссылке: http://secure-bank-verify.example/login'),
        ('D', 'url', 'https://www.example.com/about'),
        ('E', 'sms', 'Сотрудник службы безопасности банка: назовите код из SMS для отмены подозрительной операции.'),
    ]
    results = []
    for number, (name, kind, text) in enumerate(scenarios, start=1):
        field = 'text' if kind == 'sms' else 'url'
        response = client.post(f'/api/v1/check/{kind}/', {field: text})
        assert response.status_code == 201
        data = response.json()
        assert client.get(f"/checks/{data['id']}/").status_code == 200
        assert client.get(f"/api/v1/checks/{data['id']}/").status_code == 200
        assert client.get('/history/').status_code == 200
        assert data['id'] in [row['id'] for row in client.get('/api/v1/checks/').json()['results']]
        assert client.get('/').status_code == 200
        stats = client.get('/api/v1/stats/summary/').json()
        assert stats['sms_count'] + stats['url_count'] == number
        results.append({'scenario': name, 'category': data['category'], 'risk_score': data.get('risk_score'),
                        'history_detail_dashboard': 'PASS', 'real_models': True})
    evidence('five_real_http_flows', results)


def test_preprocessing_workflow_uses_only_temporary_data(tmp_path, monkeypatch):
    frame = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'text': ['Hello', 'HELLO', '', 'Offer 1000 тг', 'hello'], 'label': [0, 0, 0, 1, 1]})
    frame.to_csv(tmp_path / 'sms_raw.csv', index=False)
    monkeypatch.setattr(preprocessing, 'DATA_DIR', tmp_path)
    preprocessing.main()
    clean = pd.read_csv(tmp_path / 'sms_clean.csv')
    assert clean.id.tolist() == [4] and clean.text_clean.tolist() == ['offer <money>']
    assert pd.read_csv(tmp_path / 'sms_raw.csv').shape[0] == 5


def test_feature_workflow_uses_synthetic_partition(tmp_path, monkeypatch):
    frame = pd.DataFrame({'id': range(20), 'text_raw': ['Meeting library tomorrow ' + str(i) if i % 2 == 0 else 'Urgent prize claim reward ' + str(i) for i in range(20)], 'label': [i % 2 for i in range(20)]})
    frame.to_csv(tmp_path / 'sms_clean.csv', index=False)
    monkeypatch.setattr(features, 'DATA_DIR', tmp_path)
    features.main()
    split = json.loads((tmp_path / 'sms_split.json').read_text())
    assert sorted(i for indices in split.values() for i in indices) == list(range(20))


def test_baseline_cli_with_saved_model_and_no_training(tmp_path, monkeypatch):
    model = joblib.load(baseline.MODEL_PATH)
    frame = pd.DataFrame({'id': [1, 2], 'text_raw': ['Meet at the library.', 'URGENT claim free cash prize now'], 'label': [0, 1]})
    train = Mock(return_value=model)
    monkeypatch.setattr(baseline, 'load_partitions', lambda: (frame, frame))
    monkeypatch.setattr(baseline, 'train_baseline', train)
    monkeypatch.setattr(baseline, 'MODEL_PATH', tmp_path / 'baseline.joblib')
    baseline.main()
    train.assert_called_once_with(frame)
    assert (tmp_path / 'baseline.joblib').is_file()


def test_comparison_metrics_and_figures_without_fitting(tmp_path, monkeypatch):
    (tmp_path / 'docs').mkdir()
    monkeypatch.setattr(compare_sms, 'ROOT', tmp_path)
    result = compare_sms.metrics([0, 1, 0, 1], np.array([.1, .9, .2, .8]), .5)
    assert result['confusion_matrix'] == [[2, 0], [0, 2]] and result['f2'] == 1
    compare_sms.plot_curves([0, 1, 0, 1], {'diagnostic': [.1, .9, .2, .8]})
    assert all((tmp_path / 'docs' / name).stat().st_size > 0 for name in ['08_pr.png', '08_roc.png'])


def test_url_report_writer_uses_saved_metadata_in_temporary_directory(tmp_path, monkeypatch):
    metadata = json.loads((url_model.ARTIFACTS / 'url_model_meta.json').read_text(encoding='utf-8'))
    (tmp_path / 'docs').mkdir()
    monkeypatch.setattr(url_model, 'ROOT', tmp_path)
    url_model.write_report(metadata)
    report = (tmp_path / 'docs/11_url_model.md').read_text(encoding='utf-8')
    assert 'RandomForestClassifier' in report and '0.981293' in report


def test_calibration_observations_without_recalibration():
    model = SimpleNamespace(predict_sms=lambda text: {'probability': .2},
        predict_url_features=lambda flags: {'probability': .3}, url_metadata={'dataset': {'strip_www': True}})
    frame = pd.DataFrame({'text': ['hello', 'Visit https://example.com'], 'label': [0, 1]})
    observations = calibrate_risk.collect_observations(frame, model, 'text')
    assert observations[0]['p_url_max'] is None and observations[1]['p_url_max'] == .3
    row = {'en_validation': {'recall_medium_high': .95, 'false_high_rate': .03}, 'ru_kk_development': {'recall_medium_high': .95, 'false_high_rate': .03}}
    assert calibrate_risk.violation(row) == 0
    assert 'ml/artifacts/sms_model.joblib' in calibrate_risk.input_hashes()
    with pytest.raises(ValueError):
        calibrate_risk.evaluate(observations[:1], calibrate_risk.load_config(), .5)


@pytest.mark.parametrize('defect', ['sizes', 'overlap', 'duplicate_text'])
def test_sms_partition_guards_reject_bad_saved_data(tmp_path, monkeypatch, defect):
    data = pd.read_csv(compare_sms.DATA_DIR / 'sms_clean.csv', keep_default_na=False)
    split = json.loads((compare_sms.DATA_DIR / 'sms_split.json').read_text(encoding='utf-8'))
    if defect == 'sizes':
        split['train'].pop()
    elif defect == 'overlap':
        split['test'][0] = split['train'][0]
    else:
        data.loc[1, 'text_clean'] = data.loc[0, 'text_clean']
    data.to_csv(tmp_path / 'sms_clean.csv', index=False)
    (tmp_path / 'sms_split.json').write_text(json.dumps(split), encoding='utf-8')
    monkeypatch.setattr(compare_sms, 'DATA_DIR', tmp_path)
    with pytest.raises(ValueError):
        compare_sms.load_partitions()


def test_calibration_rejects_invalid_partition_before_loading_or_fitting(tmp_path, monkeypatch):
    folder = tmp_path / 'data/processed'
    folder.mkdir(parents=True)
    (folder / 'sms_split.json').write_text(json.dumps({'train': [0], 'validation': [1], 'test': []}))
    pd.DataFrame({'text_raw': ['hello', 'urgent'], 'label': [0, 1]}).to_csv(folder / 'sms_clean.csv', index=False)
    monkeypatch.setattr(calibrate_risk, 'ROOT', tmp_path)
    monkeypatch.setattr(calibrate_risk, 'input_hashes', lambda: {'unchanged': 'test'})
    load = Mock(side_effect=AssertionError('Model loading must not occur after invalid split'))
    monkeypatch.setattr(calibrate_risk.joblib, 'load', load)
    with pytest.raises(ValueError, match='Train/Validation'):
        calibrate_risk.calibrate()
    load.assert_not_called()
