from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import socket
from urllib import request

import numpy as np
import pandas as pd
import pytest
import requests

from apps.analysis import services
from ml import features, predictor
from ml.predictor import ModelPredictor
from ml.rules import RULES, find_social_engineering_groups
from ml.url_dataset import FEATURE_NAMES


class FakeModel:
    # Класс 1 намеренно находится в первой, а не второй колонке.
    classes_ = np.array([1, 0])

    def __init__(self, scores=.4):
        self.scores = scores if isinstance(scores, list) else [scores]
        self.calls = []

    def predict_proba(self, values):
        score = self.scores[min(len(self.calls), len(self.scores) - 1)]
        self.calls.append(values.copy())
        return np.array([[score, 1 - score]])


def metadata():
    return ({"threshold": .3, "version": "sms-test"},
            {"t1": .2, "t2": .8, "version": "url-test", "feature_names": list(FEATURE_NAMES),
             "dataset": {"strip_www": True}})


@pytest.fixture
def fake_predictor(monkeypatch):
    sms, url = FakeModel(.4), FakeModel([.1, .5, .9])
    sms_meta, url_meta = metadata()
    instance = ModelPredictor(sms_model=sms, url_model=url, sms_meta=sms_meta, url_meta=url_meta)
    monkeypatch.setattr(services, "get_predictor", lambda: instance)
    return instance, sms, url


def test_sms_without_urls_fills_risk_but_does_not_fake_storage(fake_predictor):
    result = services.analyze_sms("Bring the blue notebook tomorrow.")
    assert isinstance(result, services.AnalysisResult)
    assert result.type == "sms" and result.language == "en"
    assert result.urls == [] and result.p_url_max is None
    assert result.p_sms == .4 and result.features["sms_label"] == 1
    assert result.features["social_engineering_groups"] == []
    for key in ("id", "created_at"):
        assert getattr(result, key) is None
    assert type(result.risk_score) is int and 0 <= result.risk_score <= 100
    assert result.category in {"SAFE", "SUSPICIOUS", "FRAUDULENT"}
    assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}
    assert result.recommendation and 1 <= len(result.explanation) <= 6
    assert result.model_versions == {"sms": "sms-test", "url": "url-test"}
    assert "вероятностный" in result.disclaimer


def test_sms_with_one_url(fake_predictor):
    result = services.analyze_sms("Open https://example.com/a?q=1.")
    assert [u.url for u in result.urls] == ["https://example.com/a?q=1"]
    assert result.p_url_max == .1 and result.urls[0].category == "SAFE"
    assert result.category in {"SAFE", "SUSPICIOUS", "FRAUDULENT"}
    assert services.URL_WARNING in result.explanation


@pytest.mark.parametrize("count", [3, 4])
def test_only_first_three_urls_are_analyzed_in_order(fake_predictor, count):
    text = " ".join(f"https://host{i}.example.com/path" for i in range(count))
    result = services.analyze_sms(text)
    assert [r.url for r in result.urls] == [f"https://host{i}.example.com/path" for i in range(3)]
    assert [r.category for r in result.urls] == ["SAFE", "SUSPICIOUS", "DANGEROUS"]
    assert result.p_url_max == .9
    assert len(fake_predictor[2].calls) == 3
    assert len(fake_predictor[1].calls) == 1  # Risk Score не вызывает повторных предсказаний.


def test_standalone_url_has_original_url_probability_and_all_features(fake_predictor):
    result = services.analyze_url("https://WWW.example.com/?Token=XY")
    assert isinstance(result, services.URLAnalysis)
    assert result.url == "https://WWW.example.com/?Token=XY"
    assert result.p_url == .1 and result.category == "SAFE"
    assert tuple(result.features) == FEATURE_NAMES
    assert result.features["subdomain_count"] == 0  # www нормализован как при обучении.
    assert result.features["path_length"] == 0
    assert result.to_dict()["features"] == result.features
    assert services.URL_WARNING in result.explanation


@pytest.mark.parametrize("url", ["example.org/Path?Token=XY", "bit.ly/abc123", "www.example.com", "https://пример.рф/путь?код=123"])
def test_url_formats_and_cyrillic_are_supported(fake_predictor, url):
    assert services.extract_urls("Ссылка: " + url + ".") == [url]
    assert 0 <= services.analyze_url(url).p_url <= 1


def test_email_ordinary_text_and_decimal_numbers_are_not_urls():
    assert services.extract_urls("Пишите user@example.com и имя@пример.рф. Версия 1.2.3, file.txt, т.д.") == []
    assert services.extract_urls("user@example.com/path other@example.org; https://example.net") == ["https://example.net"]


def test_punctuation_balanced_parentheses_and_ipv6():
    text = 'См. (https://example.com/a?q=(x)), https://[2001:db8::1]/, bit.ly/abc!'
    assert services.extract_urls(text) == ["https://example.com/a?q=(x)", "https://[2001:db8::1]/", "bit.ly/abc"]


def test_invalid_potential_url_does_not_abort_sms(fake_predictor):
    result = services.analyze_sms("Visit http://999.0.2.1/a and https://example.org.")
    assert len(result.urls) == 1 and result.urls[0].url == "https://example.org"
    with pytest.raises(ValueError):
        services.analyze_url("ftp://example.com")


def test_individual_url_validation_failure_does_not_abort_sms(fake_predictor, monkeypatch):
    def invalid(url):
        raise ValueError("Некорректная ссылка")

    monkeypatch.setattr(services, "analyze_url", invalid)
    result = services.analyze_sms("Visit https://example.org.")
    assert result.p_sms == .4 and result.urls == [] and result.p_url_max is None


@pytest.mark.parametrize("text,language", [("Срочно пришлите код подтверждения", "ru"),
                                         ("Шұғыл кодты жіберіңіз", "kk"), ("12345 !!!", "und")])
def test_language_warnings(fake_predictor, text, language):
    result = services.analyze_sms(text)
    assert result.language == language
    if language in {"ru", "kk"}:
        assert any("Надёжность" in warning and "снижена" in warning for warning in result.explanation)
    else:
        assert any("определения языка" in warning for warning in result.explanation)


def test_all_six_rule_groups_have_ten_expressions_per_language_and_match():
    assert tuple(RULES) == ("URGENCY", "MONEY_TRANSFER", "BANK_CARD", "ACCOUNT_BLOCK", "PRIZE", "CODE_REQUEST")
    for group, languages in RULES.items():
        assert set(languages) == {"ru", "kk", "en"}
        for phrases in languages.values():
            assert len(set(phrases)) >= 10
            for phrase in phrases:
                assert group in find_social_engineering_groups(phrase.upper())


def test_phrase_whitespace_case_boundaries_and_group_deduplication():
    assert find_social_engineering_groups("URGENT! urgent! Send\n  money; VERIFICATION CODE") == ["URGENCY", "MONEY_TRANSFER", "CODE_REQUEST"]
    assert find_social_engineering_groups("urgently noturgent urgentness xсрочно срочноx") == []


def test_sms_with_several_rule_groups(fake_predictor):
    result = services.analyze_sms("URGENT! transfer money; card number; account blocked; claim your prize; send the code")
    assert result.features["social_engineering_groups"] == list(RULES)
    assert result.risk_score >= 70 and result.recommendation
    assert result.risk_level == "HIGH" and result.category == "FRAUDULENT"


def test_risk_keeps_warnings_and_url_category_without_new_predictions(fake_predictor):
    result = services.analyze_sms("Срочно! URGENT transfer money; card number; account blocked; "
                                  "claim your prize; send the code https://example.com")
    assert result.category == "FRAUDULENT" and result.urls[0].category == "SAFE"
    assert len(result.explanation) <= 6
    assert any("снижена" in reason for reason in result.explanation)
    assert services.URL_WARNING in result.explanation
    assert result.disclaimer and result.id is None and result.created_at is None
    assert len(fake_predictor[1].calls) == len(fake_predictor[2].calls) == 1


def test_sms_reaches_predictor_without_preprocessing(fake_predictor):
    text = "  URGENT £20! Visit HTTPS://example.com/Case?Token=XY  "
    services.analyze_sms(text)
    assert fake_predictor[1].calls == [[text]]


def test_json_serialization_contains_all_required_fields(fake_predictor):
    result = services.analyze_sms("Visit https://example.com")
    payload = result.to_dict()
    assert set(payload) == {"id", "type", "language", "category", "risk_level", "risk_score", "p_sms",
        "p_url_max", "urls", "features", "explanation", "recommendation", "disclaimer", "model_versions", "created_at"}
    assert json.loads(json.dumps(payload))["urls"][0]["category"] == "SAFE"
    payload["features"]["sms_label"] = 99
    assert result.features["sms_label"] != 99  # asdict делает независимый словарь.


@pytest.mark.parametrize("text", ["", "  \n ", None, 123])
def test_empty_or_non_string_sms_has_readable_error(text):
    with pytest.raises(ValueError, match="SMS"):
        services.analyze_sms(text)


@pytest.mark.parametrize("score,expected", [(.299, 0), (.3, 1), (.5, 1)])
def test_sms_threshold_is_taken_from_metadata(fake_predictor, score, expected):
    instance, sms, _ = fake_predictor
    sms.scores = [score]
    assert instance.predict_sms("unchanged")["label"] == expected


@pytest.mark.parametrize("score,expected", [(.199, "SAFE"), (.2, "SUSPICIOUS"), (.8, "DANGEROUS")])
def test_url_thresholds_and_metadata_column_order(fake_predictor, score, expected):
    instance, _, url = fake_predictor
    url.scores = [score]
    result = services.analyze_url("https://example.com")
    assert result.category == expected
    assert url.calls[-1].columns.tolist() == instance.url_metadata["feature_names"]


def prepare_fake_artifacts(tmp_path, monkeypatch):
    sms_meta, url_meta = metadata()
    for name, meta in (("sms", sms_meta), ("url", url_meta)):
        (tmp_path / f"{name}_model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(predictor, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(ModelPredictor, "_shared_bundle", None)
    monkeypatch.setattr(ModelPredictor, "_shared_error", None)
    return FakeModel(), FakeModel()


def test_each_final_model_loads_once_even_across_instances_threads(tmp_path, monkeypatch, caplog):
    sms, url = prepare_fake_artifacts(tmp_path, monkeypatch)
    loaded = []

    def load(path):
        loaded.append(path.name)
        return sms if path.name == "sms_model.joblib" else url

    monkeypatch.setattr(predictor.joblib, "load", load)
    instances = [ModelPredictor() for _ in range(8)]
    with caplog.at_level("INFO", logger="ml.predictor"):
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda instance: instance.predict_sms("private-secret-text"), instances))
    assert loaded == ["sms_model.joblib", "url_model.joblib"]
    assert all(instance.versions == {"sms": "sms-test", "url": "url-test"} for instance in instances)
    assert "загружены" in caplog.text and "private-secret-text" not in caplog.text


@pytest.mark.parametrize("failure", ["missing", "json", "joblib"])
def test_missing_or_corrupt_artifacts_have_safe_cached_error(tmp_path, monkeypatch, failure):
    prepare_fake_artifacts(tmp_path, monkeypatch)
    calls = []
    if failure == "missing":
        (tmp_path / "sms_model_meta.json").unlink()
    elif failure == "json":
        (tmp_path / "sms_model_meta.json").write_text("invalid json", encoding="utf-8")

    def broken(path):
        calls.append(path.name)
        raise EOFError("secret model contents")

    monkeypatch.setattr(predictor.joblib, "load", broken)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="итоговые SMS/URL-модели") as error:
            ModelPredictor().predict_sms("text")
        assert "secret" not in str(error.value)
    assert len(calls) == (1 if failure == "joblib" else 0)


def test_fake_injection_never_uses_joblib(fake_predictor, monkeypatch):
    monkeypatch.setattr(predictor.joblib, "load", lambda path: pytest.fail("Unexpected joblib load"))
    services.analyze_sms("Visit https://example.com")


def test_bad_model_probability_is_not_returned_as_a_result(fake_predictor):
    fake_predictor[1].scores = [np.nan]
    with pytest.raises(RuntimeError, match="вероятность"):
        services.analyze_sms("Text")


def test_analysis_does_not_access_network(fake_predictor, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Анализ SMS и URL не должен обращаться к сети")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    assert len(services.analyze_sms("См https://пример.рф и bit.ly/abc123").urls) == 2
    assert services.analyze_url("https://example.org").p_url == .9


def test_real_final_models_normalize_sms_once_and_preserve_artifacts(monkeypatch):
    artifacts = Path(predictor.ARTIFACTS)
    paths = [artifacts / name for name in ("sms_model.joblib", "sms_model_meta.json", "url_model.joblib", "url_model_meta.json")]
    before = {p:sha256(p.read_bytes()).hexdigest() for p in paths}
    monkeypatch.setattr(ModelPredictor, "_shared_bundle", None)
    monkeypatch.setattr(ModelPredictor, "_shared_error", None)
    instance = ModelPredictor()
    monkeypatch.setattr(services, "get_predictor", lambda: instance)
    calls = []
    original = features.normalize_sms

    def counted(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(features, "normalize_sms", counted)
    text = "Bring the blue notebook tomorrow. Visit https://example.com"
    result = services.analyze_sms(text)
    assert calls == [text]
    assert "<url>" in original(text)
    assert 0 <= result.p_sms <= 1 and 0 <= result.p_url_max <= 1
    assert result.model_versions == {"sms": "1.0.0", "url": "1.0.0"}
    assert result.features["sms_label"] == int(result.p_sms >= instance.sms_metadata["threshold"])
    assert {p:sha256(p.read_bytes()).hexdigest() for p in paths} == before
