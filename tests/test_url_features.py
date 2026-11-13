import importlib
import math
import socket
from urllib import request

import pytest
import requests

from ml import url_features as module
from ml.url_features import extract_url_features, explain_url, validate_url


EXPECTED_KEYS = (
    "url_length", "host_length", "path_length", "dot_count", "subdomain_count",
    "host_hyphen_count", "digit_count", "host_digit_ratio", "special_char_count",
    "is_ip", "is_https", "has_nonstandard_port", "has_at", "has_double_slash_path",
    "is_shortener", "suspicious_keyword_count", "has_punycode", "has_mixed_scripts",
    "is_brand_similar", "host_entropy", "has_suspicious_tld", "query_param_count",
)
FLAGS = ("is_ip", "is_https", "has_nonstandard_port", "has_at",
         "has_double_slash_path", "is_shortener", "has_punycode",
         "has_mixed_scripts", "is_brand_similar", "has_suspicious_tld")


def test_normal_https_url_has_all_expected_values():
    url = "https://login.example.com/verify?user=123"
    features = extract_url_features(url)
    assert features == dict(zip(EXPECTED_KEYS, [
        len(url), 17, 7, 2, 1, 0, 3, 0.0, 2, 0, 1, 0, 0, 0, 0, 2,
        0, 0, 0, features["host_entropy"], 0, 1,
    ]))
    assert math.isfinite(features["host_entropy"])


def test_normal_http_url():
    features = extract_url_features("http://example.org")
    assert features["is_https"] == 0
    assert features["path_length"] == features["query_param_count"] == 0
    assert explain_url(features) == ["Не используется HTTPS"]


def test_missing_scheme_only_changes_parsing():
    url = "example.com/path"
    parsed = validate_url(url)
    assert parsed.scheme == "http" and parsed.hostname == "example.com"
    features = extract_url_features(url)
    assert features["url_length"] == len(url)
    assert features["path_length"] == 5 and features["is_https"] == 0


@pytest.mark.parametrize("url", ["", " ", " https://example.com", "https://example.com/ ",
                                 "https://exa mple.com", "https://example.com/\n",
                                 "https://example.com/\x00", "https://example.com\\x", None, 123])
def test_invalid_empty_whitespace_and_type_have_readable_error(url):
    with pytest.raises(ValueError, match="URL"):
        extract_url_features(url)


def test_url_length_boundary():
    prefix = "https://example.com/"
    assert extract_url_features(prefix + "a" * (2048 - len(prefix)))["url_length"] == 2048
    with pytest.raises(ValueError, match="2048"):
        validate_url(prefix + "a" * (2049 - len(prefix)))


@pytest.mark.parametrize("url", ["javascript:alert(1)", "javascript:123", "ftp://example.com",
                                 "file:///example.com", "data:text/plain,hello", "mailto:a@example.com"])
def test_forbidden_schemes(url):
    with pytest.raises(ValueError, match="HTTP"):
        validate_url(url)


@pytest.mark.parametrize("url", ["http:///path", "https://", "https:example.com",
                                 "http://-example.com", "http://example-.com", "http://a..com",
                                 "http://example_.com", "http://xn--.com",
                                 "http://" + "a" * 64 + ".com"])
def test_invalid_hostname(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_ipv4():
    f = extract_url_features("http://192.0.2.1/path")
    assert f["is_ip"] == 1 and f["host_length"] == 9
    assert f["host_digit_ratio"] == pytest.approx(6 / 9)
    assert f["subdomain_count"] == f["is_brand_similar"] == f["has_suspicious_tld"] == 0


def test_ipv6():
    f = extract_url_features("https://[2001:db8::1]:443/a")
    assert f["is_ip"] == 1 and f["host_length"] == 11
    assert f["has_nonstandard_port"] == f["subdomain_count"] == 0
    assert extract_url_features("[2001:db8::1]/a")["is_ip"] == 1


@pytest.mark.parametrize("url", ["http://999.0.2.1", "http://192.0.2.999", "http://192.00.2.1",
                                 "http://1.2.3", "http://[2001:db8:::1]", "http://[::1]junk",
                                 "http://2001:db8::1", "http://[::1"])
def test_invalid_ip_is_rejected(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_subdomains_use_public_suffix_and_reserved_test_suffix():
    assert extract_url_features("https://a.b.example.co.uk")["subdomain_count"] == 2
    assert extract_url_features("https://a.b.example.test")["subdomain_count"] == 2
    assert extract_url_features("https://example.com")["subdomain_count"] == 0


@pytest.mark.parametrize("port,flag", [(80, 0), (443, 0), (8080, 1), (0, 1), (65535, 1)])
def test_ports(port, flag):
    assert extract_url_features(f"https://example.com:{port}")["has_nonstandard_port"] == flag
    assert extract_url_features(f"example.com:{port}")["has_nonstandard_port"] == flag


@pytest.mark.parametrize("port", ["", "abc", "-1", "65536", "80:90"])
def test_invalid_ports(port):
    with pytest.raises(ValueError, match="[Пп]орт"):
        validate_url("https://example.com:" + port)


def test_userinfo_does_not_replace_real_host():
    f = extract_url_features("https://example.com@sub.example.org/path")
    assert f["has_at"] == 1 and f["host_length"] == len("sub.example.org")
    assert f["subdomain_count"] == 1


def test_double_slash_only_counts_in_path():
    assert extract_url_features("https://example.com/a//b")["has_double_slash_path"] == 1
    assert extract_url_features("https://example.com/?next=//x")["has_double_slash_path"] == 0


def test_shortener_list_and_exact_domain_matching():
    assert len(module.SHORTENERS) >= 15
    assert extract_url_features("https://bit.ly/synthetic")["is_shortener"] == 1
    assert extract_url_features("https://bit.ly.example.com/synthetic")["is_shortener"] == 0


def test_keyword_occurrences_are_case_insensitive():
    f = extract_url_features("https://example.com/LOGIN/login/verify?password=x&bonus=free")
    assert f["suspicious_keyword_count"] == 6


def test_punycode_and_unicode_have_same_masking_flags():
    unicode_host = "gооgle.com"  # Две кириллические о.
    ascii_host = unicode_host.encode("idna").decode("ascii")
    a = extract_url_features("https://" + unicode_host)
    b = extract_url_features("https://" + ascii_host)
    for name in ("has_punycode", "has_mixed_scripts", "is_brand_similar"):
        assert a[name] == b[name] == 1


def test_cyrillic_domain_and_latin_suffix_are_mixed():
    assert extract_url_features("https://пример.test")["has_mixed_scripts"] == 1
    assert extract_url_features("https://пример.рф")["has_mixed_scripts"] == 0
    assert extract_url_features("https://пример.рф")["has_punycode"] == 1


def test_brand_distance_and_exact_brand_subdomain():
    assert len(module.BRAND_DOMAINS) >= 20
    assert extract_url_features("https://goggle.com")["is_brand_similar"] == 1
    assert extract_url_features("https://googgle.com")["is_brand_similar"] == 1
    assert extract_url_features("https://login.google.com")["is_brand_similar"] == 0
    assert extract_url_features("https://google.com.example.test")["is_brand_similar"] == 0


def test_levenshtein_insert_delete_replace_and_empty():
    assert module._levenshtein("", "abc") == 3
    assert module._levenshtein("abc", "") == 3
    assert module._levenshtein("abc", "abc") == 0
    assert module._levenshtein("kitten", "sitting") == 3


def test_suspicious_tld_is_not_inferred_from_path():
    assert extract_url_features("https://example.xyz")["has_suspicious_tld"] == 1
    assert extract_url_features("https://example.zip")["has_suspicious_tld"] == 1
    assert extract_url_features("https://example.com/path.xyz")["has_suspicious_tld"] == 0


def test_query_parameters_include_blank_and_repeated_names():
    f = extract_url_features("https://example.com/?a=1&a=2&blank=&flag&&#ignored=1")
    assert f["query_param_count"] == 4


def test_counts_and_entropy():
    f = extract_url_features("http://a-b2.example.test/p_1~%20?x=2&y=3@")
    assert f["host_hyphen_count"] == 1
    assert f["digit_count"] == 6
    assert f["special_char_count"] == 8
    assert extract_url_features("http://aaaa")["host_entropy"] == 0
    assert extract_url_features("http://abab")["host_entropy"] == 1


@pytest.mark.parametrize("url", ["https://example.com", "example.org/path", "http://192.0.2.1",
                                 "https://[2001:db8::1]", "https://пример.рф"])
def test_numeric_values_and_key_order_are_stable(url):
    f = extract_url_features(url)
    assert tuple(f) == EXPECTED_KEYS
    assert len(f) == 22
    assert all(type(value) in (int, float) and math.isfinite(value) for value in f.values())
    assert all(type(f[k]) is int and f[k] in (0, 1) for k in FLAGS)


def test_explanations_describe_flags_without_verdict():
    f = extract_url_features("http://192.0.2.1:8080/login")
    reasons = explain_url(f)
    assert "Вместо доменного имени используется IP-адрес" in reasons
    assert "Используется нестандартный порт" in reasons
    assert "В URL присутствуют подозрительные ключевые слова" in reasons
    assert explain_url(extract_url_features("https://example.com")) == []
    assert explain_url({}) == []


def test_analysis_never_accesses_network_even_on_first_psl_use(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Анализ URL не должен обращаться к сети")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    # Новый extractor ещё не загружал даже встроенный PSL.
    importlib.reload(module)
    assert module.extract_url_features("https://a.example.com/login")["subdomain_count"] == 1
    assert module.extract_url_features("https://пример.рф")["has_punycode"] == 1
