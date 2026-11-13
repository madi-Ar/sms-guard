"""Лексические признаки URL: только разбор строки, без обращения к сети."""

from collections import Counter
import ipaddress
import math
import re
import unicodedata
from urllib.parse import SplitResult, parse_qsl, urlsplit

import tldextract


# Используется встроенный снимок PSL; отключены загрузка и дисковый кеш.
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, fallback_to_snapshot=True,
    extra_suffixes=("test",),
)
SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "lnkd.in",
    "youtu.be", "v.gd", "clck.ru", "s.id", "bl.ink", "short.io", "trib.al",
})
BRAND_DOMAINS = (
    "kaspi.kz", "halykbank.kz", "homebank.kz", "forte.kz", "bcc.kz",
    "sberbank.ru", "tinkoff.ru", "paypal.com", "google.com", "microsoft.com",
    "apple.com", "amazon.com", "facebook.com", "instagram.com", "whatsapp.com",
    "telegram.org", "netflix.com", "ebay.com", "yahoo.com", "yandex.ru",
    "mail.ru", "github.com", "dropbox.com", "booking.com",
)
SUSPICIOUS_WORDS = (
    "login", "verify", "secure", "account", "update", "bank", "confirm",
    "password", "bonus", "free", "prize", "kaspi", "halyk", "signin",
    "wallet", "payment", "reward",
)
SUSPICIOUS_TLDS = frozenset({"xyz", "top", "click", "zip", "work", "loan", "gq", "tk"})
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


def validate_url(url: str) -> SplitResult:
    """Возвращает разобранный URL; при ошибке — ValueError с понятным сообщением.

    http:// добавляется только для разбора, исходная строка не меняется.
    Пробелы (включая крайние), управляющие символы и обратный слеш запрещены.
    """
    if not isinstance(url, str):
        raise ValueError("URL должен быть строкой")
    if not url:
        raise ValueError("URL не должен быть пустым")
    if len(url) > 2048:
        raise ValueError("Длина URL не должна превышать 2048 символов")
    if any(c.isspace() or unicodedata.category(c).startswith("C") for c in url):
        raise ValueError("URL не должен содержать пробелы или управляющие символы")
    if "\\" in url:
        raise ValueError("URL не должен содержать обратный слеш")

    scheme = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):", url)
    # example.com:8080 — hostname с портом, а не схема example.com.
    host_with_port = scheme and ("." in scheme[1] or scheme[1].lower() == "localhost")
    if scheme and not host_with_port:
        if scheme[1].lower() not in {"http", "https"}:
            raise ValueError("Допустимы только схемы HTTP и HTTPS")
        if not url[len(scheme[0]):].startswith("//"):
            raise ValueError("После HTTP или HTTPS требуется ://")
        candidate = url
    else:
        candidate = "http:" + url if url.startswith("//") else "http://" + url
    try:
        parsed = urlsplit(candidate)
        host = parsed.hostname
    except ValueError:
        raise ValueError("Некорректный hostname или IP-адрес в URL") from None
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Допустимы только схемы HTTP и HTTPS")
    if not host:
        raise ValueError("В URL отсутствует hostname")
    try:
        parsed.port
    except ValueError:
        raise ValueError("Порт должен быть целым числом от 0 до 65535") from None
    authority = parsed.netloc.rsplit("@", 1)[-1]
    if authority.startswith("[") and not re.fullmatch(r"\[[^\]]+\](?::[0-9]+)?", authority):
        raise ValueError("Некорректная запись IPv6-адреса или порта")
    if authority.endswith(":"):
        raise ValueError("После двоеточия должен быть указан порт")
    host = host.removesuffix(".")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if ":" in host or authority.startswith("[") or re.fullmatch(r"[0-9.]+", host):
            raise ValueError("Некорректный IP-адрес") from None
        try:
            ascii_host = host.encode("idna").decode("ascii").lower()
            # Проверяет также корректность входных xn-- меток.
            ascii_host.encode("ascii").decode("idna")
        except UnicodeError:
            raise ValueError("Некорректное IDNA-имя домена") from None
        if len(ascii_host) > 253 or not all(_LABEL.fullmatch(p) for p in ascii_host.split(".")):
            raise ValueError("Некорректное доменное имя: проверьте метки и дефисы")
    else:
        if "%" in host:
            raise ValueError("Зона интерфейса в IPv6-адресе не поддерживается")
    return parsed


def _levenshtein(left: str, right: str) -> int:
    """Расстояние редактирования: вставка, удаление, замена за единицу."""
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def extract_url_features(url: str) -> dict:
    """Возвращает ровно 22 числовых признака в постоянном порядке.

    Счётчики строки относятся к исходному URL. Host — без порта, скобок IPv6
    и завершающей точки, в нижнем регистре. Для IDNA-анализа используется
    каноническая ASCII-форма, для смешения алфавитов — декодированная.
    """
    parsed = validate_url(url)
    host = parsed.hostname.removesuffix(".")
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False
    subdomains, registered, suffix = 0, "", ""
    ascii_host = host
    unicode_host = host
    if not is_ip:
        ascii_host = host.encode("idna").decode("ascii").lower()
        unicode_host = ascii_host.encode("ascii").decode("idna")
        domain = _EXTRACT(ascii_host)
        subdomains = len(domain.subdomain.split(".")) if domain.subdomain else 0
        registered = domain.top_domain_under_public_suffix
        suffix = domain.suffix
    latin = any("LATIN" in unicodedata.name(c, "") for c in unicode_host)
    cyrillic = any("CYRILLIC" in unicodedata.name(c, "") for c in unicode_host)
    decoded_registered = registered.encode("ascii").decode("idna") if registered else ""
    similar_brand = bool(registered and registered not in BRAND_DOMAINS and any(
        0 < _levenshtein(decoded_registered, brand) <= 2 for brand in BRAND_DOMAINS
    ))
    entropy = -sum((n / len(host)) * math.log2(n / len(host)) for n in Counter(host).values())
    lowered = url.lower()
    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(parsed.path),
        "dot_count": url.count("."),
        "subdomain_count": subdomains,
        "host_hyphen_count": host.count("-"),
        "digit_count": sum(c in "0123456789" for c in url),
        "host_digit_ratio": sum(c in "0123456789" for c in host) / len(host),
        "special_char_count": sum(url.count(c) for c in "@~%=&_?"),
        "is_ip": int(is_ip),
        "is_https": int(parsed.scheme.lower() == "https"),
        "has_nonstandard_port": int(parsed.port is not None and parsed.port not in {80, 443}),
        "has_at": int("@" in url),
        "has_double_slash_path": int("//" in parsed.path),
        "is_shortener": int(any(ascii_host == d or ascii_host.endswith("." + d) for d in SHORTENERS)),
        "suspicious_keyword_count": sum(lowered.count(word) for word in SUSPICIOUS_WORDS),
        "has_punycode": int(any(p.startswith("xn--") for p in ascii_host.split("."))),
        "has_mixed_scripts": int(latin and cyrillic),
        "is_brand_similar": int(similar_brand),
        "host_entropy": float(entropy),
        "has_suspicious_tld": int(suffix.split(".")[-1] in SUSPICIOUS_TLDS),
        "query_param_count": len(parse_qsl(parsed.query, keep_blank_values=True)),
    }


def explain_url(features: dict) -> list[str]:
    """Объясняет наблюдаемые признаки, не вынося вердикт о мошенничестве."""
    reasons = (
        ("is_ip", "Вместо доменного имени используется IP-адрес"),
        ("has_nonstandard_port", "Используется нестандартный порт"),
        ("has_at", "В URL присутствует символ @; проверяйте фактическое имя хоста"),
        ("has_double_slash_path", "В пути URL присутствует двойной слеш"),
        ("is_shortener", "Используется сервис сокращения ссылок"),
        ("suspicious_keyword_count", "В URL присутствуют подозрительные ключевые слова"),
        ("has_punycode", "Домен представлен или может быть представлен в Punycode"),
        ("has_mixed_scripts", "В доменном имени обнаружено смешение алфавитов"),
        ("is_brand_similar", "Домен похож на название известного сервиса"),
        ("has_suspicious_tld", "Доменная зона является потенциальным признаком риска, но не доказательством мошенничества"),
    )
    result = [message for key, message in reasons if features.get(key, 0)]
    if "is_https" in features and not features["is_https"]:
        result.append("Не используется HTTPS")
    return result
