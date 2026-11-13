"""Совместный анализ без Risk Score, сохранения, API и сетевых обращений."""

from dataclasses import asdict, dataclass, field
import re

import tldextract

from ml.predictor import get_predictor
from ml.rules import find_social_engineering_groups
from ml.url_dataset import normalize_url
from ml.url_features import explain_url, extract_url_features, validate_url


DISCLAIMER = ("Оценка носит вероятностный характер и не гарантирует безопасность или отсутствие "
              "мошенничества. SMS-модель оценивает spam, а не доказанный факт мошенничества.")
URL_WARNING = ("URL-модель зависит от артефактов обучающего датасета и может ошибочно "
               "помечать обычные ссылки как опасные.")
_SUFFIXES = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, extra_suffixes=("test",))
_LABEL = r"[^\W_](?:[^\W_]|-)*"
_DOMAIN = rf"(?:{_LABEL}\.)+(?:xn--[a-z0-9-]+|[^\W\d_]{{2,63}})(?![\w@.-])"
_URLS = re.compile(
    r"(?P<email>[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+)"
    r"|(?P<url>(?<![\w@./-])(?:"
    r"(?:https?://|www\.)[^\s<>\"«»]+"
    rf"|{_DOMAIN}(?::[0-9]+)?(?:[/?#][^\s<>\"«»]*)?))", re.IGNORECASE,
)


@dataclass
class URLAnalysis:
    url: str
    p_url: float
    category: str
    features: dict
    explanation: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    id: str | None = None
    type: str = "sms"
    language: str | None = None
    category: str | None = None
    risk_level: str | None = None
    risk_score: float | None = None
    p_sms: float | None = None
    p_url_max: float | None = None
    urls: list[URLAnalysis] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)
    recommendation: str | None = None
    disclaimer: str = DISCLAIMER
    model_versions: dict = field(default_factory=dict)
    created_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _trim_punctuation(url):
    url = url.rstrip(".,!?;:»”\"'")
    pairs = {")": "(", "]": "[", "}": "{"}
    while url and url[-1] in pairs and url.count(url[-1]) > url.count(pairs[url[-1]]):
        url = url[:-1].rstrip(".,!?;:»”\"'")
    return url


def extract_urls(text: str) -> list[str]:
    urls = []
    for match in _URLS.finditer(text):
        if match.group("email"):
            continue
        url = _trim_punctuation(match.group("url"))
        try:
            parsed = validate_url(url)
            # Для голого домена требуется известный суффикс: file.txt не ссылка.
            if not url.lower().startswith(("http://", "https://", "www.")):
                if not _SUFFIXES(parsed.hostname.encode("idna").decode("ascii")).suffix:
                    continue
        except ValueError:
            continue
        urls.append(url)
        if len(urls) == 3:
            break
    return urls


def detect_language(text: str) -> str:
    """Детерминированная учебная эвристика; und — недостаточно букв."""
    if re.search("[әғқңөұүһі]", text, re.IGNORECASE):
        return "kk"
    if re.search("[а-яё]", text, re.IGNORECASE):
        return "ru"
    if re.search("[a-z]", text, re.IGNORECASE):
        return "en"
    return "und"


def analyze_url(url: str) -> URLAnalysis:
    validate_url(url)
    predictor = get_predictor()
    normalized = normalize_url(url, strip_www=predictor.url_metadata["dataset"]["strip_www"])
    features = extract_url_features(normalized)
    prediction = predictor.predict_url_features(features)
    return URLAnalysis(url=url, p_url=prediction["probability"], category=prediction["category"],
                       features=features, explanation=explain_url(features) + [URL_WARNING])


def analyze_sms(text: str) -> AnalysisResult:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("SMS должно быть непустой строкой")
    urls = extract_urls(text)
    language = detect_language(text)
    predictor = get_predictor()
    sms = predictor.predict_sms(text)
    groups = find_social_engineering_groups(text)
    results = []
    for url in urls:
        try:
            results.append(analyze_url(url))
        except ValueError:
            # Ошибка отдельного адреса не отменяет анализ исходного SMS.
            continue
    explanation = [f"Обнаружена группа признаков социальной инженерии: {group}" for group in groups]
    if language in {"ru", "kk"}:
        name = "русского" if language == "ru" else "казахского"
        explanation.append(f"Надёжность ML-оценки SMS для {name} языка снижена: модель обучена на английском корпусе.")
    elif language == "und":
        explanation.append("Недостаточно букв для надёжного определения языка SMS.")
    if results:
        explanation.append(URL_WARNING)
    return AnalysisResult(language=language, p_sms=sms["probability"],
        p_url_max=max((result.p_url for result in results), default=None), urls=results,
        features={"sms_label": sms["label"], "social_engineering_groups": groups},
        explanation=explanation, model_versions=predictor.versions)
