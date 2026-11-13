"""Детерминированная оценка риска: формула, независимые минимумы и причины."""

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from numbers import Real
from pathlib import Path

import yaml

from ml.rules import RULES
from ml.url_features import explain_url


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "risk_config.yaml"
URL_META_PATH = ROOT / "ml/artifacts/url_model_meta.json"
MAX_EXPLANATIONS = 6
GROUP_NAMES = {
    "URGENCY": "срочность", "MONEY_TRANSFER": "деньги и переводы",
    "BANK_CARD": "карты и реквизиты", "ACCOUNT_BLOCK": "блокировка аккаунта",
    "PRIZE": "призы и выигрыши", "CODE_REQUEST": "запрос кодов и паролей",
}
RECOMMENDATIONS = {
    "LOW": "Сохраняйте осторожность. При сомнениях проверьте отправителя через независимый официальный канал.",
    "MEDIUM": "Не переходите по неожиданным ссылкам. Перепроверьте сообщение через официальный канал организации.",
    "HIGH": "Не переходите по ссылкам, не сообщайте коды и данные карт. Обратитесь в организацию через независимый официальный контакт.",
}


@dataclass
class RiskResult:
    risk_score: int
    risk_level: str
    category: str
    explanation: list[str]
    recommendation: str
    disclaimer: str = "Оценка носит вероятностный характер и не гарантирует безопасность или отсутствие мошенничества."

    def to_dict(self):
        return asdict(self)


def _number(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name}: требуется конечное число")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name}: значение вне допустимого диапазона")
    return value


def validate_config(config):
    try:
        if not isinstance(config, dict) or set(config) != {"weights", "levels", "overrides"}:
            raise ValueError("Ожидаются разделы weights, levels и overrides")
        weights, levels, overrides = config["weights"], config["levels"], config["overrides"]
        if set(weights) != {"sms", "url", "social_engineering"}:
            raise ValueError("Неверные имена весов")
        for name, value in weights.items():
            _number(value, name)
        if not math.isfinite(sum(weights.values())) or weights["sms"] + weights["social_engineering"] <= 0:
            raise ValueError("Сумма весов SMS и правил должна быть положительной даже без URL")
        if set(levels) != {"medium", "high"} or any(type(v) is not int for v in levels.values()):
            raise ValueError("Границы уровней должны быть целыми числами")
        if not 1 <= levels["medium"] < levels["high"] <= 100:
            raise ValueError("Требуются границы 1 <= medium < high <= 100")
        if set(overrides) != {"code_request_min", "combined_sms_threshold", "combined_min", "url_mask_min"}:
            raise ValueError("Неверные параметры override")
        for name in ("code_request_min", "combined_min", "url_mask_min"):
            _number(overrides[name], name, maximum=100)
            if type(overrides[name]) is not int:
                raise ValueError("Минимумы override должны быть целыми")
        _number(overrides["combined_sms_threshold"], "combined_sms_threshold", maximum=1)
    except (KeyError, TypeError, AttributeError):
        raise ValueError("Некорректная структура конфигурации риска") from None
    return config


def load_config(path=None):
    """Перечитывается при вызове: изменение YAML действует без перезапуска моделей."""
    try:
        config = yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeError):
        raise ValueError("Не удалось прочитать корректный risk_config.yaml") from None
    return validate_config(config)


def load_url_t2():
    try:
        value = json.loads(URL_META_PATH.read_text(encoding="utf-8"))["t2"]
        return _number(value, "URL t2", maximum=1)
    except (OSError, KeyError, TypeError, UnicodeError, ValueError):
        raise ValueError("Не удалось прочитать корректный t2 из метаданных URL-модели") from None


def _url_flags(flags):
    if flags is None:
        return []
    try:
        if isinstance(flags, (str, bytes)):
            raise TypeError
        rows = [flags] if isinstance(flags, dict) else list(flags)
    except TypeError:
        raise ValueError("url_flags должен быть словарём или списком словарей") from None
    binary = ("is_ip", "has_punycode", "is_brand_similar", "has_mixed_scripts", "has_nonstandard_port",
              "has_at", "has_double_slash_path", "is_shortener", "has_suspicious_tld", "is_https")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("url_flags должен содержать словари признаков отдельных URL")
        for key in binary:
            if key in row and row[key] not in (0, 1):
                raise ValueError(f"Некорректный URL-флаг {key}")
        if "suspicious_keyword_count" in row:
            _number(row["suspicious_keyword_count"], "suspicious_keyword_count")
    return rows


def compute_risk(p_sms, p_url_max, se_groups, url_flags, *, config=None, t2=None) -> RiskResult:
    """Pure расчёт при переданных config/t2; без моделей и новых предсказаний.

    url_flags — список признаков по отдельным URL либо словарь одной ссылки.
    Punycode и сходство бренда должны принадлежать одной и той же ссылке.
    """
    _number(p_sms, "p_sms", maximum=1)
    if p_url_max is not None:
        _number(p_url_max, "p_url_max", maximum=1)
    if isinstance(se_groups, str):
        raise ValueError("se_groups должен быть списком имён групп")
    try:
        groups = set(se_groups or [])
        if not groups.issubset(RULES):
            raise ValueError("Неизвестная группа социальной инженерии")
    except TypeError:
        raise ValueError("Некорректный список групп социальной инженерии") from None
    groups = [name for name in RULES if name in groups]
    flags = _url_flags(url_flags)
    config = load_config() if config is None else validate_config(config)
    weights, overrides = config["weights"], config["overrides"]
    d = lambda value: Decimal(str(value))
    denominator = d(weights["sms"]) + d(weights["social_engineering"])
    if p_url_max is not None:
        denominator += d(weights["url"])
    sms_part = 100 * d(weights["sms"]) * d(p_sms) / denominator
    url_part = (100 * d(weights["url"]) * d(p_url_max) / denominator) if p_url_max is not None else Decimal(0)
    se_part = 100 * d(weights["social_engineering"]) * d(min(len(groups), 3)) / 3 / denominator
    base = sms_part + url_part + se_part
    reasons = []
    floors = [base]

    def add_override(floor, message):
        floors.append(d(floor))
        reasons.append((max(0., float(d(floor) - base)), message))

    if p_sms > .5:
        reasons.append((float(sms_part), f"SMS-модель: P_sms={p_sms:.3f}, вклад {float(sms_part):.2f} балла."))
    if p_url_max is not None and p_url_max > .5:
        reasons.append((float(url_part), f"URL-модель: P_url_max={p_url_max:.3f}, вклад {float(url_part):.2f} балла."))
    for group in groups:
        reasons.append((float(se_part) / len(groups), f"Социальная инженерия: {GROUP_NAMES[group]} ({group})."))
    if "CODE_REQUEST" in groups:
        add_override(overrides["code_request_min"],
                     f"Правило CODE_REQUEST: оценка не ниже {overrides['code_request_min']} из-за запроса кодов/паролей.")
    if p_url_max is not None and p_sms >= overrides["combined_sms_threshold"]:
        threshold = load_url_t2() if t2 is None else _number(t2, "URL t2", maximum=1)
        if p_url_max >= threshold:
            add_override(overrides["combined_min"],
                         f"Высокая SMS-оценка и URL >= t2: оценка не ниже {overrides['combined_min']}.")
    if any(row.get("is_ip", 0) for row in flags):
        add_override(overrides["url_mask_min"],
                     f"IP-адрес вместо домена URL: оценка не ниже {overrides['url_mask_min']}.")
    if any(row.get("has_punycode", 0) and row.get("is_brand_similar", 0) for row in flags):
        add_override(overrides["url_mask_min"],
                     f"Punycode-маскировка URL под бренд: оценка не ниже {overrides['url_mask_min']}.")
    for row in flags:
        for message in explain_url(row):
            reasons.append((0., message))  # Факт признака не получает выдуманного добавочного веса.
    # Вклад override — независимый разрыв от базы, а не сумма нескольких повышений.
    score = int(min(Decimal(100), max(Decimal(0), max(floors))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    level = "HIGH" if score >= config["levels"]["high"] else "MEDIUM" if score >= config["levels"]["medium"] else "LOW"
    ordered = []
    for _, message in sorted(reasons, key=lambda item: -item[0]):
        if message not in ordered:
            ordered.append(message)
    explanation = ordered[:MAX_EXPLANATIONS] or ["Выраженных признаков риска не обнаружено; это не гарантирует безопасность."]
    return RiskResult(score, level, {"LOW":"SAFE", "MEDIUM":"SUSPICIOUS", "HIGH":"FRAUDULENT"}[level],
                      explanation, RECOMMENDATIONS[level])
