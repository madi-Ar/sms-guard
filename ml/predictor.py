"""Общий ленивый кеш итоговых моделей №8 и №11; без обучения и сетевых вызовов."""

from functools import lru_cache
import json
import logging
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

from ml.url_dataset import FEATURE_NAMES
from ml.url_model import category


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Все обычные экземпляры разделяют один bundle за жизнь процесса.

    Для тестов передайте все четыре зависимости; joblib при этом не вызывается.
    Ошибка ленивой загрузки кешируется: после исправления файлов нужен перезапуск.
    """

    _shared_bundle = None
    _shared_error = None
    _load_lock = Lock()

    def __init__(self, *, sms_model=None, url_model=None, sms_meta=None, url_meta=None):
        supplied = (sms_model, url_model, sms_meta, url_meta)
        if any(value is not None for value in supplied):
            if any(value is None for value in supplied):
                raise ValueError("Для подмены передайте обе модели и оба словаря метаданных")
            self._bundle = self._validate_bundle(supplied)
        else:
            self._bundle = None

    @staticmethod
    def _validate_bundle(bundle):
        sms, url, sms_meta, url_meta = bundle
        threshold = sms_meta["threshold"]
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Некорректный порог SMS")
        category(0., url_meta["t1"], url_meta["t2"])
        names = url_meta["feature_names"]
        if not names or len(names) != len(set(names)) or not set(names).issubset(FEATURE_NAMES):
            raise ValueError("Некорректный список URL-признаков")
        if not isinstance(url_meta["dataset"]["strip_www"], bool):
            raise ValueError("Не задано правило нормализации URL")
        for model, meta in ((sms, sms_meta), (url, url_meta)):
            if list(model.classes_).count(1) != 1 or not callable(model.predict_proba):
                raise ValueError("В модели отсутствует класс 1 или predict_proba")
            if not isinstance(meta["version"], str) or not meta["version"]:
                raise ValueError("Не задана версия модели")
        if hasattr(url, "feature_names_in_") and list(url.feature_names_in_) != names:
            raise ValueError("Порядок URL-признаков модели не совпадает с JSON")
        return bundle

    def _ensure_loaded(self):
        if self._bundle is not None:
            return self._bundle
        cls = type(self)
        with cls._load_lock:
            if cls._shared_error is not None:
                raise RuntimeError(cls._shared_error)
            if cls._shared_bundle is None:
                try:
                    sms_meta = json.loads((ARTIFACTS / "sms_model_meta.json").read_text(encoding="utf-8"))
                    url_meta = json.loads((ARTIFACTS / "url_model_meta.json").read_text(encoding="utf-8"))
                    sms = joblib.load(ARTIFACTS / "sms_model.joblib")
                    url = joblib.load(ARTIFACTS / "url_model.joblib")
                    cls._shared_bundle = self._validate_bundle((sms, url, sms_meta, url_meta))
                except Exception:
                    cls._shared_error = ("Не удалось загрузить итоговые SMS/URL-модели: "
                        "проверьте sms_model.joblib, url_model.joblib и соответствующие JSON в ml/artifacts")
                    logger.error("Не удалось загрузить итоговые модели SMS Guard")
                    raise RuntimeError(cls._shared_error) from None
                logger.info("Итоговые модели SMS Guard загружены: SMS=%s, URL=%s",
                            sms_meta["version"], url_meta["version"])
            return cls._shared_bundle

    @property
    def sms_metadata(self):
        return self._ensure_loaded()[2]

    @property
    def url_metadata(self):
        return self._ensure_loaded()[3]

    @property
    def versions(self):
        return {"sms": self.sms_metadata["version"], "url": self.url_metadata["version"]}

    @staticmethod
    def _probability(model, values):
        try:
            matrix = np.asarray(model.predict_proba(values))
            probability = float(matrix[0, list(model.classes_).index(1)])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("invalid probability")
        except Exception:
            raise RuntimeError("Модель не смогла вычислить корректную вероятность класса 1") from None
        return probability

    def predict_sms(self, text: str) -> dict:
        sms, _, meta, _ = self._ensure_loaded()
        # Исходный текст идёт прямо в сохранённый Pipeline: нормализация уже внутри.
        probability = self._probability(sms, [text])
        return {"probability": probability, "label": int(probability >= meta["threshold"])}

    def predict_url_features(self, features: dict) -> dict:
        _, url, _, meta = self._ensure_loaded()
        try:
            matrix = pd.DataFrame([[features[name] for name in meta["feature_names"]]],
                                  columns=meta["feature_names"])
            if not all(pd.api.types.is_numeric_dtype(t) for t in matrix.dtypes) or not np.isfinite(matrix.to_numpy()).all():
                raise ValueError("non-numeric features")
        except (KeyError, TypeError, ValueError):
            raise ValueError("Требуется числовой набор URL-признаков в соответствии с метаданными") from None
        probability = self._probability(url, matrix)
        return {"probability": probability, "category": category(probability, meta["t1"], meta["t2"])}


@lru_cache(maxsize=1)
def get_predictor() -> ModelPredictor:
    """Единый сервис; bundle защищён lock даже при одновременном первом обращении."""
    return ModelPredictor()
