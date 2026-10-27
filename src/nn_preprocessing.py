"""Reusable median-imputation and standard-scaling for MatchHubML NN inputs.

The fitted numeric parameters can be serialized to JSON and reconstructed for
inference without pickling scikit-learn objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class MedianStandardPreprocessor:
    """Apply the project's median imputation followed by StandardScaler."""

    feature_columns: list[str]
    imputer_statistics: np.ndarray | None = None
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None
    scaler_var: np.ndarray | None = None
    scaler_n_samples_seen: int | float | list[int] | None = None

    def fit(self, frame: pd.DataFrame) -> "MedianStandardPreprocessor":
        self._check_columns(frame)
        values = frame[self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        imputed = imputer.fit_transform(values)
        scaler = StandardScaler()
        scaler.fit(imputed)

        self.imputer_statistics = np.asarray(imputer.statistics_, dtype=np.float64)
        self.scaler_mean = np.asarray(scaler.mean_, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler.scale_, dtype=np.float64)
        self.scaler_var = np.asarray(scaler.var_, dtype=np.float64)
        n_seen = scaler.n_samples_seen_
        if isinstance(n_seen, np.ndarray):
            self.scaler_n_samples_seen = n_seen.astype(int).tolist()
        else:
            self.scaler_n_samples_seen = int(n_seen)
        self._check_fitted()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_columns(frame)
        self._check_fitted()
        values = frame[self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError("Modelinput indeholder uendelige værdier.")
        missing = np.isnan(values)
        if missing.any():
            values = values.copy()
            values[missing] = np.broadcast_to(
                self.imputer_statistics, values.shape
            )[missing]
        return (values - self.scaler_mean) / self.scaler_scale

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "schema_version": 1,
            "feature_columns": list(self.feature_columns),
            "imputer": {
                "strategy": "median",
                "keep_empty_features": True,
                "statistics": self.imputer_statistics.tolist(),
            },
            "scaler": {
                "type": "StandardScaler",
                "with_mean": True,
                "with_std": True,
                "mean": self.scaler_mean.tolist(),
                "scale": self.scaler_scale.tolist(),
                "variance": self.scaler_var.tolist(),
                "n_samples_seen": self.scaler_n_samples_seen,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MedianStandardPreprocessor":
        if data.get("schema_version") != 1:
            raise ValueError("Ukendt preprocessing-schema.")
        imputer = data["imputer"]
        scaler = data["scaler"]
        obj = cls(
            feature_columns=list(data["feature_columns"]),
            imputer_statistics=np.asarray(imputer["statistics"], dtype=np.float64),
            scaler_mean=np.asarray(scaler["mean"], dtype=np.float64),
            scaler_scale=np.asarray(scaler["scale"], dtype=np.float64),
            scaler_var=np.asarray(scaler["variance"], dtype=np.float64),
            scaler_n_samples_seen=scaler["n_samples_seen"],
        )
        obj._check_fitted()
        return obj

    def _check_columns(self, frame: pd.DataFrame) -> None:
        if list(self.feature_columns) != list(dict.fromkeys(self.feature_columns)):
            raise ValueError("Featurelisten indeholder dubletter.")
        missing = [name for name in self.feature_columns if name not in frame]
        if missing:
            raise ValueError(f"Input mangler features: {missing}")

    def _check_fitted(self) -> None:
        arrays = (
            self.imputer_statistics,
            self.scaler_mean,
            self.scaler_scale,
            self.scaler_var,
        )
        if any(value is None for value in arrays):
            raise RuntimeError("Preprocessoren er ikke tilpasset.")
        if any(value.shape != (len(self.feature_columns),) for value in arrays):
            raise ValueError("Preprocessingparametre matcher ikke featurelisten.")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("Preprocessingparametre indeholder ugyldige værdier.")
        if (self.scaler_scale <= 0).any():
            raise ValueError("Scalerens skala skal være positiv.")
