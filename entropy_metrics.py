"""EntropyHub-backed MSE and pairwise bMSE calculations."""

from __future__ import annotations

import os
from pathlib import Path

import EntropyHub as EH
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def default_n_jobs() -> int:
    """Return HCP_N_JOBS when valid, otherwise use six workers."""
    value = os.environ.get("HCP_N_JOBS")
    if value:
        try:
            parsed = int(value)
            if parsed != 0:
                return parsed
        except ValueError:
            pass
    return 6


def complexity_index(values: np.ndarray) -> float:
    """Trapezoidal integral across all finite entropy values."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.trapz(finite))


def _mse_one(series: np.ndarray, m: int, r: float,
             scales: int) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    model = EH.MSobject(EnType="SampEn", m=m, r=r * np.nanstd(values))
    result, _ = EH.MSEn(values, model, Scales=scales)
    return np.asarray(result)


class MSE:
    """Univariate multiscale sample entropy for each DataFrame column."""

    def __init__(self, subject_id: str, timeseries: pd.DataFrame, output: str,
                 m: int = 2, r: float = 0.3, scales: int = 10,
                 n_jobs: int | None = None):
        self.subject_id = subject_id
        self.timeseries = timeseries
        self.output = Path(output)
        self.m = m
        self.r = r
        self.scales = scales
        self.n_jobs = default_n_jobs() if n_jobs is None else n_jobs

    def process(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        roi_names = self.timeseries.columns.to_list()
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_mse_one)(self.timeseries[name].to_numpy(), self.m, self.r,
                              self.scales)
            for name in roi_names
        )
        scale_columns = [f"a{i}" for i in range(1, self.scales + 1)]
        frame = pd.DataFrame(index=roi_names,
                             columns=scale_columns + ["CI"], dtype=float)
        for name, values in zip(roi_names, results):
            frame.loc[name, scale_columns] = values
            frame.loc[name, "CI"] = complexity_index(values)
        frame.to_csv(self.output / "mse.csv", index=True)


def _bmse_pair(values: np.ndarray, m: int, r: float,
               scales: int) -> np.ndarray:
    dimensions = np.full(values.shape[1], m, dtype=int)
    model = EH.MSobject(EnType="MvSampEn", m=dimensions, r=r, Norm=True)
    result, _ = EH.cMvMSEn(values, model, Refined=True, Scales=scales)
    return np.asarray(result)


class BMSE:
    """Bivariate multiscale sample entropy for every unordered ROI pair."""

    def __init__(self, m: int = 2, r: float = 0.6, a: int = 10,
                 n_jobs: int | None = None):
        self.m = m
        self.r = r
        self.a = a
        self.n_jobs = default_n_jobs() if n_jobs is None else n_jobs

    def process_data(self, data: pd.DataFrame, output_prefix: str) -> None:
        roi_names = data.columns.to_list()
        count = len(roi_names)
        pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_bmse_pair)(data.iloc[:, [i, j]].to_numpy(), self.m,
                                self.r, self.a)
            for i, j in pairs
        )
        matrices = [np.full((count, count), np.nan)
                    for _ in range(self.a + 1)]
        for (i, j), values in zip(pairs, results):
            for scale_index, value in enumerate(values):
                matrices[scale_index][i, j] = value
                matrices[scale_index][j, i] = value
            ci = complexity_index(values)
            matrices[-1][i, j] = ci
            matrices[-1][j, i] = ci

        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for scale_index, matrix in enumerate(matrices[:-1], start=1):
            pd.DataFrame(matrix, index=roi_names, columns=roi_names).to_csv(
                f"{prefix}_a{scale_index}.csv", index=True
            )
        pd.DataFrame(matrices[-1], index=roi_names, columns=roi_names).to_csv(
            f"{prefix}_CI.csv", index=True
        )
