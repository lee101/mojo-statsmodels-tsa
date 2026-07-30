"""Classical seasonal decomposition with Mojo convolution kernels."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class DecomposeResult:
    def __init__(self, observed, seasonal, trend, resid, weights=None):
        self._observed = observed
        self._seasonal = seasonal
        self._trend = trend
        self._resid = resid
        self._weights = weights

    @property
    def observed(self):
        return self._observed

    @property
    def seasonal(self):
        return self._seasonal

    @property
    def trend(self):
        return self._trend

    @property
    def resid(self):
        return self._resid

    @property
    def weights(self):
        if self._weights is None:
            weights = np.ones_like(self._observed)
            try:
                import pandas as pd

                if isinstance(self._observed, pd.Series):
                    weights = pd.Series(
                        weights, index=self._observed.index, name="weights"
                    )
            except ImportError:
                pass
            self._weights = weights
        return self._weights


def _period_from_index(x) -> int | None:
    frequency = getattr(getattr(x, "index", None), "inferred_freq", None)
    if frequency is None:
        return None
    code = str(frequency).upper()
    if code.startswith(("A", "Y")):
        return 1
    if code.startswith("Q"):
        return 4
    if code.startswith("M"):
        return 12
    if code.startswith("W"):
        return 52
    if code.startswith(("B", "D")):
        return 5 if code.startswith("B") else 7
    if code.startswith("H"):
        return 24
    return None


def _wrap_like(original, values: np.ndarray, name: str | None):
    try:
        import pandas as pd
    except ImportError:
        return values.squeeze()
    if isinstance(original, pd.Series):
        return pd.Series(
            values.squeeze(),
            index=original.index,
            name=original.name if name is None else name,
        )
    if isinstance(original, pd.DataFrame):
        columns = original.columns if name is None else [f"{name}_{c}" for c in original.columns]
        return pd.DataFrame(values, index=original.index, columns=columns)
    return values.squeeze()


def _extrapolate_trend(trend: np.ndarray, npoints: int) -> np.ndarray:
    valid_rows = np.flatnonzero(~np.isnan(trend).any(axis=1))
    front, back = int(valid_rows[0]), int(valid_rows[-1])
    front_last = min(front + npoints, back)
    back_first = max(front, back - npoints)
    design = np.column_stack((np.arange(front, front_last), np.ones(front_last - front)))
    slope, intercept = np.linalg.lstsq(design, trend[front:front_last], rcond=-1)[0]
    trend[:front] = np.outer(np.arange(front), slope) + intercept
    design = np.column_stack((np.arange(back_first, back), np.ones(back - back_first)))
    slope, intercept = np.linalg.lstsq(design, trend[back_first:back], rcond=-1)[0]
    times = np.arange(back + 1, trend.shape[0])
    trend[back + 1 :] = np.outer(times, slope) + intercept
    return trend


def seasonal_decompose(
    x,
    model="additive",
    filt=None,
    period=None,
    two_sided=True,
    extrapolate_trend=0,
):
    original = x
    values = f64(x)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("x must be 1 or 2 dimensional")
    if not np.isfinite(values).all():
        raise ValueError("This function does not handle missing values")
    model = str(model).lower()
    if not (model.startswith("a") or model.startswith("m")):
        raise ValueError("model must be 'additive' or 'multiplicative'")
    multiplicative = model.startswith("m")
    if multiplicative and np.any(values <= 0):
        raise ValueError(
            "Multiplicative seasonality is not appropriate for zero and negative values"
        )
    if period is None:
        period = _period_from_index(original)
    if period is None:
        raise ValueError("You must specify a period or provide a pandas object with a frequency")
    period = int(period)
    nobs, columns = values.shape
    if period < 1 or nobs < 2 * period:
        raise ValueError(
            f"x must have 2 complete cycles requires {2 * period} observations. "
            f"x only has {nobs} observation(s)"
        )
    if filt is None:
        if period % 2 == 0:
            filt = np.array([0.5] + [1.0] * (period - 1) + [0.5]) / period
        else:
            filt = np.repeat(1.0 / period, period)
    coefficients = f64(filt)
    if coefficients.ndim != 1 or coefficients.size == 0 or coefficients.size > nobs:
        raise ValueError(
            "filt must be a non-empty one-dimensional filter no longer than x"
        )
    trend = np.empty_like(values)
    lib().mts_convolution(
        addr(values),
        nobs,
        columns,
        addr(coefficients),
        coefficients.size,
        int(bool(two_sided)) + 1,
        float("nan"),
        addr(trend, writable=True),
    )
    if extrapolate_trend == "freq":
        extrapolate_trend = period - 1
    if int(extrapolate_trend) > 0:
        trend = _extrapolate_trend(trend, int(extrapolate_trend) + 1)
    detrended = values / trend if multiplicative else values - trend
    period_averages = np.empty((period, columns), dtype=np.float64)
    lib().mts_seasonal_mean(
        addr(detrended),
        nobs,
        columns,
        period,
        float("nan"),
        addr(period_averages, writable=True),
    )
    if multiplicative:
        period_averages /= period_averages.mean(axis=0)
    else:
        period_averages -= period_averages.mean(axis=0)
    seasonal = np.resize(period_averages, values.shape)
    if multiplicative:
        detrended /= seasonal
    else:
        detrended -= seasonal
    resid = detrended
    return DecomposeResult(
        observed=_wrap_like(original, values, None),
        seasonal=_wrap_like(original, seasonal, "seasonal"),
        trend=_wrap_like(original, trend, "trend"),
        resid=_wrap_like(original, resid, "resid"),
    )
