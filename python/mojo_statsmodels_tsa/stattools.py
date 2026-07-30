"""Correlation functions and stationarity tests compatible with statsmodels.tsa."""

from __future__ import annotations

import math
from types import SimpleNamespace
import warnings

import numpy as np
from scipy.linalg import toeplitz
from scipy.fft import next_fast_len
from scipy.stats import chi2, norm

from ._adf import mackinnoncrit, mackinnonp
from ._lib import addr, f64, lib


class InterpolationWarning(UserWarning):
    pass


def _vector(x, name="x") -> np.ndarray:
    values = f64(x)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one observation")
    return values


def acovf(x, adjusted=False, demean=True, fft=True, missing="none", nlag=None):
    values = _vector(x)
    missing = str(missing).lower()
    if missing not in {"none", "raise", "conservative", "drop"}:
        raise ValueError("missing must be one of 'none', 'raise', 'conservative', or 'drop'")
    mask = np.isnan(values)
    if missing == "raise" and mask.any():
        raise ValueError("NaNs were encountered in the data")
    if missing == "drop" and mask.any():
        values = np.ascontiguousarray(values[~mask])
        mask = np.zeros(values.size, dtype=bool)
    nobs = values.size
    lag_count = nobs - 1 if nlag is None else int(nlag)
    if lag_count < 0 or lag_count > nobs - 1:
        raise ValueError("nlag must be smaller than nobs - 1")
    if missing == "conservative" and mask.any():
        valid = ~mask
        valid_int = valid.astype(np.int64)
        centered = values.copy()
        mean = np.nanmean(centered) if demean else 0.0
        centered[valid] -= mean
        centered[~valid] = 0.0
        result = np.empty(lag_count + 1)
        total_valid = int(valid.sum())
        for lag in range(lag_count + 1):
            result[lag] = np.dot(centered[lag:], centered[: nobs - lag])
            divisor = (
                np.dot(valid_int[lag:], valid_int[: nobs - lag])
                if adjusted
                else total_valid
            )
            result[lag] /= max(int(divisor), 1)
        return result
    if fft and nlag is None and nobs > 2048:
        centered = values - values.mean() if demean else values
        fft_size = next_fast_len(2 * nobs + 1)
        spectrum = np.fft.fft(centered, n=fft_size)
        result = np.fft.ifft(spectrum * np.conjugate(spectrum))[:nobs].real
        divisor = nobs - np.arange(nobs) if adjusted else nobs
        return result / divisor
    result = np.empty(lag_count + 1, dtype=np.float64)
    lib().mts_acovf(
        addr(values),
        nobs,
        lag_count,
        int(bool(demean)),
        int(bool(adjusted)),
        addr(result, writable=True),
    )
    return result


def q_stat(x, nobs):
    correlations = _vector(x)
    nobs = int(nobs)
    if nobs <= correlations.size:
        raise ValueError("nobs must exceed the number of autocorrelations")
    statistics = np.empty_like(correlations)
    if correlations.size:
        lib().mts_q_stat(
            addr(correlations), correlations.size, nobs, addr(statistics, writable=True)
        )
    return statistics, chi2.sf(statistics, np.arange(1, correlations.size + 1))


def acf(
    x,
    adjusted=False,
    nlags=None,
    qstat=False,
    fft=True,
    alpha=None,
    bartlett_confint=True,
    missing="none",
):
    values = _vector(x)
    nobs = values.size
    nlags = min(int(10 * np.log10(nobs)), nobs - 1) if nlags is None else int(nlags)
    covariance = acovf(
        values, adjusted=adjusted, demean=True, fft=fft, missing=missing, nlag=nlags
    )
    correlation = covariance / covariance[0]
    if not (qstat or alpha is not None):
        return correlation
    interval_alpha = 0.05 if alpha is None else float(alpha)
    if bartlett_confint:
        variance = np.ones_like(correlation) / nobs
        variance[0] = 0.0
        if correlation.size > 1:
            variance[1] = 1.0 / nobs
        if correlation.size > 2:
            variance[2:] *= 1.0 + 2.0 * np.cumsum(correlation[1:-1] ** 2)
    else:
        variance = np.full_like(correlation, 1.0 / nobs)
    width = norm.ppf(1.0 - interval_alpha / 2.0) * np.sqrt(variance)
    confidence = np.column_stack((correlation - width, correlation + width))
    if not qstat:
        return correlation, confidence
    statistics, pvalues = q_stat(correlation[1:], nobs=nobs)
    if alpha is not None:
        return correlation, confidence, statistics, pvalues
    return correlation, statistics, pvalues


def ccovf(x, y, adjusted=True, demean=True, fft=True):
    x_values, y_values = _vector(x), _vector(y, "y")
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length")
    result = np.empty(x_values.size, dtype=np.float64)
    lib().mts_ccovf(
        addr(x_values),
        addr(y_values),
        x_values.size,
        x_values.size,
        int(bool(demean)),
        int(bool(adjusted)),
        addr(result, writable=True),
    )
    return result


def ccf(x, y, adjusted=True, fft=True, *, nlags=None, alpha=None):
    x_values, y_values = _vector(x), _vector(y, "y")
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length")
    count = x_values.size if nlags is None else int(nlags)
    if count < 0 or count > x_values.size:
        raise ValueError("nlags must be between 0 and len(x)")
    result = np.empty(count, dtype=np.float64)
    if count:
        lib().mts_ccovf(
            addr(x_values),
            addr(y_values),
            x_values.size,
            count,
            1,
            int(bool(adjusted)),
            addr(result, writable=True),
        )
    result /= np.std(x_values) * np.std(y_values)
    if alpha is None:
        return result
    width = norm.ppf(1.0 - float(alpha) / 2.0) / np.sqrt(x_values.size)
    return result, result[:, None] + width * np.array([-1.0, 1.0])


def pacf_yw(x, nlags=None, method="adjusted"):
    values = _vector(x)
    method = str(method).lower()
    if method not in {"adjusted", "mle"}:
        raise ValueError("method must be 'adjusted' or 'mle'")
    if nlags is None:
        nlags = max(min(int(10 * np.log10(values.size)), values.size - 1), 1)
    nlags = int(nlags)
    covariance = acovf(
        values, adjusted=method == "adjusted", demean=True, fft=False, nlag=nlags
    )
    result = np.ones(nlags + 1)
    for lag in range(1, nlags + 1):
        matrix = toeplitz(covariance[:lag])
        try:
            coefficients = np.linalg.solve(matrix, covariance[1 : lag + 1])
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(matrix) @ covariance[1 : lag + 1]
        result[lag] = coefficients[-1]
    return result


class _OLSResult:
    def __init__(self, y: np.ndarray, x: np.ndarray):
        nobs, columns = x.shape
        gram = np.empty((columns, columns), dtype=np.float64)
        rhs = np.empty(columns, dtype=np.float64)
        lib().mts_ols_moments(
            addr(x),
            addr(y),
            nobs,
            columns,
            addr(gram, writable=True),
            addr(rhs, writable=True),
        )
        try:
            params = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            params = np.linalg.lstsq(x, y, rcond=None)[0]
        self.params = params
        self.resid = y - x @ params
        self.ssr = float(self.resid @ self.resid)
        rank = int(np.linalg.matrix_rank(x))
        self.df_resid = nobs - rank
        scale = self.ssr / self.df_resid
        covariance = np.linalg.pinv(gram) * scale
        self.bse = np.sqrt(np.diag(covariance))
        self.tvalues = params / self.bse
        log_likelihood = -0.5 * nobs * (
            math.log(2.0 * math.pi) + math.log(self.ssr / nobs) + 1.0
        )
        self.aic = -2.0 * log_likelihood + 2.0 * rank
        self.bic = -2.0 * log_likelihood + math.log(nobs) * rank


def _trend(nobs: int, regression: str) -> np.ndarray:
    time = np.arange(1.0, nobs + 1.0)
    if regression == "c":
        return np.ones((nobs, 1))
    if regression == "ct":
        return np.column_stack((np.ones(nobs), time))
    if regression == "ctt":
        return np.column_stack((np.ones(nobs), time, time * time))
    return np.empty((nobs, 0))


def _adf_base(values: np.ndarray, lag: int):
    difference = np.diff(values)
    y = np.ascontiguousarray(difference[lag:])
    columns = [values[lag:-1]]
    columns.extend(difference[lag - i : -i] for i in range(1, lag + 1))
    return y, np.ascontiguousarray(np.column_stack(columns))


def adfuller(
    x,
    maxlag=None,
    regression="c",
    autolag="AIC",
    store=False,
    regresults=False,
):
    values = _vector(x)
    if not np.isfinite(values).all():
        raise ValueError("x contains NaN or infinity")
    if values.max() == values.min():
        raise ValueError("Invalid input, x is constant")
    mapping = {None: "n", 0: "c", 1: "ct", 2: "ctt"}
    regression = mapping.get(regression, str(regression).lower())
    if regression not in {"c", "ct", "ctt", "n"}:
        raise ValueError("regression must be one of 'c', 'ct', 'ctt', or 'n'")
    if autolag is not None:
        autolag = str(autolag).lower()
        if autolag not in {"aic", "bic", "t-stat"}:
            raise ValueError("autolag must be 'AIC', 'BIC', 't-stat', or None")
    ntrend = 0 if regression == "n" else len(regression)
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * (values.size / 100.0) ** 0.25))
        maxlag = min(values.size // 2 - ntrend - 1, maxlag)
    maxlag = int(maxlag)
    if maxlag < 0 or maxlag > values.size // 2 - ntrend - 1:
        raise ValueError("maxlag must be less than (nobs/2 - 1 - ntrend)")
    icbest = None
    autolag_results = {}
    usedlag = maxlag
    if autolag is not None:
        y_fixed, base_fixed = _adf_base(values, maxlag)
        deterministic = _trend(y_fixed.size, regression)
        for lag in range(maxlag + 1):
            design = np.ascontiguousarray(
                np.column_stack((deterministic, base_fixed[:, : lag + 1]))
            )
            autolag_results[lag] = _OLSResult(y_fixed, design)
        if autolag == "aic":
            usedlag = min(autolag_results, key=lambda lag: autolag_results[lag].aic)
            icbest = autolag_results[usedlag].aic
        elif autolag == "bic":
            usedlag = min(autolag_results, key=lambda lag: autolag_results[lag].bic)
            icbest = autolag_results[usedlag].bic
        else:
            usedlag = maxlag
            for lag in range(maxlag, -1, -1):
                icbest = abs(float(autolag_results[lag].tvalues[-1]))
                usedlag = lag
                if icbest >= 1.6448536269514722:
                    break
    y, base = _adf_base(values, usedlag)
    design = np.ascontiguousarray(np.column_stack((base, _trend(y.size, regression))))
    result = _OLSResult(y, design)
    statistic = float(result.tvalues[0])
    critical_array = mackinnoncrit(regression, y.size)
    critical = dict(zip(("1%", "5%", "10%"), critical_array))
    pvalue = mackinnonp(statistic, regression)
    if store or regresults:
        result_store = SimpleNamespace(
            resols=result,
            maxlag=maxlag,
            usedlag=usedlag,
            adfstat=statistic,
            critvalues=critical,
            nobs=y.size,
            icbest=icbest,
        )
        if regresults:
            result_store.autolag_results = autolag_results
        return statistic, pvalue, critical, result_store
    base_return = (statistic, pvalue, usedlag, y.size, critical)
    return base_return + ((icbest,) if autolag is not None else ())


def _kpss_autolag(residual: np.ndarray) -> int:
    nobs = residual.size
    covariance_lags = int(nobs ** (2.0 / 9.0))
    covariance = acovf(residual, adjusted=False, demean=False, fft=False, nlag=covariance_lags)
    s0 = covariance[0]
    s1 = 0.0
    for lag in range(1, covariance_lags + 1):
        product = 2.0 * covariance[lag]
        s0 += product
        s1 += lag * product
    ratio = s1 / s0
    return int(1.1447 * (ratio * ratio) ** (1.0 / 3.0) * nobs ** (1.0 / 3.0))


def kpss(x, regression="c", nlags="auto", store=False):
    values = _vector(x)
    if not np.isfinite(values).all():
        raise ValueError("x contains NaN or infinity")
    regression = str(regression).lower()
    if regression not in {"c", "ct"}:
        raise ValueError("regression must be 'c' or 'ct'")
    nobs = values.size
    if regression == "ct":
        design = np.column_stack((np.arange(1.0, nobs + 1.0), np.ones(nobs)))
        residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        critical_values = [0.119, 0.146, 0.176, 0.216]
    else:
        residual = values - values.mean()
        critical_values = [0.347, 0.463, 0.574, 0.739]
    residual = np.ascontiguousarray(residual)
    if nlags == "legacy":
        lag_count = min(int(np.ceil(12.0 * (nobs / 100.0) ** 0.25)), nobs - 1)
    elif nlags == "auto" or nlags is None:
        lag_count = min(_kpss_autolag(residual), nobs - 1)
    elif isinstance(nlags, str):
        raise ValueError("nlags must be 'auto', 'legacy', or an integer")
    else:
        lag_count = int(nlags)
        if lag_count < 0 or lag_count >= nobs:
            raise ValueError("lags must be non-negative and less than the observations")
    moments = np.empty(2, dtype=np.float64)
    lib().mts_kpss_moments(
        addr(residual), nobs, lag_count, addr(moments, writable=True)
    )
    statistic = float(moments[0] / moments[1])
    pvalues = [0.10, 0.05, 0.025, 0.01]
    pvalue = float(np.interp(statistic, critical_values, pvalues))
    if pvalue in (pvalues[0], pvalues[-1]):
        direction = "greater" if pvalue == pvalues[0] else "smaller"
        warnings.warn(
            f"The actual p-value is {direction} than the p-value returned.",
            InterpolationWarning,
            stacklevel=2,
        )
    critical = dict(zip(("10%", "5%", "2.5%", "1%"), critical_values))
    if store:
        return statistic, pvalue, critical, SimpleNamespace(lags=lag_count, nobs=nobs)
    return statistic, pvalue, lag_count, critical
