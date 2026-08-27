import warnings

import numpy as np
import pytest

from mojo_statsmodels_tsa import stattools as mts
from statsmodels.tsa import stattools as sm


@pytest.fixture(scope="module")
def stationary():
    rng = np.random.default_rng(1042)
    innovations = rng.normal(size=600)
    values = np.empty_like(innovations)
    values[0] = innovations[0]
    for i in range(1, values.size):
        values[i] = 0.65 * values[i - 1] + innovations[i]
    return values


@pytest.mark.parametrize("adjusted", [False, True])
@pytest.mark.parametrize("demean", [False, True])
def test_acovf_direct_parity(stationary, adjusted, demean):
    ours = mts.acovf(
        stationary, adjusted=adjusted, demean=demean, fft=False, nlag=45
    )
    theirs = sm.acovf(
        stationary, adjusted=adjusted, demean=demean, fft=False, nlag=45
    )
    assert np.allclose(ours, theirs, rtol=2e-14, atol=2e-14)


def test_acovf_fft_full_parity():
    rng = np.random.default_rng(9)
    values = rng.normal(size=4097)
    ours = mts.acovf(values, adjusted=True, fft=True)
    theirs = sm.acovf(values, adjusted=True, fft=True)
    assert np.allclose(ours, theirs, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("missing", ["drop", "conservative"])
@pytest.mark.parametrize("adjusted", [False, True])
def test_acovf_missing_parity(stationary, missing, adjusted):
    values = stationary.copy()
    values[[3, 19, 88, 401]] = np.nan
    ours = mts.acovf(values, adjusted=adjusted, missing=missing, nlag=30)
    theirs = sm.acovf(values, adjusted=adjusted, missing=missing, nlag=30)
    assert np.allclose(ours, theirs, rtol=2e-14, atol=2e-14)


def test_acovf_missing_raise(stationary):
    values = stationary.copy()
    values[5] = np.nan
    with pytest.raises(ValueError, match="NaNs"):
        mts.acovf(values, missing="raise")


@pytest.mark.parametrize("adjusted", [False, True])
def test_acf_values_and_ljung_box(stationary, adjusted):
    ours = mts.acf(stationary, adjusted=adjusted, nlags=35, qstat=True)
    theirs = sm.acf(stationary, adjusted=adjusted, nlags=35, qstat=True)
    for left, right in zip(ours, theirs):
        assert np.allclose(left, right, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("bartlett", [False, True])
def test_acf_confidence_intervals(stationary, bartlett):
    ours = mts.acf(
        stationary, nlags=20, alpha=0.05, bartlett_confint=bartlett, qstat=True
    )
    theirs = sm.acf(
        stationary, nlags=20, alpha=0.05, bartlett_confint=bartlett, qstat=True
    )
    for left, right in zip(ours, theirs):
        assert np.allclose(left, right, rtol=2e-13, atol=2e-13)


def test_q_stat_published_formula():
    correlations = np.array([0.25, -0.10, 0.05, 0.02])
    ours = mts.q_stat(correlations, 200)
    theirs = sm.q_stat(correlations, 200)
    assert np.allclose(ours[0], theirs[0], rtol=1e-15)
    assert np.allclose(ours[1], theirs[1], rtol=1e-15)


def test_q_stat_empty_and_invalid_observations():
    with pytest.raises(ValueError, match="at least one"):
        mts.q_stat([], 1)
    with pytest.raises(ValueError, match="exceed"):
        mts.q_stat([0.1, 0.2], 2)


def test_noncontiguous_and_non_float64_inputs_are_converted(stationary):
    strided_float32 = stationary.astype(np.float32)[::2]
    ours = mts.acovf(strided_float32, fft=False, nlag=17)
    theirs = sm.acovf(strided_float32, fft=False, nlag=17)
    assert np.allclose(ours, theirs, rtol=2e-14, atol=2e-14)


def test_zero_lag_ccf_does_not_cross_empty_buffer(stationary):
    ours = mts.ccf(stationary, stationary, nlags=0)
    assert ours.shape == (0,)


@pytest.mark.parametrize("adjusted", [False, True])
def test_cross_covariance_and_correlation(stationary, adjusted):
    other = np.roll(stationary, 3) + np.linspace(-0.5, 0.5, stationary.size)
    assert np.allclose(
        mts.ccovf(stationary, other, adjusted=adjusted),
        sm.ccovf(stationary, other, adjusted=adjusted),
        rtol=2e-13,
        atol=2e-13,
    )
    ours = mts.ccf(stationary, other, adjusted=adjusted, nlags=40, alpha=0.1)
    theirs = sm.ccf(stationary, other, adjusted=adjusted, nlags=40, alpha=0.1)
    assert np.allclose(ours[0], theirs[0], rtol=2e-13, atol=2e-13)
    assert np.allclose(ours[1], theirs[1], rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("method", ["adjusted", "mle"])
def test_pacf_yw(stationary, method):
    ours = mts.pacf_yw(stationary, nlags=30, method=method)
    theirs = sm.pacf_yw(stationary, nlags=30, method=method)
    assert np.allclose(ours, theirs, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("regression", ["c", "ct", "ctt", "n"])
@pytest.mark.parametrize("autolag", ["AIC", "BIC", "t-stat", None])
def test_adfuller_parity(stationary, regression, autolag):
    ours = mts.adfuller(stationary, maxlag=8, regression=regression, autolag=autolag)
    theirs = sm.adfuller(stationary, maxlag=8, regression=regression, autolag=autolag)
    assert ours[0] == pytest.approx(theirs[0], rel=2e-9, abs=2e-9)
    assert ours[1] == pytest.approx(theirs[1], rel=2e-9, abs=1e-14)
    assert ours[2:4] == theirs[2:4]
    assert ours[4] == pytest.approx(theirs[4], rel=1e-14, abs=1e-14)
    if autolag is not None:
        assert ours[5] == pytest.approx(theirs[5], rel=2e-9, abs=2e-9)


def test_adfuller_store_shape(stationary):
    statistic, pvalue, critical, stored = mts.adfuller(
        stationary, maxlag=4, store=True, regresults=True
    )
    assert statistic == stored.adfstat
    assert 0.0 <= pvalue <= 1.0
    assert set(critical) == {"1%", "5%", "10%"}
    assert stored.resols.params.ndim == 1
    assert len(stored.autolag_results) == 5


def test_adfuller_fixed_lag_fast_path_matches_stored_result(stationary):
    fast = mts.adfuller(
        stationary[:-3], maxlag=7, regression="ctt", autolag=None
    )
    statistic, pvalue, critical, stored = mts.adfuller(
        stationary[:-3],
        maxlag=7,
        regression="ctt",
        autolag=None,
        store=True,
    )
    assert fast[0] == pytest.approx(statistic, rel=2e-9, abs=2e-9)
    assert fast[1] == pytest.approx(pvalue, rel=2e-9, abs=1e-14)
    assert fast[2:4] == (stored.usedlag, stored.nobs)
    assert fast[4] == pytest.approx(critical, rel=1e-14, abs=1e-14)
    assert stored.resols.resid.shape == (stored.nobs,)


@pytest.mark.parametrize("regression", ["c", "ct"])
@pytest.mark.parametrize("nlags", ["auto", "legacy", 7])
def test_kpss_parity(stationary, regression, nlags):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = mts.kpss(stationary, regression=regression, nlags=nlags)
        theirs = sm.kpss(stationary, regression=regression, nlags=nlags)
    assert ours[0] == pytest.approx(theirs[0], rel=2e-13, abs=2e-13)
    assert ours[1] == pytest.approx(theirs[1], rel=2e-13, abs=2e-13)
    assert ours[2:] == theirs[2:]


def test_validation_errors(stationary):
    with pytest.raises(ValueError, match="same length"):
        mts.ccf(stationary, stationary[:-1])
    with pytest.raises(ValueError, match="constant"):
        mts.adfuller(np.ones(100))
    with pytest.raises(ValueError, match="lags"):
        mts.kpss(stationary, nlags=stationary.size)
