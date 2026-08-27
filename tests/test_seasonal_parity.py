import numpy as np
import pandas as pd
import pytest

from mojo_statsmodels_tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose as sm_decompose


@pytest.fixture(scope="module")
def monthly():
    rng = np.random.default_rng(14)
    time = np.arange(120, dtype=np.float64)
    return 20.0 + 0.08 * time + 2.5 * np.sin(2 * np.pi * time / 12) + rng.normal(
        scale=0.2, size=time.size
    )


@pytest.mark.parametrize("period", [7, 12])
@pytest.mark.parametrize("two_sided", [False, True])
def test_additive_parity(monthly, period, two_sided):
    ours = seasonal_decompose(monthly, period=period, two_sided=two_sided)
    theirs = sm_decompose(monthly, period=period, two_sided=two_sided)
    for name in ("observed", "trend", "seasonal", "resid"):
        assert np.allclose(
            getattr(ours, name),
            getattr(theirs, name),
            rtol=2e-13,
            atol=2e-13,
            equal_nan=True,
        )


def test_multiplicative_parity():
    time = np.arange(96, dtype=np.float64)
    values = (10.0 + 0.04 * time) * (1.0 + 0.15 * np.sin(2 * np.pi * time / 12))
    ours = seasonal_decompose(values, model="multiplicative", period=12)
    theirs = sm_decompose(values, model="multiplicative", period=12)
    for name in ("trend", "seasonal", "resid"):
        assert np.allclose(
            getattr(ours, name), getattr(theirs, name), rtol=2e-13, atol=2e-13, equal_nan=True
        )


@pytest.mark.parametrize("extrapolate", [3, "freq"])
def test_extrapolated_trend_parity(monthly, extrapolate):
    ours = seasonal_decompose(monthly, period=12, extrapolate_trend=extrapolate)
    theirs = sm_decompose(monthly, period=12, extrapolate_trend=extrapolate)
    assert np.allclose(ours.trend, theirs.trend, rtol=2e-13, atol=2e-13)
    assert np.allclose(ours.resid, theirs.resid, rtol=2e-13, atol=2e-13)


def test_custom_filter_parity(monthly):
    coefficients = np.array([0.05, 0.10, 0.20, 0.30, 0.25, 0.10])
    ours = seasonal_decompose(monthly, period=12, filt=coefficients)
    theirs = sm_decompose(monthly, period=12, filt=coefficients)
    for name in ("trend", "seasonal", "resid"):
        assert np.allclose(
            getattr(ours, name), getattr(theirs, name), rtol=2e-13, atol=2e-13, equal_nan=True
        )


def test_two_dimensional_numpy_parity(monthly):
    values = np.column_stack((monthly, monthly * 0.4 + 3.0))
    ours = seasonal_decompose(values, period=12, extrapolate_trend="freq")
    theirs = sm_decompose(values, period=12, extrapolate_trend="freq")
    assert ours.trend.shape == values.shape
    assert np.allclose(ours.seasonal, theirs.seasonal, rtol=2e-13, atol=2e-13)
    assert np.allclose(ours.resid, theirs.resid, rtol=2e-13, atol=2e-13)


def test_simd_tail_parity():
    time = np.arange(53, dtype=np.float64)
    values = np.column_stack(
        (
            8.0 + 0.03 * time + np.sin(2 * np.pi * time / 7),
            4.0 - 0.02 * time + 0.5 * np.cos(2 * np.pi * time / 7),
            2.0 + 0.01 * time,
        )
    )
    ours = seasonal_decompose(values, period=7)
    theirs = sm_decompose(values, period=7)
    for name in ("trend", "seasonal", "resid"):
        assert np.allclose(
            getattr(ours, name),
            getattr(theirs, name),
            rtol=2e-13,
            atol=2e-13,
            equal_nan=True,
        )


@pytest.mark.parametrize("size", [131_083, 131_084])
def test_large_input_parity(size):
    time = np.arange(size, dtype=np.float64)
    values = (
        30.0
        + time * 1e-5
        + 2.0 * np.sin(2 * np.pi * time / 12)
        + 0.2 * np.cos(2 * np.pi * time / 5)
    )
    ours = seasonal_decompose(values, period=12)
    theirs = sm_decompose(values, period=12)
    for name in ("trend", "seasonal", "resid"):
        assert np.allclose(
            getattr(ours, name),
            getattr(theirs, name),
            rtol=2e-13,
            atol=2e-13,
            equal_nan=True,
        )


def test_pandas_frequency_and_metadata(monthly):
    index = pd.date_range("2015-01-01", periods=monthly.size, freq="MS")
    series = pd.Series(monthly, index=index, name="sales")
    ours = seasonal_decompose(series)
    theirs = sm_decompose(series)
    assert isinstance(ours.trend, pd.Series)
    assert ours.trend.index.equals(index)
    assert ours.observed.name == "sales"
    assert ours.trend.name == "trend"
    assert isinstance(ours.weights, pd.Series)
    assert ours.weights.index.equals(index)
    assert ours.weights.name == "weights"
    assert np.array_equal(ours.weights, np.ones(monthly.size))
    assert np.allclose(ours.trend, theirs.trend, equal_nan=True)


def test_decomposition_validation(monthly):
    with pytest.raises(ValueError, match="period"):
        seasonal_decompose(monthly)
    with pytest.raises(ValueError, match="2 complete cycles"):
        seasonal_decompose(monthly[:20], period=12)
    with pytest.raises(ValueError, match="zero and negative"):
        seasonal_decompose(monthly - 100, model="multiplicative", period=12)
    broken = monthly.copy()
    broken[3] = np.nan
    with pytest.raises(ValueError, match="missing"):
        seasonal_decompose(broken, period=12)
    with pytest.raises(ValueError, match="non-empty"):
        seasonal_decompose(monthly, period=12, filt=[])
