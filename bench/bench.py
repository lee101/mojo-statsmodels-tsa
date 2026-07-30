"""Benchmarks against statsmodels.tsa on identical float64 arrays."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_statsmodels_tsa import seasonal_decompose  # noqa: E402
from mojo_statsmodels_tsa import stattools as mts  # noqa: E402
from statsmodels.tsa import stattools as sm  # noqa: E402
from statsmodels.tsa.seasonal import seasonal_decompose as sm_decompose  # noqa: E402


def time_best(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.1f} ms"


def quiet(function):
    def wrapped():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function()

    return wrapped


def cases():
    rng = np.random.default_rng(481)

    values = np.ascontiguousarray(rng.normal(size=2_000_000))
    yield (
        "acovf, 64 lags (2M)",
        lambda: mts.acovf(values, fft=False, nlag=64),
        lambda: sm.acovf(values, fft=False, nlag=64),
    )

    yield (
        "acf, 40 lags (2M)",
        lambda: mts.acf(values, nlags=40),
        lambda: sm.acf(values, nlags=40),
    )

    other = np.ascontiguousarray(0.3 * values + rng.normal(size=values.size))
    yield (
        "ccf, 40 lags (2M)",
        lambda: mts.ccf(values, other, nlags=40),
        lambda: sm.ccf(values, other, nlags=40),
    )

    time_axis = np.arange(values.size, dtype=np.float64)
    seasonal_values = np.ascontiguousarray(
        50.0
        + time_axis * 1e-5
        + 4.0 * np.sin(2.0 * np.pi * time_axis / 24.0)
        + 0.2 * values
    )
    yield (
        "seasonal_decompose, p=24 (2M)",
        lambda: seasonal_decompose(seasonal_values, period=24),
        lambda: sm_decompose(seasonal_values, period=24),
    )

    yield (
        "KPSS, 32 lags (2M)",
        quiet(lambda: mts.kpss(values, nlags=32)),
        quiet(lambda: sm.kpss(values, nlags=32)),
    )

    adf_values = values[:500_000]
    yield (
        "ADF, fixed 12 lags (500k)",
        lambda: mts.adfuller(adf_values, maxlag=12, autolag=None),
        lambda: sm.adfuller(adf_values, maxlag=12, autolag=None),
    )


def main():
    print(f"Machine: {cpu_name()}, {platform.system()} {platform.machine()}")
    print()
    print("| case | Mojo port | statsmodels | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream in cases():
        ours()
        upstream()
        mojo_time = time_best(ours)
        upstream_time = time_best(upstream)
        ratio = upstream_time / mojo_time
        result = f"{ratio:.2f}x faster" if ratio >= 1.0 else f"{1.0 / ratio:.2f}x slower"
        print(
            f"| {name} | {format_time(mojo_time)} | "
            f"{format_time(upstream_time)} | {result} |"
        )


if __name__ == "__main__":
    main()
