# mojo-statsmodels-tsa

Classical time-series decomposition and statistical tests implemented with
[Mojo](https://www.modular.com/mojo), exposed to Python through APIs matching
the covered functions in `statsmodels.tsa`.

There is no separate `statsmodels-tsa` package on conda-forge or PyPI. The
upstream implementation lives inside the `statsmodels` distribution. This
repository does not import statsmodels at runtime. Its test suite compares every
covered operation against `statsmodels.tsa` 0.14.6 on the same inputs.

## Covered subset

| Upstream module | Covered API |
| --- | --- |
| `statsmodels.tsa.seasonal` | `seasonal_decompose`, `DecomposeResult` properties; additive and multiplicative models, custom filters, one- and two-sided trends, endpoint extrapolation, 1D/2D NumPy data, pandas Series |
| `statsmodels.tsa.stattools` | `acovf`, `acf`, `ccovf`, `ccf`, `pacf_yw`, `q_stat` |
| `statsmodels.tsa.stattools` tests | `adfuller` with all regression and autolag modes; `kpss` with automatic, legacy, or fixed lags |

Tests cover numerical results and the documented return shapes for each API in
the table, including confidence intervals, critical-value dictionaries, and
pandas Series metadata.

Not covered are STL/MSTL, `DecomposeResult.plot`, HP/BK/CF filters, AR/ARIMA and
state-space models, cointegration and causality tests, Zivot-Andrews, range
unit-root tests, and the rest of statsmodels. `seasonal_decompose` accepts 2D
NumPy arrays; pandas DataFrame decomposition is intentionally not claimed
because statsmodels 0.14.6 itself fails while wrapping those results.

This is a source distribution for Linux x86-64. It does not currently publish a
wheel or promise ABI compatibility with other Mojo releases.

## Install

```bash
git clone https://github.com/lee101/mojo-statsmodels-tsa.git
cd mojo-statsmodels-tsa
pixi install
pixi run build
pixi run test
```

`pixi` installs the pinned Mojo nightly, Python, NumPy, SciPy, and the
development-only parity dependencies. The build task writes
`dist/libmojo-statsmodels-tsa.so`.

NumPy and SciPy are the only Python runtime dependencies. Statsmodels and
pandas are present in the pixi development environment for parity testing.

## Usage

```python
import numpy as np
from mojo_statsmodels_tsa import acf, adfuller, seasonal_decompose

t = np.arange(120, dtype=np.float64)
rng = np.random.default_rng(7)
x = 10.0 + 0.03 * t + np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.05, t.size)

parts = seasonal_decompose(x, period=12, extrapolate_trend="freq")
correlations = acf(parts.resid, nlags=12)
adf_stat, pvalue, usedlag, nobs, critical, icbest = adfuller(
    parts.resid, autolag="AIC"
)

print(parts.seasonal[:3])
print(correlations[:3])
print(adf_stat, pvalue, usedlag)
```

The example runs after `pixi run build` with
`pixi run python path/to/example.py`.

## Performance

Measured through `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. Times are the best of five warm runs using identical contiguous
float64 arrays. The comparison is against statsmodels 0.14.6.

| case | Mojo port | statsmodels | result |
| --- | ---: | ---: | ---: |
| `acovf`, 64 lags (2M) | 82.8 ms | 1042.9 ms | 12.60x faster |
| `acf`, 40 lags (2M) | 45.4 ms | 810.2 ms | 17.83x faster |
| `ccf`, 40 lags (2M) | 122.5 ms | 549.5 ms | 4.48x faster |
| `seasonal_decompose`, period 24 (2M) | 48.1 ms | 115.6 ms | 2.40x faster |
| KPSS, 32 lags (2M) | 73.8 ms | 537.2 ms | 7.27x faster |
| ADF, fixed 12 lags (500k) | 573.2 ms | 832.2 ms | 1.45x faster |

These results favor the port's intended workload: long series with a bounded
number of requested lags. `acovf(x)` still uses an FFT for the full
autocovariance sequence, since a direct quadratic calculation is not useful
there. Run `pixi run bench` to measure the current machine; benchmark output is
not hard-coded by the script.

There is intentionally no GPU path. These kernels operate on host NumPy
buffers, and the benchmark covers only the CPU implementation.

## How it works

All Mojo kernels live in one compilation unit and export a small C ABI. Python
uses `ctypes` once per operation. Buffers cross that boundary as integer
addresses because exported Mojo functions cannot be parametric; each export
reconstructs an `UnsafePointer[Float64, AnyOrigin[mut=True]]`.

Inputs are C-contiguous row-major float64 arrays. Existing compatible arrays
cross without a copy, while other array-like inputs are converted once. Before
each call the bridge checks native float64 dtype, C contiguity, alignment,
non-empty/non-null storage, and output writability. Python keeps references to
every input, output, and scratch allocation for the duration of the synchronous
call, so there is no cross-language allocator or borrowed buffer lifetime.
Mojo handles SIMD lagged dot products,
native-width SIMD convolution, seasonal aggregation, Ljung-Box accumulation,
KPSS moments, and ADF regression moments. Large independent convolution ranges
are divided into 32,768-value chunks and run on at most 16 workers; smaller
ranges remain serial. Python handles validation, small dense solves, MacKinnon
response surfaces, SciPy probability distributions, and pandas metadata.
Seasonal expansion avoids an integer indexing buffer, the detrended allocation
is reused for residuals, and decomposition weights are allocated only if their
property is accessed.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The tests contain numerical and behavioral parity checks, not smoke tests.
They cover missing-value policies, adjusted estimators, confidence intervals,
ADF autolag selection, KPSS lag selection, filter alignment, trend
extrapolation, multiplicative decomposition, and pandas Series wrapping.

## License

MIT
