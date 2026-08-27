"""ctypes bridge to the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-statsmodels-tsa.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mts_acovf": [I, I, I, I, I, I],
    "mts_ccovf": [I, I, I, I, I, I, I],
    "mts_q_stat": [I, I, I, I],
    "mts_convolution": [I, I, I, I, I, I, F, I],
    "mts_seasonal_mean": [I, I, I, I, F, I],
    "mts_seasonal_mean_detrended": [I, I, I, I, I, I, F, I, I],
    "mts_seasonal_resid": [I, I, I, I, I, I, I, I, I],
    "mts_kpss_moments": [I, I, I, I],
    "mts_ols_moments": [I, I, I, I, I, I],
}

_loaded: ctypes.CDLL | None = None


def build() -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not os.path.exists(LIB) or os.path.getmtime(source) > os.path.getmtime(LIB):
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
        )
    return LIB


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, argtypes in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = None
    return _loaded


def f64(values, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(values, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(values, dtype=np.float64)


def addr(array: np.ndarray, *, writable: bool = False) -> int:
    """Return an address only for buffers satisfying the Mojo ABI contract."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != np.dtype(np.float64) or not array.dtype.isnative:
        raise TypeError("FFI buffers must contain native-endian float64 values")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError("FFI output buffers must be writable")
    if array.size == 0:
        raise ValueError("empty buffers must not cross the FFI boundary")
    if not array.ctypes.data:
        raise ValueError("FFI buffers must have a non-null data pointer")
    return int(array.ctypes.data)
