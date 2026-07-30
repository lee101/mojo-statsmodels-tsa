"""MacKinnon response surfaces used by the augmented Dickey-Fuller test."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_MIN = {"n": -19.04, "c": -18.83, "ct": -16.18, "ctt": -17.17}
_MAX = {"n": np.inf, "c": 2.74, "ct": 0.70, "ctt": 0.54}
_STAR = {"n": -1.04, "c": -1.61, "ct": -2.89, "ctt": -3.21}
_SMALL = {
    "n": [0.6344, 1.2378, 0.032496],
    "c": [2.1659, 1.4412, 0.038269],
    "ct": [3.2512, 1.6047, 0.049588],
    "ctt": [4.0003, 1.6580, 0.048288],
}
_LARGE = {
    "n": [0.4797, 0.93557, -0.06999, 0.033066],
    "c": [1.7339, 0.93202, -0.12745, -0.010368],
    "ct": [2.5261, 0.61654, -0.37956, -0.060285],
    "ctt": [3.0778, 0.49529, -0.41477, -0.059355],
}
_CRITICAL = {
    "n": [
        [-2.56574, -2.2358, -3.627, 0.0],
        [-1.94100, -0.2686, -3.365, 31.223],
        [-1.61682, 0.2656, -2.714, 25.364],
    ],
    "c": [
        [-3.43035, -6.5393, -16.786, -79.433],
        [-2.86154, -2.8903, -4.234, -40.040],
        [-2.56677, -1.5384, -2.809, 0.0],
    ],
    "ct": [
        [-3.95877, -9.0531, -28.428, -134.155],
        [-3.41049, -4.3904, -9.036, -45.374],
        [-3.12705, -2.5856, -3.925, -22.380],
    ],
    "ctt": [
        [-4.37113, -11.5882, -35.819, -334.047],
        [-3.83239, -5.9057, -12.490, -118.284],
        [-3.55326, -3.6596, -5.293, -63.559],
    ],
}


def mackinnonp(statistic: float, regression: str) -> float:
    if statistic > _MAX[regression]:
        return 1.0
    if statistic < _MIN[regression]:
        return 0.0
    coefficients = _SMALL[regression] if statistic <= _STAR[regression] else _LARGE[regression]
    return float(norm.cdf(np.polyval(coefficients[::-1], statistic)))


def mackinnoncrit(regression: str, nobs: int) -> np.ndarray:
    inverse_n = 1.0 / nobs
    return np.array(
        [np.polyval(coefficients[::-1], inverse_n) for coefficients in _CRITICAL[regression]]
    )
