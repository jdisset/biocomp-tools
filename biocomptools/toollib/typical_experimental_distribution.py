"""
Typical experimental distribution sampling for design predictions.

KDE fitted from MatrixPgu experiment (2023-11-26) with independent input variables.
Only supports 1D and 2D sampling (no 3D - would need different experiment).
"""

from __future__ import annotations

import importlib.resources
from functools import lru_cache

import numpy as np
from scipy.stats import gaussian_kde


def _logb(x, base=10):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(x) / np.log(base)


def _cubic_exp_fwd(x, threshold, base, scale: float = 1):
    logthresh, logbase = np.log(threshold), np.log(base)
    a = -0.5 * (3 - 2 * scale * logthresh) / (threshold**3 * logbase)
    b = -(-4 + 3 * scale * logthresh) / (threshold**2 * logbase)
    c = -0.5 * (5 - 6 * scale * logthresh) / (threshold * logbase)
    return a * x**3 + b * x**2 + c * x


def _cubic_exp_inv(y, threshold, base, scale: float):
    lT, lB, cb2 = np.log(threshold), np.log(base), np.cbrt(2)
    T, T2, T3 = threshold, threshold**2, threshold**3
    A = T3 * (
        56
        + y * lB * (486 - 648 * scale * lT + 216 * scale**2 * lT**2)
        - 522 * scale * lT
        + 648 * scale**2 * lT**2
        - 216 * scale**3 * lT**3
    )
    B = np.sqrt(4 * (-19 * T2 + 12 * scale * T2 * lT) ** 3 + A**2)
    C = np.cbrt(A + B)
    D = -9 + 6 * scale * lT
    E = 2 * T * (-4 + 3 * scale * lT) / D
    F = cb2 * (-19 * T2 + 12 * scale * T2 * lT)
    return E - (F / (D * C)) + (C / (cb2 * D))


def _log_poly_log(x, threshold=300, base=10, compression=0.4):
    x = np.asarray(x)
    sign = np.sign(x)
    x = np.abs(x)
    diff = _logb(threshold, base) * (1.0 - compression)
    return (
        np.where(
            x > threshold,
            _logb(x, base) - diff,
            _cubic_exp_fwd(x, threshold, base=base, scale=compression),
        )
        * sign
    )


def _inverse_log_poly_log(y, threshold=300, base=10, compression=0.4):
    y = np.asarray(y)
    sign = np.sign(y)
    y = np.abs(y)
    diff = _logb(threshold, base) * (1.0 - compression)
    transformed_threshold = _cubic_exp_fwd(threshold, threshold, base=base, scale=compression)
    return (
        np.where(
            y > transformed_threshold,
            base ** (y + diff),
            _cubic_exp_inv(y, threshold, base=base, scale=compression),
        )
        * sign
    )


class _FrozenRescaler:
    INPUT_MIN = 500
    INPUT_MAX = 1e8
    LOW_END_COMPRESSION = 100
    POLY_THRESHOLD = 300
    POLY_COEF = 0.4

    def __init__(self):
        self._log_start = _log_poly_log(
            self.INPUT_MIN / self.LOW_END_COMPRESSION, self.POLY_THRESHOLD, 10, self.POLY_COEF
        )
        self._log_end = _log_poly_log(
            self.INPUT_MAX / self.LOW_END_COMPRESSION, self.POLY_THRESHOLD, 10, self.POLY_COEF
        )

    def fwd(self, x):
        xp = (
            _log_poly_log(
                1 + np.asarray(x) / self.LOW_END_COMPRESSION,
                self.POLY_THRESHOLD,
                10,
                self.POLY_COEF,
            )
            - self._log_start
        )
        return xp / (self._log_end - self._log_start)

    def inv(self, y):
        yp = np.asarray(y) * (self._log_end - self._log_start) + self._log_start
        return self.LOW_END_COMPRESSION * (
            _inverse_log_poly_log(yp, self.POLY_THRESHOLD, 10, self.POLY_COEF) - 1
        )


RESCALER = _FrozenRescaler()


@lru_cache(maxsize=1)
def _load_data():
    with (
        importlib.resources.files('biocomptools.toollib')
        .joinpath('typical_experimental_data.npz')
        .open('rb') as f
    ):
        data = np.load(f)
        return data['data_1d'].copy(), data['data_2d'].copy()


@lru_cache(maxsize=2)
def _get_kde(dim: int) -> gaussian_kde:
    data_1d, data_2d = _load_data()
    if dim == 1:
        return gaussian_kde(data_1d.T)
    elif dim == 2:
        return gaussian_kde(data_2d.T)
    raise ValueError(f"dim must be 1 or 2, got {dim}")


def sample_latent(n_samples: int, dim: int, seed: int | None = None) -> np.ndarray:
    """Sample from typical experimental distribution in latent space [0, 1]."""
    assert dim in (1, 2), f"dim must be 1 or 2, got {dim}"
    rng = np.random.default_rng(seed)
    kde = _get_kde(dim)
    samples = kde.resample(n_samples, seed=rng).T
    return np.clip(samples, 0, 1)


def sample_raw(n_samples: int, dim: int, seed: int | None = None) -> np.ndarray:
    """Sample from typical experimental distribution in raw space."""
    return RESCALER.inv(sample_latent(n_samples, dim, seed))
