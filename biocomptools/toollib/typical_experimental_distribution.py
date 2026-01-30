"""
Typical experimental distribution sampling for design predictions.

KDE fitted from MatrixPgu experiment (2023-11-26). Empirical correlation ~0.475 between
inputs in Gaussian copula space. For dim > 2, uses Gaussian copula with equicorrelation
and pooled marginal from 2D data.
"""

from __future__ import annotations

import importlib.resources
from functools import lru_cache

import numpy as np
from scipy.stats import gaussian_kde, norm

EMPIRICAL_COPULA_RHO = 0.475


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


@lru_cache(maxsize=3)
def _get_kde(kind: str) -> gaussian_kde:
    data_1d, data_2d = _load_data()
    if kind == '1d':
        return gaussian_kde(data_1d.T)
    elif kind == '2d':
        return gaussian_kde(data_2d.T)
    elif kind == 'pooled':
        pooled = np.concatenate([data_2d[:, 0], data_2d[:, 1]])
        return gaussian_kde(pooled)
    raise ValueError(f"kind must be '1d', '2d', or 'pooled', got {kind}")


def _inverse_kde_cdf(kde: gaussian_kde, u: np.ndarray, n_grid: int = 1000) -> np.ndarray:
    """Invert KDE CDF via interpolation."""
    x_grid = np.linspace(-0.5, 1.5, n_grid)
    cdf_grid = np.array([kde.integrate_box_1d(-np.inf, xi) for xi in x_grid])
    return np.interp(u, cdf_grid, x_grid)


def sample_latent(
    n_samples: int,
    dim: int,
    seed: int | None = None,
    rho: float | None = None,
) -> np.ndarray:
    """Sample from typical experimental distribution in latent space [0, 1].

    For dim=1,2: uses empirical KDE directly.
    For dim>2: uses Gaussian copula with equicorrelation and pooled marginal.

    Args:
        n_samples: number of samples
        dim: dimensionality
        seed: random seed
        rho: pairwise correlation in Gaussian copula space. If None, uses
             EMPIRICAL_COPULA_RHO (~0.475) for dim>2, ignored for dim<=2.
             Set to 0 for independent sampling.
    """
    rng = np.random.default_rng(seed)

    if dim == 1:
        samples = _get_kde('1d').resample(n_samples, seed=rng).T
    elif dim == 2:
        samples = _get_kde('2d').resample(n_samples, seed=rng).T
    else:
        if rho is None:
            rho = EMPIRICAL_COPULA_RHO
        cov = np.full((dim, dim), rho)
        np.fill_diagonal(cov, 1.0)
        z = rng.multivariate_normal(np.zeros(dim), cov, size=n_samples)
        u = norm.cdf(z)
        marginal_kde = _get_kde('pooled')
        samples = np.column_stack([_inverse_kde_cdf(marginal_kde, u[:, d]) for d in range(dim)])

    return np.clip(samples, 0, 1)


def sample_raw(
    n_samples: int,
    dim: int,
    seed: int | None = None,
    rho: float | None = None,
) -> np.ndarray:
    """Sample from typical experimental distribution in raw space."""
    return RESCALER.inv(sample_latent(n_samples, dim, seed, rho))
