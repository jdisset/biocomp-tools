"""Overlay of "ideal" activation functions (ReLU, GELU, SELU) on the
ERN-diff latent axes, for visual comparison against the learned ERN
response shown by `ern_diff_density.histogram`.

All three curves are evaluated in latent space directly — the same
convention used by `ern_diff_density` for its dashed ReLU reference —
so they share the same latent xlims/ylims and MEF tick labeling.
"""

from typing import Any, Sequence

import numpy as np
from matplotlib.axes import Axes

from biocomp.datautils import DataRescaler
from biocomptools.toollib.figuremakers.latent_projection_density import apply_symmetric_log_axis


SELU_LAMBDA = 1.0507009873554805
SELU_ALPHA = 1.6732632423543772


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _selu(x: np.ndarray) -> np.ndarray:
    return SELU_LAMBDA * np.where(x > 0.0, x, SELU_ALPHA * (np.exp(x) - 1.0))


_ACTIVATIONS = {
    "relu": (_relu, "ReLU"),
    "gelu": (_gelu, "GELU"),
    "selu": (_selu, "SELU"),
}


def plot_activations(
    ax: Axes,
    xlims: Sequence[float] = (-0.7, 0.7),
    ylims: Sequence[float] = (-0.05, 0.7),
    label_rescaler: DataRescaler | None = None,
    activations: Sequence[str] = ("relu", "gelu", "selu"),
    n_points: int = 400,
    xtitle: str = r"$X_2 - X_1$ (fluorescence diff)",
    ytitle: str = "output fluorescence (MEF)",
    line_kwargs: dict[str, Any] | None = None,
    **_runner_kwargs: Any,
):
    assert set(activations).issubset(_ACTIVATIONS), (
        f"unknown activations: {set(activations) - set(_ACTIVATIONS)}"
    )
    xlims = tuple(float(v) for v in xlims)
    ylims = tuple(float(v) for v in ylims)

    xs = np.linspace(xlims[0], xlims[1], n_points)
    base = dict(lw=1.6, alpha=0.9)
    if line_kwargs:
        base.update(line_kwargs)

    palette = {
        "relu": "#1f77b4",
        "gelu": "#d62728",
        "selu": "#2ca02c",
    }

    for key in activations:
        fn, label = _ACTIVATIONS[key]
        ys = fn(xs)
        ax.plot(xs, ys, color=palette[key], label=label, **base)

    ax.axhline(0.0, color="#888888", lw=0.5, alpha=0.6)
    ax.axvline(0.0, color="#888888", lw=0.5, alpha=0.6)

    if label_rescaler is not None:
        apply_symmetric_log_axis(ax, "x", xlims, label_rescaler, symmetric=True)
        apply_symmetric_log_axis(ax, "y", ylims, label_rescaler, symmetric=False)
    else:
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
