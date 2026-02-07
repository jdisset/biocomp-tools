"""TunerSession: Core session management for biocomp-tuner."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np

from biocomp.design_targets import TargetUnion
from biocomp.designloss import (
    compute_grid_losses,
    GridLossResult,
    GridLossWeights,
    ratio_spread_penalty,
    soft_tucount_penalty,
)
from biocomp.network import Network
from biocomp.parameters import ParameterTree
from biocomptools.logging_config import get_logger
from biocomptools.modelmodel import BiocompModel, NetworkModel

logger = get_logger(__name__)


@dataclass
class TunerConfig:
    """Configuration for TunerSession loss weights and grid settings."""

    grid_resolution: tuple[int, int] = (32, 32)
    weights: GridLossWeights = field(default_factory=lambda: GridLossWeights(w_mse=1.0, w_simse=1.0))


@dataclass
class TunerResult:
    """Enhanced result with design-mode compatible fields."""

    Y_pred: np.ndarray
    Y_target: np.ndarray
    X_lattice: np.ndarray
    losses: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    loss_contributions: Optional[dict[str, np.ndarray]] = None

    @property
    def Y_diff(self) -> np.ndarray:
        return self.Y_pred - self.Y_target

    def to_dict(self) -> dict:
        result = {
            "Y_pred": self.Y_pred.tolist(),
            "Y_target": self.Y_target.tolist(),
            "Y_diff": self.Y_diff.tolist(),
            "X_lattice": self.X_lattice.tolist(),
            "losses": self.losses,
            "penalties": self.penalties,
            "total_loss": self.losses.get("total", 0.0),
        }
        if self.loss_contributions is not None:
            result["loss_contributions"] = {
                k: v.tolist() for k, v in self.loss_contributions.items()
            }
        return result


class TunerSession:
    def __init__(
        self,
        model: BiocompModel,
        network: Network,
        target: TargetUnion,
        grid_resolution: tuple[int, int] = (32, 32),
        weights: GridLossWeights | None = None,
        # backward-compatible kwargs (used if weights is None)
        w_sinkhorn: float = 1.0,
        w_lncc: float = 0.5,
        w_mse: float = 1.0,
        w_simse: float = 1.0,
        eps_sinkhorn: float = 0.1,
        n_sinkhorn_iters: int = 50,
        lncc_kernel: int = 7,
    ):
        self.network = network
        self.target = target
        self.grid_resolution = grid_resolution
        if weights is not None:
            self.weights = weights
        else:
            self.weights = GridLossWeights(
                w_sinkhorn=w_sinkhorn, w_lncc=w_lncc, w_mse=w_mse, w_simse=w_simse,
                eps_sinkhorn=eps_sinkhorn, n_sinkhorn_iters=n_sinkhorn_iters, lncc_kernel=lncc_kernel,
            )

        self.network_model: Optional[NetworkModel] = None
        self.X_lattice: Optional[np.ndarray] = None
        self.Y_target: Optional[np.ndarray] = None
        self._model = model
        self._initialized = False

    def initialize(self) -> None:
        logger.info(f"Initializing TunerSession for network: {self.network.name}")

        self.network_model = NetworkModel(model=self._model, network=self.network)
        self._sample_target_lattice()
        self._unlock_bias_params()

        self._initialized = True
        logger.info("TunerSession initialized successfully")

    def _unlock_bias_params(self) -> None:
        """Unlock bias parameters that have min_value == max_value (locked in recipe).

        In the tuner UI, users should be able to modify bias values even if the recipe
        specified a fixed value. This sets a reasonable editable range for locked biases.
        """
        assert self.network_model is not None

        for path, _ in self.network_model._params.data.iter_leaves():
            path_str = str(path)
            if "bias" not in path_str or "min_value" not in path_str:
                continue

            base_path = path_str.rsplit("/min_value", 1)[0]
            min_path = f"{base_path}/min_value"
            max_path = f"{base_path}/max_value"

            if (
                min_path not in self.network_model._params
                or max_path not in self.network_model._params
            ):
                continue

            min_vals = np.asarray(self.network_model._params[min_path])
            max_vals = np.asarray(self.network_model._params[max_path])

            locked_mask = np.isclose(min_vals, max_vals)
            if not np.any(locked_mask):
                continue

            new_min = np.where(locked_mask, 0.0, min_vals)
            new_max = np.where(locked_mask, 1.0, max_vals)

            self.network_model._params[min_path] = jnp.array(new_min)
            self.network_model._params[max_path] = jnp.array(new_max)
            self.network_model._local_params[min_path] = jnp.array(new_min)
            self.network_model._local_params[max_path] = jnp.array(new_max)

            n_unlocked = int(np.sum(locked_mask))
            logger.info(f"Unlocked {n_unlocked} bias parameter(s) at {base_path}")

    def _sample_target_lattice(self) -> None:
        xres, yres = self.grid_resolution
        self.X_lattice, self.Y_target = self.target.get_lattice(resolution=(xres, yres), seed=0)
        logger.info(f"Sampled target lattice: X={self.X_lattice.shape}, Y={self.Y_target.shape}")

    @property
    def local_params(self) -> ParameterTree:
        assert self.network_model is not None
        return self.network_model._local_params

    @local_params.setter
    def local_params(self, value: ParameterTree):
        assert self.network_model is not None
        self.network_model._local_params = value

    def set_param(self, path: str, value: Union[float, list, np.ndarray]) -> None:
        import re

        assert self.network_model is not None

        # Parse indexed paths like "local/6/aggregation3x/ratios[0][1]"
        match = re.match(r"(.+?)(\[(\d+)\])+$", path)
        if match:
            base_path = match.group(1)
            indices = [int(i) for i in re.findall(r"\[(\d+)\]", path)]
            original = np.array(self.network_model._params[base_path])
            original[tuple(indices)] = float(value)
            jax_value = jnp.array(original)
            self.network_model._local_params[base_path] = jax_value
            self.network_model._params[base_path] = jax_value
            return

        original = self.network_model._params[path]
        original_dtype = original.dtype if hasattr(original, "dtype") else np.float32
        original_shape = original.shape

        def flatten_nested(v):
            if isinstance(v, list):
                return [
                    x
                    for item in v
                    for x in (flatten_nested(item) if isinstance(item, list) else [item])
                ]
            return [v]

        if isinstance(value, list):
            flat = flatten_nested(value)
            value = np.array(flat, dtype=original_dtype).reshape(original_shape)
        elif isinstance(value, (int, float)):
            value = np.array(value, dtype=original_dtype).reshape(original_shape)
        else:
            value = np.asarray(value, dtype=original_dtype).reshape(original_shape)

        jax_value = jnp.array(value)
        self.network_model._local_params[path] = jax_value
        self.network_model._params[path] = jax_value

    def get_param(self, path: str) -> np.ndarray:
        assert self.network_model is not None
        return np.asarray(self.network_model._local_params[path])

    def compute(self, include_contributions: bool = False) -> TunerResult:
        assert self._initialized and self.network_model is not None
        assert self.X_lattice is not None and self.Y_target is not None

        Y_pred_full, _ = self.network_model.predict(self.X_lattice)

        dep_mask = self.network_model.stack.get_dependent_output_mask()
        Y_pred = Y_pred_full[:, dep_mask]
        if Y_pred.shape[1] > 1:
            Y_pred = Y_pred[:, 0:1]
        Y_pred = Y_pred.squeeze(-1)

        xres, yres = self.grid_resolution
        Y_pred_grid = Y_pred.reshape(yres, xres)
        Y_target_grid = np.asarray(self.Y_target)
        if Y_target_grid.ndim == 1:
            Y_target_grid = Y_target_grid.reshape(yres, xres)

        losses, loss_result = self._compute_losses(
            Y_pred_grid, Y_target_grid, include_contributions=include_contributions
        )
        penalties = self._compute_penalties()

        result = TunerResult(
            Y_pred=Y_pred_grid,
            Y_target=Y_target_grid,
            X_lattice=np.asarray(self.X_lattice),
            losses=losses,
            penalties=penalties,
        )

        if (
            include_contributions
            and loss_result is not None
            and loss_result.lncc_contrib is not None
        ):
            result.loss_contributions = {"lncc": np.asarray(loss_result.lncc_contrib)}

        return result

    def _compute_losses(
        self, Y_pred: np.ndarray, Y_target: np.ndarray, include_contributions: bool = False
    ) -> tuple[dict[str, float], Optional[GridLossResult]]:
        """Compute losses using shared compute_grid_losses function."""
        result = compute_grid_losses(
            jnp.array(Y_pred, dtype=jnp.float32),
            jnp.array(Y_target, dtype=jnp.float32),
            weights=self.weights,
            return_contributions=include_contributions,
        )
        logger.debug(
            f"Losses: w_mse={self.weights.w_mse}, mse={result.mse}, simse={result.simse}, total={result.total}"
        )
        return result.to_dict(), result if include_contributions else None

    def _compute_penalties(self) -> dict[str, float]:
        assert self.network_model is not None
        params = ParameterTree.merge(self._model.shared_params, self.local_params)

        ratio_paths = [
            str(path)
            for path, _ in params.data.iter_leaves()
            if "ratio" in str(path) and "quantization" not in str(path)
        ]

        tucount_total = 0.0
        spread_total = 0.0

        for path in ratio_paths:
            ratios = params[path]
            if hasattr(ratios, "shape") and len(ratios.shape) >= 1:
                if ratios.ndim == 1:
                    ratios = ratios.reshape(1, -1)
                tucount_total += float(soft_tucount_penalty(ratios, max_tus=5))
                spread_total += float(ratio_spread_penalty(ratios, max_ratio=100.0))

        return {"tucount": tucount_total, "spread": spread_total}

    def reset_params(self, seed: Optional[int] = None) -> None:
        assert self.network_model is not None
        import jax

        if seed is None:
            seed = int(np.random.randint(0, 2**31))

        key = jax.random.key(seed)
        full_params = self.network_model.stack.init(key)
        _, self.network_model._local_params = full_params.filter_by_tag(["shared"])
        self.network_model._params = ParameterTree.merge(
            self._model.shared_params, self.network_model._local_params
        )
        logger.info(f"Reset params with seed {seed}")

    def get_params_dict(self) -> dict[str, list]:
        from biocomp.parameters import isArrayRef

        assert self.network_model is not None
        result = {}
        for path, val in self.local_params.data.iter_leaves():
            if isArrayRef(val):
                continue
            path_str = str(path)
            arr = np.asarray(val)
            result[path_str] = arr.tolist()
        return result

    def set_params_dict(self, params_dict: dict[str, list]) -> None:
        for path, value in params_dict.items():
            self.set_param(path, value)

    def export_params_json(self) -> str:
        return json.dumps(self.get_params_dict(), indent=2)
