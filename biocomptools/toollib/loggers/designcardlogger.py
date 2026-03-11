"""Compact circuit card logger for design replay.

Generates a 1x3 figure per step: circuit schematic, network diagram, and
prediction heatmap. For replay, reconstructs the correct step-local design
stack and commits the saved step params through the production commit path.
This keeps replay aligned with the real winning design across hard-pruning
segment rebuilds.
"""

import json
import pickle
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from pydantic import PrivateAttr

from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger
from biocomptools.toollib.loggers.logger import Logger

logger = get_logger(__name__)


def _squeeze_to_3d(arr: np.ndarray | None) -> np.ndarray | None:
    """Squeeze array to 3D by taking last element of leading dims."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    while arr.ndim > 3:
        arr = arr[-1]
    return arr


def _strip_leading_singleton_axis(tree: Any) -> Any:
    """Remove one leading singleton axis from every array leaf when present.

    Replay step params in hard-pruning runs can retain an extra size-1 axis in the
    per-(replicate, target) slice. Commit expects node-local tensors to start at
    the node axis, so we normalize that context here.
    """

    def _maybe_strip(x: Any) -> Any:
        if hasattr(x, "ndim") and hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == 1:
            return x.squeeze(axis=0)
        return x

    return jax.tree.map(_maybe_strip, tree)


def _find_run_dir(start: Path) -> Path | None:
    """Walk up from start to find directory containing metadata.json."""
    for p in [start, *start.parents]:
        if (p / "metadata.json").exists():
            return p
        if p == p.parent:
            break
    return None


def _parse_rep_number(network_name: str) -> int | None:
    match = re.search(r"_rep(\d+)$", network_name)
    if match is None:
        return None
    return int(match.group(1))


def _strip_rep_suffix(network_name: str) -> str:
    match = re.search(r"_rep\d+$", network_name)
    if match is None:
        return network_name
    return network_name[: match.start()]


def _resolve_original_flat_index(
    network_name: str,
    base_network_names: list[str],
    n_replicates: int,
) -> tuple[int, int]:
    """Compute original flat index and replicate from network name with _repN suffix."""
    rep_number = _parse_rep_number(network_name)
    if rep_number is None:
        assert network_name in base_network_names, (
            f"Network name '{network_name}' not found in {base_network_names}"
        )
        return base_network_names.index(network_name), 0

    base_name = _strip_rep_suffix(network_name)
    assert base_name in base_network_names, (
        f"Base name '{base_name}' not found in {base_network_names}"
    )
    base_idx = base_network_names.index(base_name)
    return rep_number * len(base_network_names) + base_idx, rep_number


@dataclass
class ReplayStackState:
    segment_idx: int
    start_step: int
    end_step: int | None
    dmanager: Any
    stack: Any
    original_to_current_index: dict[int, int | None]
    params_index: tuple[int, int]


class DesignCardLogger(Logger):
    """Compact circuit card + smooth prediction figure for a single tracked network."""

    call_at_interval: int = 10
    file_format: str = "pdf"
    output_dir: str | None = None
    n_inputs: int = 2

    track_recipe_hash: str | None = None
    track_network_name: str | None = None
    network_id: int | None = None
    target_id: int = 0

    required_arrays: list[str] = ["yhatdep", "X", "Y"]
    history_window: int | None = 1

    _save_dir: Path | None = PrivateAttr(default=None)
    _run_dir: Path | None = PrivateAttr(default=None)
    _original_flat_index: int = PrivateAttr(default=-1)
    _resolved_name: str = PrivateAttr(default="")
    _resolved_replicate: int = PrivateAttr(default=0)
    _resolved_target_id: int = PrivateAttr(default=0)
    _target: Any = PrivateAttr(default=None)
    _base_network_names: list[str] = PrivateAttr(default_factory=list)
    _n_base_networks: int = PrivateAttr(default=0)
    _n_replicates: int = PrivateAttr(default=0)

    _context_initialized: bool = PrivateAttr(default=False)
    _ctx_db: Any = PrivateAttr(default=None)
    _design_conf: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _initial_dmanager: Any = PrivateAttr(default=None)
    _replay_states: list[ReplayStackState] = PrivateAttr(default_factory=list)
    _hard_prune_boundaries: list[int] = PrivateAttr(default_factory=list)
    _uses_hard_pruning_replay: bool = PrivateAttr(default=False)
    _max_step: int = PrivateAttr(default=0)

    _cached_step_params: dict[int, Any] = PrivateAttr(default_factory=dict)
    _cached_committed_step: int = PrivateAttr(default=-1)
    _cached_committed_index: int | None = PrivateAttr(default=None)
    _cached_committed_net: Any = PrivateAttr(default=None)

    def initialize(self, training_program: Any = None) -> None:
        if self.output_dir:
            self._save_dir = Path(self.output_dir)
        elif training_program and hasattr(training_program, "_save_dir"):
            self._save_dir = Path(training_program._save_dir) / "cards"
        else:
            self._save_dir = Path("design_cards")
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._run_dir = _find_run_dir(self._save_dir)
        if self._run_dir is None:
            logger.warning("Could not find run directory (no metadata.json found)")
            return

        self._load_metadata()
        self._resolve_tracking_target()
        self._load_target()

    def _load_metadata(self) -> None:
        assert self._run_dir is not None
        with open(self._run_dir / "metadata.json") as f:
            meta = json.load(f)

        di = meta.get("design_info", {})
        self._base_network_names = di.get("network_names", [])
        self._n_base_networks = len(self._base_network_names)
        self._n_replicates = int(di.get("n_replicates", 1))

    def _resolve_tracking_target(self) -> None:
        if self.track_recipe_hash:
            if self._resolve_tracking_from_recipe_hash(self.track_recipe_hash):
                return
            raise ValueError(f"Could not resolve recipe hash '{self.track_recipe_hash}'")

        if self.track_network_name:
            self._resolved_name = self.track_network_name
            self._original_flat_index, self._resolved_replicate = _resolve_original_flat_index(
                self.track_network_name,
                self._base_network_names,
                self._n_replicates,
            )
            return

        if self.network_id is not None:
            self._original_flat_index = self.network_id
            self._resolved_name = f"net_{self.network_id}"
            return

        self._original_flat_index = 0
        self._resolved_name = self._base_network_names[0] if self._base_network_names else "net_0"
        logger.warning("No network tracking configured, defaulting to index 0")

    def _resolve_tracking_from_recipe_hash(self, recipe_hash: str) -> bool:
        assert self._run_dir is not None
        best_designs_path = self._run_dir / "best_designs.pickle"
        if not best_designs_path.exists():
            return False

        with open(best_designs_path, "rb") as f:
            best_designs = pickle.load(f)

        for target_name, design_info in best_designs.items():
            if design_info.get("recipe_hash") != recipe_hash:
                continue
            self._resolved_name = design_info.get("network_name") or self.track_network_name or ""
            self._original_flat_index = int(design_info.get("network_id", 0))
            self._resolved_replicate = int(design_info.get("replicate", 0))
            if self.target_id == 0:
                self._resolved_target_id = int(design_info.get("target_id", 0))
                self.target_id = self._resolved_target_id
            else:
                self._resolved_target_id = self.target_id
            logger.info(
                "Resolved recipe hash '%s' -> target=%s network=%s flat_index=%s",
                recipe_hash,
                target_name,
                self._resolved_name,
                self._original_flat_index,
            )
            return True
        return False

    def _load_target(self) -> None:
        assert self._run_dir is not None
        bd_path = self._run_dir / "best_designs.pickle"
        if not bd_path.exists():
            return
        with open(bd_path, "rb") as f:
            best_designs: dict[str, dict[str, Any]] = pickle.load(f)
        for _tname, data in best_designs.items():
            target_id = int(data.get("target_id", 0))
            if target_id != self.target_id:
                continue
            self._target = data.get("target")
            if self._target:
                break
        if self._target is None:
            for data in best_designs.values():
                self._target = data.get("target")
                if self._target is not None:
                    break

    def _ensure_context_init(self, context: LoggerContext) -> None:
        if self._context_initialized:
            return
        self._context_initialized = True

        db = context.db
        if db is None:
            logger.warning("No DB in context, replay commit disabled")
            return
        self._ctx_db = db
        self._initial_dmanager = context.dmanager or db.load_artifact("dmanager")
        self._design_conf = db.load_artifact("dconfig")
        self._model = context.model or db.load_artifact("model")
        if self._ctx_db is not None:
            _, self._max_step = self._ctx_db.get_step_range()
        self._initialize_replay_states()

    def _initialize_replay_states(self) -> None:
        if self._initial_dmanager is None or self._design_conf is None or self._model is None:
            return

        dmanager = self._initial_dmanager
        dconf = self._design_conf
        params_index = (self._resolved_replicate, self.target_id)
        if getattr(dconf, "hard_pruning_enabled", False):
            from biocomp.design_prune_controller import build_stack_from_dconf
            from biocomp.design_pruning import _flatten_replicates_into_networks

            if int(getattr(dconf, "n_replicates", 1)) > 1:
                dmanager = _flatten_replicates_into_networks(dmanager, int(dconf.n_replicates))
                dconf = dconf.model_copy(update={"n_replicates": 1})
                params_index = (0, 0)
            stack = build_stack_from_dconf(dmanager, dconf, self._model, lock_ratios=False)
            self._design_conf = dconf
            self._uses_hard_pruning_replay = True
        else:
            stack = dmanager.build_stack(self._model)
            self._uses_hard_pruning_replay = False

        mapping = {idx: idx for idx in range(len(dmanager.networks))}
        self._replay_states = [
            ReplayStackState(
                segment_idx=0,
                start_step=0,
                end_step=None,
                dmanager=dmanager,
                stack=stack,
                original_to_current_index=mapping,
                params_index=params_index,
            )
        ]

        if self._uses_hard_pruning_replay:
            interval = int(getattr(self._design_conf, "hard_pruning_interval", 0))
            if interval > 0:
                self._hard_prune_boundaries = list(range(interval, self._max_step, interval))

    def _get_params_for_step(self, step: int) -> Any | None:
        if step in self._cached_step_params:
            return self._cached_step_params[step]
        if self._ctx_db is None:
            return None
        params = self._ctx_db.load_blob(step, "latest_params")
        if params is None:
            params = self._ctx_db.load_blob(step, "params")
        self._cached_step_params[step] = params
        return params

    def _get_state_for_step(self, step: int) -> ReplayStackState | None:
        if not self._replay_states:
            return None
        if not self._uses_hard_pruning_replay:
            return self._replay_states[0]

        target_idx = sum(1 for boundary in self._hard_prune_boundaries if boundary < step)
        while len(self._replay_states) - 1 < target_idx:
            prev_state = self._replay_states[-1]
            boundary_step = self._hard_prune_boundaries[len(self._replay_states) - 1]
            next_state = self._advance_state(prev_state, boundary_step)
            self._replay_states[-1].end_step = boundary_step
            self._replay_states.append(next_state)
        return self._replay_states[target_idx]

    def _advance_state(self, prev_state: ReplayStackState, boundary_step: int) -> ReplayStackState:
        from biocomp.design_pruning import (
            _compute_hard_pruning_network_keep_count,
            _select_top_network_indices_from_losses,
            hard_prune_and_rebuild,
            identify_tus_to_prune,
        )
        from biocomp.design_prune_controller import evaluate_segment_snapshot
        from biocomp.jaxutils import tree_get
        import jax

        params = self._get_params_for_step(boundary_step)
        if params is None:
            logger.warning("Boundary step %s missing params; carrying forward previous state", boundary_step)
            return ReplayStackState(
                segment_idx=prev_state.segment_idx + 1,
                start_step=boundary_step,
                end_step=None,
                dmanager=prev_state.dmanager,
                stack=prev_state.stack,
                original_to_current_index=dict(prev_state.original_to_current_index),
                params_index=prev_state.params_index,
            )

        single_params = tree_get(params, prev_state.params_index)
        tus_to_remove = identify_tus_to_prune(
            single_params,
            prev_state.stack,
            prev_state.dmanager,
            ratio_threshold=float(self._design_conf.hard_pruning_ratio_threshold),
            use_soft_pruning=bool(self._design_conf.enable_tu_masking),
            preserve_minimum=int(self._design_conf.hard_pruning_preserve_minimum_tus),
            prune_margin=float(self._design_conf.hard_pruning_prune_margin),
            auto_lock_topology_tus=bool(self._design_conf.auto_lock_topology_tus),
        )

        keep_indices = None
        keep_count = _compute_hard_pruning_network_keep_count(
            len(prev_state.dmanager.networks),
            self._design_conf.hard_pruning_top_percent,
            self._design_conf.hard_pruning_min_networks,
        )
        if keep_count is not None and keep_count < len(prev_state.dmanager.networks):
            compare_key = jax.random.fold_in(self._design_conf.seed_key, prev_state.segment_idx + 2000)
            snapshot = evaluate_segment_snapshot(prev_state.dmanager, self._design_conf, self._model, params, compare_key)
            keep_indices = _select_top_network_indices_from_losses(np.asarray(snapshot.loss), keep_count)

        total_to_remove = sum(len(v) for v in tus_to_remove.values())
        if total_to_remove == 0 and keep_indices is None:
            return ReplayStackState(
                segment_idx=prev_state.segment_idx + 1,
                start_step=boundary_step,
                end_step=None,
                dmanager=prev_state.dmanager,
                stack=prev_state.stack,
                original_to_current_index=dict(prev_state.original_to_current_index),
                params_index=prev_state.params_index,
            )

        prune_key = jax.random.fold_in(self._design_conf.seed_key, prev_state.segment_idx + 1000)
        new_dmanager, new_stack, _ = hard_prune_and_rebuild(
            prev_state.dmanager,
            self._design_conf,
            self._model,
            prev_state.stack,
            single_params,
            tus_to_remove,
            prune_key,
            lock_ratios=False,
            keep_network_indices=keep_indices,
        )
        new_name_to_idx = {net.name: idx for idx, net in enumerate(new_dmanager.networks)}
        new_mapping: dict[int, int | None] = {}
        for original_idx, current_idx in prev_state.original_to_current_index.items():
            if current_idx is None or current_idx >= len(prev_state.dmanager.networks):
                new_mapping[original_idx] = None
                continue
            old_name = prev_state.dmanager.networks[current_idx].name
            new_mapping[original_idx] = new_name_to_idx.get(old_name)

        return ReplayStackState(
            segment_idx=prev_state.segment_idx + 1,
            start_step=boundary_step,
            end_step=None,
            dmanager=new_dmanager,
            stack=new_stack,
            original_to_current_index=new_mapping,
            params_index=prev_state.params_index,
        )

    def _get_current_network_index(self, step: int) -> int | None:
        state = self._get_state_for_step(step)
        if state is None:
            return self._original_flat_index if self._original_flat_index >= 0 else None
        return state.original_to_current_index.get(self._original_flat_index)

    def _get_committed_network_at_step(self, step: int) -> tuple[Any | None, int | None]:
        if step == self._cached_committed_step and self._cached_committed_net is not None:
            return self._cached_committed_net, self._cached_committed_index

        state = self._get_state_for_step(step)
        if state is None:
            return None, None
        current_idx = state.original_to_current_index.get(self._original_flat_index)
        if current_idx is None:
            logger.warning("Tracked design %s no longer exists at step %s", self._resolved_name, step)
            return None, None

        params = self._get_params_for_step(step)
        if params is None:
            return None, current_idx

        from biocomp.jaxutils import tree_get

        commit_params = _strip_leading_singleton_axis(tree_get(params, state.params_index))
        try:
            committed_networks = state.stack.commit(commit_params)
            if current_idx >= len(committed_networks):
                logger.warning(
                    "Tracked network index %s out of bounds for committed set size %s at step %s",
                    current_idx,
                    len(committed_networks),
                    step,
                )
                return None, None
            committed_net = committed_networks[current_idx]
        except Exception as e:
            logger.warning(
                "Step %s: commit failed for %s, falling back to segment-local network: %s",
                step,
                self._resolved_name,
                e,
            )
            if current_idx >= len(state.dmanager.networks):
                return None, None
            committed_net = deepcopy(state.dmanager.networks[current_idx])
        self._cached_committed_step = step
        self._cached_committed_index = current_idx
        self._cached_committed_net = committed_net
        return committed_net, current_idx

    def _get_display_network(self, step: int | None = None) -> tuple[Any | None, int | None]:
        if step is not None and self._ctx_db is not None:
            return self._get_committed_network_at_step(step)
        return None, None

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        self._ensure_context_init(context)
        step = context.current_step
        latest = view.latest()
        if latest is None:
            return
        step_history: dict[str, Any] = {"loss": latest.loss}
        step_history.update(latest.metrics)
        step_history.update(latest.arrays)
        self._render_card(step, step_history)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        self._ensure_context_init(context)
        latest = view.latest()
        if latest is None:
            return
        step = context.current_step
        step_history: dict[str, Any] = {"loss": latest.loss}
        step_history.update(latest.metrics)
        step_history.update(latest.arrays)
        self._render_card(step, step_history)

    def _render_card(self, step: int, step_history: dict[str, Any]) -> None:
        from biocomp.datautils import IdentityRescaler
        from biocomp.plotutils import PlotData, smooth

        assert self._save_dir is not None

        tid = self.target_id
        display_net, current_idx = self._get_display_network(step)
        if current_idx is None:
            current_idx = self._get_current_network_index(step)
        if current_idx is None or display_net is None:
            logger.warning("Step %s: tracked design unavailable, skipping card", step)
            return

        yhatdep = _squeeze_to_3d(step_history.get("yhatdep"))
        X_hist = _squeeze_to_3d(step_history.get("X"))
        Y_hist = _squeeze_to_3d(step_history.get("Y"))

        if yhatdep is None or X_hist is None or Y_hist is None:
            logger.warning(f"Step {step}: missing arrays, skipping card")
            return

        n_nets = yhatdep.shape[2] if yhatdep.ndim == 3 else 1
        if current_idx >= n_nets:
            logger.warning(
                f"Step {step}: network_id {current_idx} >= n_nets {n_nets}, skipping"
            )
            return

        ni = self.n_inputs
        X = X_hist[:, tid, current_idx * ni : (current_idx + 1) * ni]
        Yhat = yhatdep[:, tid, current_idx]
        X = np.asarray(X)
        Yhat = np.asarray(Yhat)
        while X.ndim > 2:
            X = X[:, 0, :]
        while Yhat.ndim > 2:
            Yhat = Yhat[:, 0]
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Yhat.ndim == 1:
            Yhat = Yhat.reshape(-1, 1)

        input_names = [f"x{i + 1}" for i in range(ni)]
        rescaler = IdentityRescaler()
        pred_data = PlotData(xval=X, yval=Yhat, input_names=input_names)

        if self._target is not None:
            lx = getattr(self._target, "latent_x", (None, None))
            ly = getattr(self._target, "latent_y", (None, None))
            xlims: tuple[float | None, float | None] = (lx[0], lx[1])
            ylims: tuple[float | None, float | None] = (ly[0], ly[1])
        else:
            xlims = (None, None)
            ylims = (None, None)

        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.35)
        ax_circuit = fig.add_subplot(gs[0, 0])
        ax_diagram = fig.add_subplot(gs[0, 1])
        ax_pred = fig.add_subplot(gs[0, 2])

        try:
            from biocomptools.toollib.figuremakers.geneticcircuit import render_circuit_to_ax

            render_circuit_to_ax(display_net, ax_circuit, title="Circuit")
        except Exception as e:
            logger.warning(f"Circuit render failed: {e}")
            ax_circuit.text(0.5, 0.5, "Circuit\nunavailable", ha="center", va="center", transform=ax_circuit.transAxes)
            ax_circuit.axis("off")

        try:
            from biocomptools.toollib.figuremakers.networkdiagram import render_diagram_to_ax

            render_diagram_to_ax(display_net, ax_diagram, title="Diagram", show_ratios=True)
        except Exception as e:
            logger.warning(f"Diagram render failed: {e}")
            ax_diagram.text(0.5, 0.5, "Diagram\nunavailable", ha="center", va="center", transform=ax_diagram.transAxes)
            ax_diagram.axis("off")

        try:
            smooth(
                plot_data=pred_data,
                ax=ax_pred,
                title="Prediction",
                rescaler=rescaler,
                draw_colorbar=True,
                xlims=xlims,
                ylims=ylims,
                smooth_2d_params={"vlims": (None, None)},
                vlims=(None, None),
            )
        except Exception as e:
            logger.warning(f"Prediction smooth failed: {e}")
            ax_pred.set_title("Prediction")
            ax_pred.text(0.5, 0.5, f"Render failed:\n{e}", ha="center", va="center", transform=ax_pred.transAxes, fontsize=8)

        loss = step_history.get("loss", float("nan"))
        name = self._resolved_name or f"net_{self._original_flat_index}"
        fig.suptitle(
            f"Step {step} | {name} | current_id={current_idx} | loss={loss:.4f}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        out_path = self._save_dir / f"step_{step:06d}_card.{self.file_format}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved card: {out_path.name}")
