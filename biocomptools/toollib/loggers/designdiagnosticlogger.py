# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, PrivateAttr

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.loggers.utils import to_scalar as _to_scalar
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger
from biocomp.compute import ComputeStack

logger = get_logger(__name__)


@dataclass
class RatioInfo:
    """Metadata for ratio parameters in an aggregation layer."""

    cotx_name: str
    tu_names: list[str]


def _build_ratio_metadata(stack: ComputeStack) -> dict[str, list[RatioInfo]]:
    """Build mapping from namespace to ratio metadata (cotx names, TU names).

    This enables descriptive labeling of ratio parameters in the particle plot,
    e.g., "CoTx1:TU_A" instead of "ratios.0".

    Similar to biocomp-tuner's param_schema._build_ratio_metadata.
    """
    metadata: dict[str, list[RatioInfo]] = {}
    if stack.layers is None:
        return metadata

    for i, layer in enumerate(stack.layers):
        type_name = layer.type_str()
        if "aggregation" not in type_name.lower() or "inv" in type_name.lower():
            continue

        ns = stack.get_layer_namespace(i)
        ratio_infos = []
        for node in layer.nodes:
            full_node = node.get(stack)
            extra = full_node.extra
            cotx_name = extra.get("cotx_group", "unknown")
            members_data = extra.get("members", {})
            tu_names = sorted(members_data.keys()) if isinstance(members_data, dict) else []
            ratio_infos.append(RatioInfo(cotx_name=cotx_name, tu_names=tu_names))
        metadata[ns] = ratio_infos

    return metadata


def _build_transform_metadata(stack: ComputeStack) -> dict[str, list[list[str]]]:
    """Build mapping from namespace to input TU names for transform layers.

    Transform layers (translation, transcription) have multiple inputs, each associated
    with a source TU. This enables labeling like "tc_rate:TU_A" instead of "tc_rate.0".
    """
    metadata: dict[str, list[list[str]]] = {}
    if stack.layers is None:
        return metadata

    for i, layer in enumerate(stack.layers):
        type_name = layer.type_str()
        if "translation" not in type_name.lower() and "transcription" not in type_name.lower():
            continue
        if "inv" in type_name.lower():
            continue

        ns = stack.get_layer_namespace(i)
        node_tu_names = []
        for node in layer.nodes:
            incoming_edges = node.get_incoming_edges(stack)
            tu_names = []
            for edge in incoming_edges:
                if edge.extra:
                    tu_id_list = edge.extra.get("tu_id", [])
                    if tu_id_list:
                        tu_names.append(tu_id_list[0].split("_")[0] if tu_id_list else "?")
                    else:
                        tu_names.append("?")
                else:
                    tu_names.append("?")
            node_tu_names.append(tu_names)
        metadata[ns] = node_tu_names

    return metadata


def unroll_dict(d: dict, prefix: str = "") -> dict[str, float]:
    result = {}
    for key, val in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            result.update(unroll_dict(val, full_key))
        elif hasattr(val, 'shape') and np.asarray(val).size > 1:
            flat = np.asarray(val).ravel()
            for i, v in enumerate(flat):
                result[f"{full_key}.{i}"] = float(v)
        else:
            result[full_key] = _to_scalar(val)
    return result


@dataclass
class ParamMetadata:
    """Metadata for descriptive parameter labeling."""

    ratio_metadata: dict[str, list[RatioInfo]]
    transform_metadata: dict[str, list[list[str]]]
    tu_idx_to_id: dict[int, str] | None = None


def unroll_params(
    params,
    replicate: int,
    target: int,
    network: int,
    tu_idx_to_id: dict[int, str] | None = None,
    param_metadata: ParamMetadata | None = None,
) -> dict[str, float]:
    """Extract flattened param dict for a specific replicate/target/network.

    Handles both:
    - latest_params: shape (n_replicates, n_targets, ...)
    - params from scan: shape (batches_per_step, n_replicates, n_targets, ...)

    When param_metadata is provided, creates descriptive labels for:
    - Ratios: "CoTx1:TU_A" instead of "ratios.0"
    - Transform rates: "tl_rate:TU_A" instead of "tl_rate.0"
    - TU log alpha: "TU:name" instead of "tu_log_alpha.0"
    """
    result = {}
    if params is None:
        return result

    try:
        # Get ALL leaves first (including non_grad) for network_id_map
        if hasattr(params, 'data') and hasattr(params.data, 'iter_leaves'):
            all_leaves = list(params.data.iter_leaves())
        else:
            return result

        # Get dynamic leaves for param extraction
        if hasattr(params, 'filter_by_tag'):
            _, dynamic = params.filter_by_tag(["non_grad", "shared"])
            leaves = list(dynamic.data.iter_leaves()) if hasattr(dynamic, 'data') else []
        else:
            leaves = all_leaves
    except (AttributeError, TypeError):
        return result

    skip_patterns = [
        'indices',
        'node_layer',
        'mask',
        'random_variable',
        'input_tu_indices',
        'output_tu_indices',
        'node_network_ids',
        'original_positions',
        'original_slots',
        'fwd_node_positions',
        'dependent_output_mask',
        'number_of_random_variables',
    ]

    def extract_replicate_target(arr, rep, tgt):
        """Extract (rep, tgt) slice, handling both 3D and 4D (with batches dim) arrays."""
        arr = np.asarray(arr)
        # If 4D+: (batches, reps, targets, ...) - take last batch
        if arr.ndim >= 4:
            arr = arr[-1]  # last batch
        # Now handle standard (reps, targets, ...) or (reps, ...) shape
        if arr.ndim >= 3 and arr.shape[0] > rep:
            if arr.shape[1] > tgt:
                return arr[rep, tgt]
            return arr[rep, 0]
        elif arr.ndim >= 2 and arr.shape[0] > rep:
            return arr[rep]
        return arr

    # Build a map of namespace -> node_network_ids for filtering by network
    # Use all_leaves since node_network_ids are tagged as non_grad
    network_id_map: dict[str, np.ndarray] = {}
    # Track which local indices map to which original node positions per namespace
    network_node_indices: dict[str, list[int]] = {}

    for path, value in all_leaves:
        path_str = str(path)
        if 'node_network_ids' in path_str:
            try:
                arr = value.get_array() if hasattr(value, 'get_array') else value
                arr = extract_replicate_target(arr, replicate, target)
                while arr.ndim > 1 and arr.shape[0] == 1:
                    arr = arr[0]
                namespace = path_str.rsplit('/node_network_ids', 1)[0]
                net_ids = np.asarray(arr).ravel()
                network_id_map[namespace] = net_ids
                # Track original indices for nodes belonging to this network
                network_node_indices[namespace] = [
                    i for i, nid in enumerate(net_ids) if nid == network
                ]
            except (IndexError, KeyError, TypeError, ValueError):
                pass

    # Extract ratio and transform metadata for labeling
    ratio_meta = param_metadata.ratio_metadata if param_metadata else {}
    transform_meta = param_metadata.transform_metadata if param_metadata else {}
    tu_map = param_metadata.tu_idx_to_id if param_metadata else tu_idx_to_id

    for path, value in leaves:
        path_str = str(path)
        if any(skip in path_str for skip in skip_patterns):
            continue

        try:
            if hasattr(value, 'get_array'):
                arr = np.asarray(value.get_array())
            elif hasattr(value, 'shape'):
                arr = np.asarray(value)
            else:
                continue

            if arr.ndim < 2 or arr.size == 0:
                continue

            # Check if this is tu_log_alpha (special case: indexed by network, not rep/target)
            is_tu_log_alpha = 'tu_log_alpha' in path_str
            if is_tu_log_alpha:
                # tu_log_alpha has shape (n_networks, n_tus) - no replicate/target dims
                # May have batch dim: (batches, n_networks, n_tus)
                if arr.ndim >= 3:
                    arr = arr[-1]  # Take last batch
                # Now shape is (n_networks, n_tus) - extract this network's TUs
                if arr.ndim == 2 and arr.shape[0] > network:
                    arr = arr[network]  # (n_tus,) for this network
                else:
                    continue  # Skip if can't extract for this network
            else:
                # Handle replicate/target/batch dimensions for other params
                arr = extract_replicate_target(arr, replicate, target)

                # Squeeze singleton dimensions
                while arr.ndim > 1 and arr.shape[0] == 1:
                    arr = arr[0]

            # Determine namespace and check for descriptive metadata
            namespace = None
            original_node_indices = None
            for ns in network_id_map:
                if path_str.startswith(ns + '/'):
                    namespace = ns
                    original_node_indices = network_node_indices.get(ns)
                    break

            if namespace and namespace in network_id_map:
                net_ids = network_id_map[namespace]
                # arr shape is typically (n_nodes, ...) - filter to nodes belonging to this network
                if arr.ndim >= 1 and arr.shape[0] == len(net_ids):
                    mask = net_ids == network
                    if not np.any(mask):
                        continue  # No nodes for this network in this layer
                    arr = arr[mask]

            param_flat = np.asarray(arr).ravel()
            max_per_param = 100
            if param_flat.size > max_per_param:
                param_flat = param_flat[:max_per_param]

            # Determine param type for labeling
            is_ratio = '/ratios' in path_str and namespace in ratio_meta
            is_rate = '_rate' in path_str or 'tl_rate' in path_str or 'tc_rate' in path_str
            is_transform_rate = is_rate and namespace in transform_meta

            # Build short path for fallback
            short_path = (
                path_str.replace('design/', 'd/')
                .replace('local/', 'l/')
                .replace('shared/', 's/')
                .replace('/aggregation', '/agg')
                .replace('/translation', '/tl')
                .replace('/transcription', '/tx')
                .replace('/sequestron_ERN', '/ern')
                .replace('/bias', '/b')
            )

            for i, v in enumerate(param_flat):
                key = None

                if is_tu_log_alpha and tu_map and i in tu_map:
                    tu_name = tu_map[i][:20]
                    key = f"TU:{tu_name}"

                elif is_ratio and original_node_indices and namespace:
                    # Use ratio metadata for descriptive label
                    ratio_infos = ratio_meta[namespace]
                    # Map local index to original node index
                    if len(original_node_indices) == 1:
                        # Single node for this network
                        orig_node_idx = original_node_indices[0]
                        if orig_node_idx < len(ratio_infos):
                            ri = ratio_infos[orig_node_idx]
                            # i indexes into TU names within this node's ratios
                            if i < len(ri.tu_names):
                                cotx_short = ri.cotx_name[:8] if ri.cotx_name else "?"
                                tu_short = ri.tu_names[i][:12] if ri.tu_names[i] else f"TU{i}"
                                key = f"{cotx_short}:{tu_short}"
                    else:
                        # Multiple nodes: i might span across nodes
                        # Each node has n_ratios values
                        # Figure out which node and which ratio within that node
                        cumulative = 0
                        for _local_idx, orig_idx in enumerate(original_node_indices):
                            if orig_idx < len(ratio_infos):
                                ri = ratio_infos[orig_idx]
                                n_ratios = len(ri.tu_names) if ri.tu_names else 1
                                if i < cumulative + n_ratios:
                                    ratio_idx = i - cumulative
                                    cotx_short = ri.cotx_name[:8] if ri.cotx_name else "?"
                                    tu_short = (
                                        ri.tu_names[ratio_idx][:12]
                                        if ratio_idx < len(ri.tu_names) and ri.tu_names[ratio_idx]
                                        else f"TU{ratio_idx}"
                                    )
                                    key = f"{cotx_short}:{tu_short}"
                                    break
                                cumulative += n_ratios

                elif is_transform_rate and original_node_indices and namespace:
                    # Use transform metadata for descriptive label
                    node_tu_lists = transform_meta[namespace]
                    rate_type = (
                        "tl" if "tl_rate" in path_str or "/translation" in path_str else "tc"
                    )
                    if len(original_node_indices) == 1:
                        orig_node_idx = original_node_indices[0]
                        if orig_node_idx < len(node_tu_lists):
                            tu_names = node_tu_lists[orig_node_idx]
                            if i < len(tu_names):
                                key = f"{rate_type}:{tu_names[i][:12]}"
                    else:
                        # Multiple nodes: i spans across nodes
                        # Each node has n_inputs rate values (one per input edge)
                        cumulative = 0
                        for _local_idx, orig_idx in enumerate(original_node_indices):
                            if orig_idx < len(node_tu_lists):
                                tu_names = node_tu_lists[orig_idx]
                                n_inputs = len(tu_names) if tu_names else 1
                                if i < cumulative + n_inputs:
                                    input_idx = i - cumulative
                                    if input_idx < len(tu_names):
                                        key = f"{rate_type}:{tu_names[input_idx][:12]}"
                                    break
                                cumulative += n_inputs

                # Fallback to generic label
                if key is None:
                    if param_flat.size > 1:
                        key = f"{short_path}.{i}"
                    else:
                        key = short_path

                result[key] = float(v)

        except (IndexError, KeyError, TypeError, ValueError):
            continue
    return result


def unroll_grads(
    grads,
    replicate: int,
    target: int,
    network: int,
    tu_idx_to_id: dict[int, str] | None = None,
    param_metadata: ParamMetadata | None = None,
) -> dict[str, float]:
    return unroll_params(grads, replicate, target, network, tu_idx_to_id, param_metadata)


def prepare_particle_data(
    history: list[dict],
    keys: list[str],
    derivative_history: list[dict] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    n_time = len(history)
    n_vars = len(keys)
    data = np.full((n_vars, n_time), np.nan)
    for t, h in enumerate(history):
        for i, key in enumerate(keys):
            data[i, t] = h.get(key, np.nan)
    derivatives = None
    if derivative_history and len(derivative_history) > 0:
        last_deriv = derivative_history[-1]
        derivatives = np.array([last_deriv.get(key, 0.0) for key in keys])
    return data, keys, derivatives


def _squeeze_to_3d(arr: np.ndarray | None) -> np.ndarray | None:
    """Squeeze array to 3D by taking the last element of leading dimensions.

    Takes arr[-1] (not arr[0]) to get the latest batch when arrays have
    extra dimensions from scan/replicates: (n_reps, batches_per_step, batch_size, ...)
    """
    if arr is None:
        return None
    arr = np.asarray(arr)
    while arr.ndim > 3:
        arr = arr[-1]
    return arr


class DesignDiagnosticLogger(Logger):
    """Design diagnostic logger with new pattern support.

    Supports both legacy get_callbacks pattern and new on_batch/on_end pattern.
    The new pattern uses centralized history from the handler.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # configuration
    output_dir: str | None = None
    history_dir: str | None = None  # For replay: directory containing step_*.pkl files
    call_at_interval: int = 10
    max_history_len: int = 100
    generate_plots: bool = True
    final_figure_only: bool = False
    save_history: bool = True
    max_networks_to_plot: int = 4
    max_targets_to_plot: int = 2
    particle_plot_spacing: int = 55
    file_format: Literal["pdf", "png"] = "png"

    # output structure: "step_first" (default) or "network_first"
    # step_first:    step_XXXXXX/target_N_name/network_M.pdf
    # network_first: network_M/target_name/design_diagnostic_step_XXXXXX.pdf
    output_structure: Literal["step_first", "network_first"] = "step_first"

    # network selection strategy - CRITICAL for stable tracking across steps
    # "fixed": use network indices 0, 1, ... max_networks_to_plot (default, original behavior)
    # "best_at_start": pick the best networks at first step, track those same networks throughout
    # "by_name": track specific networks by name (use network_names_to_plot)
    network_selection: Literal["fixed", "best_at_start", "by_name"] = "fixed"
    network_names_to_plot: list[str] | None = None  # for "by_name" selection

    # parallel figure generation: defer figure generation and batch them
    deferred_figures: bool = False  # if True, queue figures instead of generating

    # new pattern attributes
    history_window: int | None = 50
    required_metrics: list[str] = [
        'loss',
        'all_losses',
        'sublosses',
        'tu_stats',
        'ratio_stats',
        'l0_penalty',
        'tucount_penalty',
        'spread_penalty',
        'coupling_penalty',
        'ern_tying_penalty',
    ]
    required_arrays: list[str] = ['yhatdep', 'X', 'Y', 'params', 'latest_params', 'grad']

    # internal state
    _save_dir: Path | None = PrivateAttr(default=None)
    _dmanager: Any = PrivateAttr(default=None)
    _design_config: Any = PrivateAttr(default=None)
    _grid_resolution: tuple[int, int] | None = PrivateAttr(default=None)
    _total_steps: int = PrivateAttr(default=0)
    _history: dict = PrivateAttr(default_factory=dict)
    _param_history: dict = PrivateAttr(default_factory=dict)
    _grad_history: dict = PrivateAttr(default_factory=dict)
    _network_names: list[str] = PrivateAttr(default_factory=list)
    _tu_idx_to_id: dict[int, str] | None = PrivateAttr(default=None)
    _stack: Any = PrivateAttr(default=None)
    _networks: list = PrivateAttr(default_factory=list)  # Networks loaded from replay
    _pending_figures: list = PrivateAttr(default_factory=list)  # Deferred figure tasks
    _selected_network_indices: dict[int, list[int]] = PrivateAttr(
        default_factory=dict
    )  # tid -> [nid, ...]
    _param_metadata: ParamMetadata | None = PrivateAttr(
        default=None
    )  # For descriptive param labels
    _history_dir: Path | None = PrivateAttr(default=None)  # For replay mode stack loading

    def initialize(self, training_program=None):
        if self.output_dir:
            self._save_dir = Path(self.output_dir)
        elif training_program and hasattr(training_program, '_save_dir'):
            self._save_dir = training_program._save_dir / 'diagnostics'
        else:
            self._save_dir = Path('diagnostics')
        self._save_dir.mkdir(parents=True, exist_ok=True)

        if self.history_dir:
            self._history_dir = Path(self.history_dir)

        has_dmanager = False
        if training_program:
            if hasattr(training_program, '_dmanager') and training_program._dmanager is not None:
                has_dmanager = True
                self._dmanager = training_program._dmanager
                if hasattr(self._dmanager, 'grid_resolution') and self._dmanager.grid_resolution:
                    self._grid_resolution = self._dmanager.grid_resolution
                if hasattr(self._dmanager, 'tu_id_to_idx') and self._dmanager.tu_id_to_idx:
                    self._tu_idx_to_id = {v: k for k, v in self._dmanager.tu_id_to_idx.items()}
            if hasattr(training_program, 'design_conf'):
                self._design_config = training_program.design_conf
                if hasattr(self._design_config, 'n_epochs') and hasattr(
                    self._design_config, 'n_batches_per_epoch'
                ):
                    self._total_steps = (
                        self._design_config.n_epochs * self._design_config.n_batches_per_epoch
                    )

        # Replay mode: try loading networks from run directory if no dmanager available
        if not has_dmanager:
            self._load_networks_from_run_dir()

        logger.info(
            f"DesignDiagnosticLogger initialized: {self._save_dir}, "
            f"resolution={self._grid_resolution}, total_steps={self._total_steps}"
        )

    def _load_networks_from_run_dir(self):
        """Try to load networks from design_networks.pickle or best_designs.pickle."""
        import pickle

        if not self._save_dir:
            return

        # output_dir could be .../replay_output or .../replay_output/final/...
        # run_dir is parent of step_history_data, which is sibling of replay_output
        candidates = [
            self._save_dir.parent,  # replay_output -> run_dir
            self._save_dir.parent.parent,  # step_history_data/replay_output -> run_dir
            self._save_dir.parent.parent.parent,  # replay_output/final/target -> run_dir
            self._save_dir.parent.parent.parent.parent,  # replay_output/final/target/net -> run
        ]

        # First try design_networks.pickle (available early, created at start of design)
        for candidate in candidates:
            networks_file = candidate / 'design_networks.pickle'
            if networks_file.exists():
                try:
                    with open(networks_file, 'rb') as f:
                        data = pickle.load(f)
                    self._networks = data.get('networks', [])
                    self._network_names = data.get('network_names', [])
                    logger.info(f"Loaded {len(self._networks)} networks from {networks_file}")
                    return
                except Exception as e:
                    logger.debug(f"Could not load from {networks_file}: {e}")

        # Fall back to best_designs.pickle (created after evaluation)
        for candidate in candidates:
            designs_file = candidate / 'best_designs.pickle'
            if designs_file.exists():
                try:
                    with open(designs_file, 'rb') as f:
                        designs_data = pickle.load(f)
                    for _target_name, target_data in designs_data.items():
                        if isinstance(target_data, dict) and 'network' in target_data:
                            self._networks.append(target_data['network'])
                            self._network_names.append(
                                target_data.get(
                                    'network_name', f'network_{len(self._networks) - 1}'
                                )
                            )
                    logger.info(f"Loaded {len(self._networks)} networks from {designs_file}")
                    return
                except Exception as e:
                    logger.debug(f"Could not load from {designs_file}: {e}")

        logger.debug("Could not locate any network files for replay")

    def _select_networks_for_target(
        self, tid: int, n_networks: int, step_history: dict | None = None
    ) -> list[int]:
        """Select which network indices to track for this target.

        Selection strategies:
        - "fixed": use indices 0, 1, ... max_networks_to_plot
        - "best_at_start": pick best networks by loss at first step, track those forever
        - "by_name": find indices matching network_names_to_plot
        """
        max_nets = min(n_networks, self.max_networks_to_plot)

        if self.network_selection == "fixed":
            return list(range(max_nets))

        if self.network_selection == "by_name":
            if not self.network_names_to_plot:
                logger.warning("network_selection='by_name' but no network_names_to_plot specified")
                return list(range(max_nets))
            indices = []
            for name in self.network_names_to_plot:
                if name in self._network_names:
                    indices.append(self._network_names.index(name))
                else:
                    logger.warning(f"Network '{name}' not found in available networks")
            return indices[:max_nets] if indices else list(range(max_nets))

        if self.network_selection == "best_at_start":
            if step_history is None:
                return list(range(max_nets))
            all_losses = step_history.get("all_losses")
            if all_losses is None:
                return list(range(max_nets))
            arr = np.asarray(all_losses)
            # Extract losses for this target across all networks
            # Shape: (n_replicates, batches_per_step, n_targets, n_networks) or (n_targets, n_networks)
            if arr.ndim == 4:
                net_losses = np.nanmean(arr[0, :, tid, :], axis=0)  # avg over batches, rep 0
            elif arr.ndim == 3:
                net_losses = np.nanmean(arr[:, tid, :], axis=0)
            elif arr.ndim == 2:
                net_losses = arr[tid, :]
            else:
                return list(range(max_nets))
            # Sort by loss (ascending = best first)
            sorted_indices = np.argsort(net_losses).tolist()
            return sorted_indices[:max_nets]

        return list(range(max_nets))

    def _get_selected_networks(
        self, tid: int, n_networks: int, step_history: dict | None = None
    ) -> list[int]:
        """Get or create stable network selection for a target."""
        if tid not in self._selected_network_indices:
            self._selected_network_indices[tid] = self._select_networks_for_target(
                tid, n_networks, step_history
            )
            net_names = [self._get_network_name(nid) for nid in self._selected_network_indices[tid]]
            logger.info(
                f"Target {tid}: selected networks {self._selected_network_indices[tid]} "
                f"({net_names}) using '{self.network_selection}' strategy"
            )
        return self._selected_network_indices[tid]

    def _extract_metrics(
        self, step: int, step_history: dict, target_id: int, network_id: int
    ) -> dict:
        """Extract per-target/network metrics from step_history.

        Step history arrays have shape (n_replicates, batches_per_step, n_targets, n_networks)
        after scan+vmap in the training loop. We extract for replicate 0, average over batches.
        """
        metrics = {"step": step, "progress": step / max(self._total_steps, 1)}
        metrics["loss"] = _to_scalar(step_history.get("loss"))

        # Helper to extract per-network metric with correct 4D indexing
        # Shape: (n_replicates, batches_per_step, n_targets, n_networks)
        def extract_4d(arr, tid, nid, use_mean=True):
            arr = np.asarray(arr)
            if arr.ndim == 4:
                # (n_replicates, batches_per_step, n_targets, n_networks)
                if use_mean:
                    return float(np.nanmean(arr[0, :, tid, nid]))
                return float(arr[0, -1, tid, nid])  # last batch
            elif arr.ndim == 3:
                # Fallback: (batches_per_step, n_targets, n_networks) or (n_reps, n_tgts, n_nets)
                if use_mean:
                    return float(np.nanmean(arr[:, tid, nid]))
                return float(arr[-1, tid, nid])
            elif arr.ndim == 2:
                # (n_targets, n_networks)
                return float(arr[tid, nid])
            return float('nan')

        all_losses = step_history.get("all_losses")
        if all_losses is not None:
            try:
                metrics["network_loss"] = extract_4d(all_losses, target_id, network_id)
            except (IndexError, ValueError):
                metrics["network_loss"] = float('nan')

        # Extract per-network metrics from sublosses
        sublosses = step_history.get("sublosses", {})
        for key in ["sinkhorn", "lncc", "mse", "spectral"]:
            pn_key = f"{key}_per_network"
            if pn_key in sublosses:
                try:
                    metrics[key] = extract_4d(sublosses[pn_key], target_id, network_id)
                except (IndexError, ValueError):
                    pass

        # Extract per-network metrics from tu_stats
        tu_stats = step_history.get("tu_stats", {})
        for key in [
            "enabled_count",
            "mean_prob",
            "max_log_alpha",
            "min_log_alpha",
            "std_log_alpha",
        ]:
            pn_key = f"{key}_per_network"
            if pn_key in tu_stats:
                try:
                    metrics[f"tu_{key}"] = extract_4d(tu_stats[pn_key], target_id, network_id)
                except (IndexError, ValueError):
                    pass

        # Extract per-network prediction stats
        pred_stats = step_history.get("pred_stats_per_network", {})
        for key in ["mean", "std", "min", "max"]:
            if key in pred_stats:
                try:
                    metrics[f"pred_{key}"] = extract_4d(pred_stats[key], target_id, network_id)
                except (IndexError, ValueError):
                    pass

        # Extract per-network l0_penalty
        l0_pn = step_history.get("l0_penalty_per_network")
        if l0_pn is not None:
            try:
                metrics["l0_penalty"] = extract_4d(l0_pn, target_id, network_id)
            except (IndexError, ValueError):
                pass

        # Global penalties (not per-network)
        for pname in ["tucount_penalty", "spread_penalty", "coupling_penalty", "ern_tying_penalty"]:
            val = step_history.get(pname)
            if val is not None:
                metrics[pname] = _to_scalar(val)

        return metrics

    def _append_to_history(self, target_id: int, network_id: int, metrics: dict):
        key = (target_id, network_id)
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(metrics)
        if len(self._history[key]) > self.max_history_len:
            self._history[key] = self._history[key][-self.max_history_len :]

    def _append_param_history(self, target_id: int, network_id: int, params_dict: dict):
        key = (target_id, network_id)
        if key not in self._param_history:
            self._param_history[key] = []
        self._param_history[key].append(params_dict)
        if len(self._param_history[key]) > self.max_history_len:
            self._param_history[key] = self._param_history[key][-self.max_history_len :]

    def _append_grad_history(self, target_id: int, network_id: int, grad_dict: dict):
        key = (target_id, network_id)
        if key not in self._grad_history:
            self._grad_history[key] = []
        self._grad_history[key].append(grad_dict)
        if len(self._grad_history[key]) > self.max_history_len:
            self._grad_history[key] = self._grad_history[key][-self.max_history_len :]

    def _extract_network_names(self, stack) -> list[str]:
        if stack is None or not hasattr(stack, 'networks'):
            return []
        return [getattr(n, 'name', f'network_{i}') for i, n in enumerate(stack.networks)]

    def _get_network_name(self, nid: int) -> str | None:
        return self._network_names[nid] if nid < len(self._network_names) else None

    def _build_param_metadata(self, stack) -> None:
        """Build metadata for descriptive parameter labeling from the stack."""
        try:
            ratio_meta = _build_ratio_metadata(stack)
            transform_meta = _build_transform_metadata(stack)
            self._param_metadata = ParamMetadata(
                ratio_metadata=ratio_meta,
                transform_metadata=transform_meta,
                tu_idx_to_id=self._tu_idx_to_id,
            )
            n_ratio_layers = len(ratio_meta)
            n_transform_layers = len(transform_meta)
            logger.debug(
                f"Built param metadata: {n_ratio_layers} ratio layers, {n_transform_layers} transform layers"
            )
        except Exception as e:
            logger.warning(f"Could not build param metadata: {e}")
            self._param_metadata = None

    def _try_load_stack_from_step_files(self) -> None:
        """Try to load stack from step history files (for replay mode)."""
        if self._param_metadata is not None:
            return

        import dill

        # Build candidate directories to search for step files
        candidates = []
        if self._history_dir is not None:
            candidates.append(self._history_dir)
        if self._save_dir is not None:
            candidates.extend(
                [
                    self._save_dir.parent,
                    self._save_dir.parent.parent,
                    self._save_dir.parent.parent.parent,
                ]
            )

        if not candidates:
            return

        for candidate in candidates:
            if not candidate.exists():
                continue
            step_files = sorted(candidate.glob("step_*.pkl"))[:5]
            for step_file in step_files:
                if "_start" in step_file.name or "_end" in step_file.name:
                    continue
                try:
                    with open(step_file, 'rb') as f:
                        data = dill.load(f)
                    stack = data.get('stack')
                    if stack is not None:
                        logger.info(f"Loaded stack from {step_file} for param metadata")
                        self._build_param_metadata(stack)
                        return
                except Exception as e:
                    logger.debug(f"Could not load stack from {step_file}: {e}")
                    continue

    def _render_scatter_plot(
        self,
        ax,
        X: np.ndarray,
        values: np.ndarray,
        title: str,
        cmap: str = "bc_blues",
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        import matplotlib.pyplot as plt

        try:
            import biocomp.plotting.plotting_core as _

            del _
        except ImportError:
            if cmap == "bc_blues":
                cmap = "Blues"

        values = np.asarray(values).ravel()
        X = np.asarray(X)
        valid_mask = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1]) & np.isfinite(values)
        X_valid = X[valid_mask]
        values_valid = values[valid_mask]
        if len(values_valid) == 0:
            ax.text(
                0.5, 0.5, "No valid data points", ha='center', va='center', transform=ax.transAxes
            )
            ax.set_title(title, fontsize=9)
            return
        if vmin is None:
            vmin = float(np.min(values_valid))
        if vmax is None:
            vmax = float(np.max(values_valid))
        scatter = ax.scatter(
            X_valid[:, 0],
            X_valid[:, 1],
            c=values_valid,
            cmap=cmap,
            s=25,
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("X0")
        ax.set_ylabel("X1")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        plt.colorbar(scatter, ax=ax, fraction=0.046)

    _SYMLOG_LINTHRESH = 0.1

    def _render_particle_plot(
        self,
        ax,
        data: np.ndarray,
        names: list[str],
        title: str,
        derivatives: np.ndarray | None = None,
        use_symlog: bool = True,
    ):
        try:
            from biocomp.plotting.plotting_particle import particle_plot

            particle_plot(
                ax,
                data,
                names,
                derivative=derivatives,
                value_spacing=self.particle_plot_spacing,
                max_line_extend=min(data.shape[1], self.max_history_len),
                vaxis_params={'symlog_linthresh': self._SYMLOG_LINTHRESH if use_symlog else None},
            )

            if use_symlog:
                ax.set_yscale('symlog', linthresh=self._SYMLOG_LINTHRESH)

            ax.set_title(title, fontsize=10)
        except ImportError:
            if use_symlog:
                ax.set_yscale('symlog', linthresh=self._SYMLOG_LINTHRESH)
            for i, name in enumerate(names):
                ax.plot(data[i, :], label=name[:15], alpha=0.7)
            ax.legend(fontsize=7, ncol=3, loc='upper left')
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Time")

    def _render_network_diagram(self, ax, network_id: int):
        network = None
        if (
            self._dmanager
            and hasattr(self._dmanager, 'networks')
            and network_id < len(self._dmanager.networks)
        ):
            network = self._dmanager.networks[network_id]
        elif self._networks and network_id < len(self._networks):
            network = self._networks[network_id]
        if network is None:
            ax.axis('off')
            ax.text(
                0.5, 0.5, "No network available", ha='center', va='center', transform=ax.transAxes
            )
            ax.set_title("Network Diagram", fontsize=9)
            return
        try:
            from biocomptools.toollib.figuremakers.networkdiagram import render_diagram_to_ax

            render_diagram_to_ax(network, ax, simplified=True, title="Network Diagram")
        except Exception as e:
            ax.axis('off')
            info = [f"Diagram error: {str(e)[:40]}"]
            if hasattr(network, 'name'):
                info.append(f"Network: {network.name}")
            if hasattr(network, 'compute_graph') and network.compute_graph:
                info.append(f"Nodes: {len(network.compute_graph.nodes)}")
                info.append(f"Edges: {len(network.compute_graph.edges)}")
            ax.text(
                0.5,
                0.5,
                '\n'.join(info),
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.set_title("Network Diagram", fontsize=9)

    def _render_circuit_schematic(self, ax, network_id: int):
        network = None
        if (
            self._dmanager
            and hasattr(self._dmanager, 'networks')
            and network_id < len(self._dmanager.networks)
        ):
            network = self._dmanager.networks[network_id]
        elif self._networks and network_id < len(self._networks):
            network = self._networks[network_id]
        if network is None:
            ax.axis('off')
            ax.text(
                0.5, 0.5, "No circuit available", ha='center', va='center', transform=ax.transAxes
            )
            ax.set_title("Genetic Circuit", fontsize=9)
            return
        try:
            from biocomptools.toollib.figuremakers.geneticcircuit import render_circuit_to_ax

            render_circuit_to_ax(network, ax, hide_marker_tus=True, title="Genetic Circuit")
        except Exception as e:
            ax.axis('off')
            info = [f"Circuit error: {str(e)[:40]}"]
            if hasattr(network, 'recipe') and network.recipe:
                recipe = network.recipe
                info.append(f"Recipe: {getattr(recipe, 'name', 'unnamed')}")
            ax.text(
                0.5,
                0.5,
                '\n'.join(info),
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.set_title("Genetic Circuit", fontsize=9)

    def _generate_network_figure(
        self,
        step: int,
        target_id: int,
        network_id: int,
        history: list[dict],
        param_history: list[dict],
        grad_history: list[dict],
        X: np.ndarray | None,
        Y: np.ndarray | None,
        Yhat: np.ndarray | None,
        target_name: str,
        output_path: Path,
        network_name: str | None = None,
    ):
        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            logger.warning("matplotlib not available")
            return

        # Calculate number of parameter rows needed (max 40 params per row)
        max_params_per_row = 40
        all_param_keys = set()
        if param_history:
            for ph in param_history:
                all_param_keys.update(ph.keys())
        param_keys = sorted(list(all_param_keys))
        n_params = len(param_keys)
        n_param_rows = max(1, (n_params + max_params_per_row - 1) // max_params_per_row)
        params_per_row = (n_params + n_param_rows - 1) // n_param_rows if n_param_rows > 0 else 0

        n_rows = 3 + n_param_rows
        height_ratios = [2.25, 1.2, 1] + [1] * n_param_rows
        fig_height = 28.8 + (n_param_rows - 1) * 7.2

        fig = plt.figure(figsize=(38.4, fig_height))
        gs = GridSpec(n_rows, 4, figure=fig, height_ratios=height_ratios, hspace=0.35, wspace=0.25)

        ax_diagram = fig.add_subplot(gs[0, :2])
        self._render_network_diagram(ax_diagram, network_id)

        ax_schematic = fig.add_subplot(gs[0, 2:])
        self._render_circuit_schematic(ax_schematic, network_id)

        ax_xy = fig.add_subplot(gs[1, 0])
        if X is not None and Y is not None:
            self._render_scatter_plot(ax_xy, X, Y, f"Target: {target_name}")
        else:
            ax_xy.text(
                0.5, 0.5, "No target data", ha='center', va='center', transform=ax_xy.transAxes
            )
            ax_xy.set_title(f"Target: {target_name}")

        ax_xyhat = fig.add_subplot(gs[1, 1])
        if X is not None and Yhat is not None:
            all_vals = []
            if Y is not None:
                all_vals.extend(Y[np.isfinite(Y)].tolist())
            all_vals.extend(Yhat[np.isfinite(Yhat)].tolist())
            vmin = min(all_vals) if all_vals else 0
            vmax = max(all_vals) if all_vals else 1
            self._render_scatter_plot(ax_xyhat, X, Yhat, "Prediction", vmin=vmin, vmax=vmax)
        else:
            ax_xyhat.text(
                0.5, 0.5, "No prediction", ha='center', va='center', transform=ax_xyhat.transAxes
            )
            ax_xyhat.set_title("Prediction")

        ax_loss = fig.add_subplot(gs[1, 2:])
        # Per-network loss metrics (new format)
        loss_keys = [
            "network_loss",
            "sinkhorn",
            "lncc",
            "mse",
            "spectral",
            "l0_penalty",
        ]
        available_loss = [k for k in loss_keys if any(k in h for h in history)]
        if history and available_loss:
            data, _, _ = prepare_particle_data(history, available_loss)
            self._render_particle_plot(ax_loss, data, available_loss, "Per-Network Losses")
        else:
            ax_loss.text(
                0.5, 0.5, "No loss history", ha='center', va='center', transform=ax_loss.transAxes
            )
            ax_loss.set_title("Per-Network Losses")

        ax_stats = fig.add_subplot(gs[2, :])
        # Per-network TU stats and prediction stats (new format)
        tu_keys = [
            "tu_enabled_count",
            "tu_mean_prob",
            "tu_max_log_alpha",
            "tu_min_log_alpha",
            "tu_std_log_alpha",
        ]
        pred_keys = ["pred_mean", "pred_std", "pred_min", "pred_max"]
        penalty_keys = [
            "spread_penalty",
            "coupling_penalty",
            "tucount_penalty",
            "ern_tying_penalty",
        ]
        combined_stats = tu_keys + pred_keys + penalty_keys
        available_stats = [k for k in combined_stats if any(k in h for h in history)]
        stats_display = {k: k.replace('tu_', '').replace('pred_', 'ŷ_') for k in available_stats}
        if history and available_stats:
            data, _, _ = prepare_particle_data(history, available_stats)
            names = [stats_display.get(k, k) for k in available_stats]
            self._render_particle_plot(ax_stats, data, names, "Per-Network Stats", use_symlog=True)
        else:
            ax_stats.text(
                0.5, 0.5, "No stats", ha='center', va='center', transform=ax_stats.transAxes
            )
            ax_stats.set_title("Stats")

        # Render parameter rows (evenly distributed)
        for row_idx in range(n_param_rows):
            ax_params = fig.add_subplot(gs[3 + row_idx, :])
            start_idx = row_idx * params_per_row
            end_idx = min(start_idx + params_per_row, n_params)
            row_param_keys = param_keys[start_idx:end_idx]

            if param_history and row_param_keys:
                data, names, derivatives = prepare_particle_data(
                    param_history,
                    row_param_keys,
                    derivative_history=grad_history if grad_history else None,
                )
                title = f"Parameters [{start_idx}-{end_idx - 1}]"
                if derivatives is not None and np.any(np.abs(derivatives) > 1e-10):
                    title += " + Gradients"
                self._render_particle_plot(
                    ax_params, data, names, title, derivatives=derivatives, use_symlog=True
                )
            else:
                ax_params.text(
                    0.5,
                    0.5,
                    "No param history" if not param_history else "No param keys",
                    ha='center',
                    va='center',
                    transform=ax_params.transAxes,
                )
                ax_params.set_title(f"Parameters [{start_idx}-{end_idx - 1}]")

        if history:
            steps = [h.get('step', 0) for h in history]
            step_range = (
                f"batches {min(steps)}-{max(steps)}" if len(steps) > 1 else f"batch {steps[0]}"
            )
        else:
            step_range = f"batch {step}"

        title = f"Step {step} ({step_range})"
        net_display = network_name or f"network_{network_id}"
        subtitle = f"Target: {target_name}  |  Network {network_id}: {net_display}"

        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.995)
        fig.text(0.5, 0.975, subtitle, ha='center', fontsize=10, style='italic')

        if self.file_format == 'pdf':
            diagfig_path = output_path.with_suffix('.pdf')
        else:
            diagfig_path = output_path.with_suffix('.png')
        plt.savefig(diagfig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved diagnostic figure to {diagfig_path}")

    def _save_summary_json(self, step: int, step_history: dict, output_path: Path):
        summary = {
            "step": step,
            "progress": step / max(self._total_steps, 1),
            "loss": _to_scalar(step_history.get("loss")),
        }
        for key in ["tu_stats", "ratio_stats", "sublosses"]:
            val = step_history.get(key, {})
            if isinstance(val, dict):
                for k, v in val.items():
                    summary[f"{key}.{k}"] = _to_scalar(v)
        for pname in ["l0_penalty", "spread_penalty", "coupling_penalty"]:
            val = step_history.get(pname)
            if val is not None:
                summary[pname] = _to_scalar(val)
        output_path.write_text(json.dumps(summary, indent=2, default=str))

    def get_metrics(self, replicate: int | None = None) -> dict | None:
        if not self._history:
            return None
        return {
            "targets_networks_tracked": len(self._history),
            "total_entries": sum(len(v) for v in self._history.values()),
        }

    def finalize(self):
        total_entries = sum(len(v) for v in self._history.values())
        logger.info(
            f"DesignDiagnosticLogger finalized with {total_entries} entries "
            f"across {len(self._history)} target/network pairs"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # New pattern methods (on_batch / on_end)
    # ──────────────────────────────────────────────────────────────────────────

    def _process_view_data(self, view: HistoryView) -> dict:
        """Extract step_history-like dict from HistoryView for figure generation."""
        latest = view.latest()
        if latest is None:
            return {}
        result = {'loss': latest.loss}
        result.update(latest.metrics)
        result.update(latest.arrays)
        return result

    def _get_dims(self, step_history: dict) -> tuple[int, int]:
        """Extract n_targets, n_networks from step_history."""
        yhatdep = _squeeze_to_3d(step_history.get("yhatdep"))
        all_losses = step_history.get("all_losses")
        n_targets, n_networks = 1, 1
        if yhatdep is not None and yhatdep.ndim == 3:
            _, n_targets, n_networks = yhatdep.shape
        elif all_losses is not None:
            arr = np.asarray(all_losses)
            if arr.ndim >= 3:
                n_targets = arr.shape[1]
                n_networks = arr.shape[2]
        return n_targets, n_networks

    def _accumulate_history(self, step: int, step_history: dict, n_targets: int, n_networks: int):
        """Accumulate metrics/params/grads into internal history."""
        if self._param_metadata is None:
            self._try_load_stack_from_step_files()

        params = step_history.get("latest_params") or step_history.get("params")
        grad = step_history.get("grad")
        for tid in range(min(n_targets, self.max_targets_to_plot)):
            selected_nets = self._get_selected_networks(tid, n_networks, step_history)
            for nid in selected_nets:
                metrics = self._extract_metrics(step, step_history, tid, nid)
                self._append_to_history(tid, nid, metrics)
                if params is not None:
                    self._append_param_history(
                        tid,
                        nid,
                        unroll_params(
                            params, 0, tid, nid, self._tu_idx_to_id, self._param_metadata
                        ),
                    )
                if grad is not None:
                    self._append_grad_history(
                        tid,
                        nid,
                        unroll_grads(grad, 0, tid, nid, self._tu_idx_to_id, self._param_metadata),
                    )

    def _build_history_from_view(self, view: HistoryView, target_id: int, network_id: int):
        """Build per-target/network history from HistoryView."""
        if self._param_metadata is None:
            self._try_load_stack_from_step_files()

        for batch in view.iter_batches():
            step = batch.step_index
            step_history = {'loss': batch.loss}
            step_history.update(batch.metrics)
            step_history.update(batch.arrays)

            metrics = self._extract_metrics(step, step_history, target_id, network_id)
            self._append_to_history(target_id, network_id, metrics)

            params = step_history.get("latest_params") or step_history.get("params")
            grad = step_history.get("grad")
            if params is not None:
                self._append_param_history(
                    target_id,
                    network_id,
                    unroll_params(
                        params, 0, target_id, network_id, self._tu_idx_to_id, self._param_metadata
                    ),
                )
            if grad is not None:
                self._append_grad_history(
                    target_id,
                    network_id,
                    unroll_grads(
                        grad, 0, target_id, network_id, self._tu_idx_to_id, self._param_metadata
                    ),
                )

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        """Process a batch using new pattern with centralized history."""
        step = context.current_step
        step_history = self._process_view_data(view)
        if not step_history:
            return

        n_targets, n_networks = self._get_dims(step_history)
        self._accumulate_history(step, step_history, n_targets, n_networks)

        if self.generate_plots and step > 0 and not self.final_figure_only:
            step_dir = self._save_dir / f"step_{step:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            self._save_summary_json(step, step_history, step_dir / "summary.json")
            targets = self._dmanager.targets if self._dmanager else []
            self._generate_figures_for_step(
                step, step_history, targets, n_targets, n_networks, step_dir
            )

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        step = context.current_step
        step_history = self._process_view_data(view)
        n_targets, n_networks = self._get_dims(step_history) if step_history else (1, 1)

        if not self._history and view.n_batches > 0:
            for tid in range(min(n_targets, self.max_targets_to_plot)):
                selected_nets = self._get_selected_networks(tid, n_networks, step_history)
                for nid in selected_nets:
                    self._build_history_from_view(view, tid, nid)

        final_dir = self._save_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        if self.generate_plots and step_history:
            targets = self._dmanager.targets if self._dmanager else []
            self._generate_figures_for_step(
                step, step_history, targets, n_targets, n_networks, final_dir
            )

        if self.save_history:
            import dill as pickle

            with open(final_dir / "full_history.pickle", 'wb') as f:
                pickle.dump(
                    {
                        "metrics_history": dict(self._history),
                        "param_history": dict(self._param_history),
                        "grad_history": dict(self._grad_history),
                    },
                    f,
                )
            if step_history:
                self._save_summary_json(step, step_history, final_dir / "summary.json")

    def _get_figure_output_path(
        self,
        step: int,
        tid: int,
        nid: int,
        target_name: str,
        step_output_dir: Path,
    ) -> Path:
        """Get output path for a figure based on output_structure setting.

        Args:
            step: Current step number
            tid: Target index
            nid: Network index
            target_name: Name of the target
            step_output_dir: Base output dir for this step (used in step_first mode)

        Returns:
            Path where the figure should be saved
        """
        safe_target = target_name.replace(' ', '_').replace('/', '_')
        network_name = self._get_network_name(nid) or f"network_{nid}"
        safe_network = network_name.replace(' ', '_').replace('/', '_')

        if self.output_structure == "network_first":
            output_dir = self._save_dir / safe_network / safe_target
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / f"design_diagnostic_step_{step:06d}.pdf"
        else:
            target_dir = step_output_dir / f"target_{tid}_{safe_target}"
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir / f"network_{nid}.pdf"

    def _generate_figures_for_step(
        self,
        step: int,
        step_history: dict,
        targets: list,
        n_targets: int,
        n_networks: int,
        output_dir: Path,
    ):
        yhatdep = _squeeze_to_3d(step_history.get("yhatdep"))
        X_hist = _squeeze_to_3d(step_history.get("X"))
        Y_hist = _squeeze_to_3d(step_history.get("Y"))

        for tid in range(min(n_targets, self.max_targets_to_plot)):
            target = targets[tid] if tid < len(targets) else None
            target_name = getattr(target, 'name', f'target_{tid}') if target else f'target_{tid}'
            selected_nets = self._get_selected_networks(tid, n_networks, step_history)

            for nid in selected_nets:
                history = self._history.get((tid, nid), [])
                param_history = self._param_history.get((tid, nid), [])
                grad_history = self._grad_history.get((tid, nid), [])

                X, Y, Yhat = None, None, None
                if X_hist is not None and X_hist.ndim == 3:
                    n_inputs = 2
                    X = X_hist[:, tid, nid * n_inputs : (nid + 1) * n_inputs]
                if Y_hist is not None and Y_hist.ndim == 3:
                    Y = Y_hist[:, tid, 0]
                if yhatdep is not None and yhatdep.ndim == 3:
                    Yhat = yhatdep[:, tid, nid]

                output_path = self._get_figure_output_path(step, tid, nid, target_name, output_dir)

                try:
                    self._generate_network_figure(
                        step,
                        tid,
                        nid,
                        history,
                        param_history,
                        grad_history,
                        X,
                        Y,
                        Yhat,
                        target_name,
                        output_path,
                        network_name=self._get_network_name(nid),
                    )
                except Exception as e:
                    logger.error(f"DiagnosticLogger: failed to generate figure: {e}", exc_info=True)
