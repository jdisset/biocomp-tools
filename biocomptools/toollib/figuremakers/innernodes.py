from __future__ import annotations

from biocomptools.toollib.plot import Figure, PlotConfig, load_default_plotconf
from biocomptools.modelmodel import BiocompModel, NetworkModel, load_model
from biocomptools.logging_config import get_logger
from biocomp.plotutils import FigureSpec
from biocomp.plotting.plotting_core import knn_stats, DEFAULT_CMAP_NAME, build_tree
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit as Unit
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext, load_lib
import biocomp.biorules as br
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure as MplFigure
from typing import Annotated, Any, Callable
from pydantic import Field, BeforeValidator
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

logger = get_logger(__name__)

N_SAMPLES = 100_000
SHOW_INVERSE = True
CMAP_TRUNCATE_MIN = 0.4


@dataclass(frozen=True)
class ApplyFn:
    single: Callable[..., float]
    batch: Callable[..., np.ndarray]

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return self.single(*args, **kwargs)


@dataclass(frozen=True)
class NodeInfo:
    name: str
    node_type: str
    apply_fn: ApplyFn
    node_id: int
    emb_name: str | None = None
    emb_val: float | None = None


class InnerNodesFigure(Figure):
    model: Annotated[BiocompModel, BeforeValidator(load_model)]
    n_samples: int = N_SAMPLES
    figure_spec: FigureSpec = Field(default_factory=FigureSpec)
    plot_config: PlotConfig = Field(default_factory=load_default_plotconf)
    print_summary: bool = True

    _stack_cache: dict = {}

    def _get_cmap(self, truncate: bool = True):
        base = plt.get_cmap(DEFAULT_CMAP_NAME)
        lo = CMAP_TRUNCATE_MIN if truncate else 0
        return LinearSegmentedColormap.from_list("cmap", base(np.linspace(lo, 1, 256)))

    def _extract_emb(self, path: str, names: list[str]) -> dict[str, float]:
        try:
            vals: Any = self.model.shared_params
            for key in path.split('/'):
                vals = vals[key]
            return {n: float(v[0]) for n, v in zip(names, vals)}
        except (KeyError, IndexError, TypeError):
            return {}

    def _build_stack(self, networks: list) -> tuple[Any, Any, Any]:
        cache_key = tuple(id(n) for n in networks)
        if cache_key not in self._stack_cache:
            net_model = NetworkModel(model=self.model, network=networks)
            self._stack_cache[cache_key] = (net_model.stack, net_model.params)
        stack, params = self._stack_cache[cache_key]
        return stack, params, jax.random.PRNGKey(0)

    def _make_apply(self, f_apply: Callable, params: Any, key: Any) -> ApplyFn:
        def apply_single(*inputs, node_id: int, random_var: float = 0.5):
            inputs_jnp = tuple(jnp.atleast_1d(jnp.asarray(x)) for x in inputs)
            random_vars = jnp.atleast_1d(jnp.asarray(random_var))
            result, _ = f_apply(
                *inputs_jnp,
                random_vars=random_vars,
                params=params,
                node_id=jnp.array(node_id),
                key=key,
            )
            return float(jnp.squeeze(result))

        def apply_batch(
            inputs_arr: np.ndarray, node_id: int, random_var: float = 0.5
        ) -> np.ndarray:
            def single_eval(*inputs):
                inputs_jnp = tuple(jnp.atleast_1d(x) for x in inputs)
                random_vars = jnp.atleast_1d(jnp.asarray(random_var))
                result, _ = f_apply(
                    *inputs_jnp,
                    random_vars=random_vars,
                    params=params,
                    node_id=jnp.array(node_id),
                    key=key,
                )
                return jnp.squeeze(result)

            if inputs_arr.ndim == 1:
                inputs_arr = inputs_arr.reshape(-1, 1)
            inputs_list = [inputs_arr[:, i] for i in range(inputs_arr.shape[1])]
            vmapped = jax.vmap(single_eval)
            return np.asarray(vmapped(*inputs_list))

        return ApplyFn(single=apply_single, batch=apply_batch)

    def _get_layer_by_type(self, stack, type_name: str):
        return next((layer for layer in stack.layers if type_name in layer.type_str()), None)

    def _build_ern_probes(self) -> list[NodeInfo]:
        cc = self.model.compute_config
        if not cc or not cc.node_functions or "sequestron_ERN" not in cc.node_functions:
            return []

        affinity_names = cc.node_functions["sequestron_ERN"].kwargs.get("affinity_names", [])
        ern_names = [n.split("::")[1].split("#")[0] for n in affinity_names]
        if not ern_names:
            return []

        ern_vals = self._extract_emb("shared/ERN_5p/affinities", ern_names)

        networks = []
        with LibraryContext.with_library(load_lib()):
            for name in ern_names:
                nets = recipe_to_networks(
                    Recipe(
                        content=[
                            CoTransfection(
                                units=[Unit(slots=["hEF1a", name]), Unit(slots=["hEF1a", "mKO2"])]
                            ),
                            CoTransfection(
                                units=[
                                    Unit(slots=["hEF1a", f"{name}_rec", "eYFP"]),
                                    Unit(slots=["hEF1a", "eBFP2"]),
                                ]
                            ),
                        ]
                    ),
                    br.ALL_RULES,
                    invert=True,
                    inversion_mode="all",
                )
                if nets:
                    networks.append(nets[0])

        if not networks:
            return []

        stack, params, key = self._build_stack(networks)
        ern_layer = self._get_layer_by_type(stack, "ERN")
        if not ern_layer:
            return []

        apply_fn = self._make_apply(ern_layer.f_apply, params, key)
        return [
            NodeInfo(
                name=name,
                node_type="ERN",
                apply_fn=apply_fn,
                node_id=i,
                emb_name="affinity",
                emb_val=ern_vals.get(name),
            )
            for i, name in enumerate(ern_names)
        ]

    def _build_uorf_probes(self) -> list[NodeInfo]:
        cc = self.model.compute_config
        if not cc or not cc.node_functions:
            return []

        try:
            uorf_raw = cc.node_functions["translation"].kwargs["quantization_names"]
        except (KeyError, AttributeError):
            return []

        uorf_vals = self._extract_emb("shared/quantization/values/tl_rate", uorf_raw)
        uorf_clean = [
            (r.replace("_uORF", "") if r != "00_empty_tc" else "none", r) for r in uorf_raw
        ]

        networks, name_map = [], {}
        with LibraryContext.with_library(load_lib()):
            for clean, raw in uorf_clean:
                nets = recipe_to_networks(
                    Recipe(
                        content=[
                            CoTransfection(
                                units=[
                                    Unit(slots=["hEF1a", "eBFP2"]),
                                    Unit(slots=["hEF1a", raw, "mKO2"]),
                                ]
                            )
                        ]
                    ),
                    br.ALL_RULES,
                    invert=True,
                    inversion_mode="all",
                )
                if nets:
                    name_map[len(networks)] = (clean, raw, uorf_vals.get(raw))
                    networks.append(nets[0])

        if not networks:
            return []

        stack, params, key = self._build_stack(networks)
        tl_layer = next(
            (
                layer
                for layer in stack.layers
                if layer.type_str() == "translation" and layer.f_apply
            ),
            None,
        )
        if not tl_layer:
            return []

        apply_fn = self._make_apply(tl_layer.f_apply, params, key)

        def find_node_pos(net_idx: int, target_uorf: str) -> int | None:
            for vnode in tl_layer.nodes:
                if vnode.network_id != net_idx:
                    continue
                net = networks[vnode.network_id]
                for e in net.compute_graph.get_incoming_edges(vnode.node_id):
                    if e.content_embedding_names.get("tl_rate") != (target_uorf,):
                        continue
                    content_names = [p.name for p in e.content] if e.content else []
                    if "mKO2" in content_names or target_uorf in content_names:
                        return vnode.node_position_in_layer
            return None

        result = []
        for net_idx, (clean, raw, emb) in name_map.items():
            node_pos = find_node_pos(net_idx, raw)
            if node_pos is not None:
                result.append(
                    NodeInfo(
                        name=clean,
                        node_type="Translation",
                        apply_fn=apply_fn,
                        node_id=node_pos,
                        emb_name="tl_rate",
                        emb_val=emb,
                    )
                )
        return result

    def _build_inv_uorf_probes(self) -> list[NodeInfo]:
        cc = self.model.compute_config
        if not cc or not cc.node_functions:
            return []

        try:
            uorf_raw = cc.node_functions["translation"].kwargs["quantization_names"]
        except (KeyError, AttributeError):
            return []

        uorf_vals = self._extract_emb("shared/quantization/values/tl_rate", uorf_raw)
        uorf_clean = [
            (r.replace("_uORF", "") if r != "00_empty_tc" else "none", r) for r in uorf_raw
        ]

        networks, name_map = [], {}
        with LibraryContext.with_library(load_lib()):
            for clean, raw in uorf_clean:
                nets = recipe_to_networks(
                    Recipe(
                        content=[
                            CoTransfection(
                                units=[
                                    Unit(slots=["hEF1a", "eBFP2"]),
                                    Unit(slots=["hEF1a", raw, "mKO2"]),
                                ]
                            )
                        ]
                    ),
                    br.ALL_RULES,
                    invert=True,
                    inversion_mode="all",
                )
                if nets:
                    name_map[len(networks)] = (clean, raw, uorf_vals.get(raw))
                    networks.append(nets[0])

        if not networks:
            return []

        stack, params, key = self._build_stack(networks)
        inv_tl_layer = next(
            (
                layer
                for layer in stack.layers
                if layer.type_str() == "inv_translation" and layer.f_apply
            ),
            None,
        )
        if not inv_tl_layer:
            return []

        # Force open quantization mask for inverse translation nodes
        # Each node should have only its specific uORF enabled (not all-open)
        inv_namespace = stack.get_layer_namespace(inv_tl_layer.layer_id)
        mask_path = f"{inv_namespace}/tl_rate_quantization_mask"
        n_uorfs = len(uorf_raw)

        # Build node->uorf mapping
        node_to_uorf_idx = {}
        for vnode in inv_tl_layer.nodes:
            net_idx = vnode.network_id
            if net_idx in name_map:
                _, raw, _ = name_map[net_idx]
                uorf_idx = uorf_raw.index(raw) if raw in uorf_raw else 0
                node_to_uorf_idx[vnode.node_position_in_layer] = uorf_idx

        # Create new masks where each node has only its target uORF enabled
        n_nodes = len(inv_tl_layer.nodes)
        n_inputs = 1  # inverse translation has 1 input
        new_masks = np.zeros((n_nodes, n_inputs, n_uorfs), dtype=bool)
        for node_id, uorf_idx in node_to_uorf_idx.items():
            new_masks[node_id, :, uorf_idx] = True

        # Override the mask in params (replace the view with actual values)
        params[mask_path] = new_masks

        apply_fn = self._make_apply(inv_tl_layer.f_apply, params, key)

        result = []
        for vnode in inv_tl_layer.nodes:
            net_idx = vnode.network_id
            if net_idx not in name_map:
                continue
            clean, raw, emb = name_map[net_idx]
            result.append(
                NodeInfo(
                    name=clean,
                    node_type="Inv Translation",
                    apply_fn=apply_fn,
                    node_id=vnode.node_position_in_layer,
                    emb_name="tl_rate",
                    emb_val=emb,
                )
            )
        return result

    def _build_source_probes(self) -> list[NodeInfo]:
        # Use same-plasmid sources (source="p0") to get a single source node with multiple outputs
        with LibraryContext.with_library(load_lib()):
            nets = recipe_to_networks(
                Recipe(
                    content=[
                        CoTransfection(
                            units=[
                                Unit(slots=["hEF1a", "eYFP"], source="p0"),
                                Unit(slots=["hEF1a", "eBFP2"], source="p0"),
                            ]
                        )
                    ]
                ),
                br.ALL_RULES,
                invert=True,
                inversion_mode="all",
            )

        if not nets:
            return []

        stack, params, key = self._build_stack([nets[0]])
        src_layer = next(
            (layer for layer in stack.layers if layer.type_str() == "source" and layer.f_apply),
            None,
        )
        if not src_layer:
            return []

        # With same-plasmid source, we have 1 node with n_outputs=2
        # We need to create probes that extract each output position
        apply_fn = self._make_apply(src_layer.f_apply, params, key)

        # Check if source has multiple outputs per node
        n_outputs = len(src_layer.f_out_shapes)
        if n_outputs > 1:
            # Multi-output source: create a probe for each output position
            return [
                NodeInfo(
                    name=f"pos {i}",
                    node_type="Source",
                    apply_fn=apply_fn,
                    node_id=0,  # single node
                    emb_name="position",
                    emb_val=float(i),
                )
                for i in range(n_outputs)
            ]
        else:
            # Single output per node: create a probe for each node
            return [
                NodeInfo(
                    name=f"pos {i}",
                    node_type="Source",
                    apply_fn=apply_fn,
                    node_id=i,
                )
                for i in range(len(src_layer.nodes))
            ]

    def _build_basic_probes(self) -> tuple[list[NodeInfo], list[NodeInfo]]:
        with LibraryContext.with_library(load_lib()):
            nets = recipe_to_networks(
                Recipe(
                    content=[
                        CoTransfection(
                            units=[Unit(slots=["hEF1a", "eYFP"]), Unit(slots=["hEF1a", "eBFP2"])]
                        )
                    ]
                ),
                br.ALL_RULES,
                invert=True,
                inversion_mode="all",
            )

        if not nets:
            return [], []

        stack, params, key = self._build_stack([nets[0]])

        forward, inverse = [], []
        fwd_types = [
            ("source", "plasmid → DNA"),
            ("transcription", "DNA → mRNA"),
            ("translation", "mRNA → PRT"),
        ]
        inv_types = [
            ("inv_source", "DNA → plasmid"),
            ("inv_transcription", "mRNA → DNA"),
            ("inv_translation", "Fluo → mRNA"),
        ]

        for layer in stack.layers:
            if not layer.f_apply:
                continue
            type_str = layer.type_str()
            apply_fn = self._make_apply(layer.f_apply, params, key)

            for node_type, label in fwd_types:
                if type_str == node_type:
                    forward.append(
                        NodeInfo(
                            name=label, node_type=node_type.title(), apply_fn=apply_fn, node_id=0
                        )
                    )
                    break

            if SHOW_INVERSE:
                for node_type, label in inv_types:
                    if type_str == node_type:
                        inverse.append(
                            NodeInfo(
                                name=label,
                                node_type=node_type.replace("_", " ").title(),
                                apply_fn=apply_fn,
                                node_id=0,
                            )
                        )
                        break

        return forward, inverse

    def _eval_node(self, node: NodeInfo, inputs: np.ndarray, random_var: float = 0.5) -> np.ndarray:
        result = node.apply_fn.batch(inputs, node_id=node.node_id, random_var=random_var)
        # Handle multi-output nodes (like source with multiple positions)
        # If emb_name is "position", extract the specific output index from emb_val
        if node.emb_name == "position" and node.emb_val is not None and result.ndim == 2:
            output_idx = int(node.emb_val)
            return result[:, output_idx]
        return result

    def _eval_1d(self, node: NodeInfo, x_range: np.ndarray, random_var: float = 0.5) -> np.ndarray:
        return node.apply_fn.batch(
            x_range.reshape(-1, 1), node_id=node.node_id, random_var=random_var
        )

    def _plot_scatter(
        self,
        ax,
        node: NodeInfo,
        inputs: np.ndarray,
        outputs: np.ndarray,
        cmap,
        size: int = 30000,
        cbar: bool = False,
    ):
        ax.set_box_aspect(1)
        idx = (
            np.random.choice(len(inputs), min(size, len(inputs)), replace=False)
            if len(inputs) > size
            else slice(None)
        )

        if inputs.shape[1] == 2:
            sc = ax.scatter(
                inputs[idx, 0],
                inputs[idx, 1],
                c=outputs[idx],
                cmap=cmap,
                s=10,
                alpha=1,
                linewidths=0,
            )
            ax.set(xlabel="Input 1", ylabel="Input 2")
            if cbar:
                plt.colorbar(sc, ax=ax, shrink=0.8).set_label("Output")
        else:
            ax.scatter(inputs[idx, 0], outputs[idx], s=4, alpha=0.05, linewidth=0, color=cmap(0.7))
            xq = np.linspace(inputs.min(), inputs.max(), 200).reshape(-1, 1)
            z = knn_stats(
                xq,
                outputs.reshape(-1, 1),
                tree=build_tree(inputs),
                stats=["mean"],
                k=min(500, len(inputs) // 2),
            )
            if z is not None:
                ax.plot(xq, z, linewidth=1.5, color='black', linestyle='dashed')
            ax.set(xlabel="Input", ylabel="Output")

    def _plot_ern_row(self, subfig, ern_nodes: list[NodeInfo]):
        if not ern_nodes:
            return

        subfig.suptitle("ERN Nodes (2D)", fontsize=16, fontweight='bold', y=1.05)

        resolution = 100
        x = np.linspace(0.01, 0.8, resolution)
        y = np.linspace(0.01, 0.8, resolution)
        xx, yy = np.meshgrid(x, y)
        grid_inputs = np.column_stack([xx.ravel(), yy.ravel()])
        cmap = self._get_cmap(truncate=False)

        n_ern = min(4, len(ern_nodes))
        axes = subfig.subplots(1, n_ern + 1, width_ratios=[1] * n_ern + [0.05], gridspec_kw={'wspace': 0.3})

        vmin, vmax = None, None
        for node in ern_nodes[:n_ern]:
            outputs = self._eval_node(node, grid_inputs)
            if vmin is None:
                vmin, vmax = outputs.min(), outputs.max()
            else:
                vmin, vmax = min(vmin, outputs.min()), max(vmax, outputs.max())

        for i, node in enumerate(ern_nodes[:n_ern]):
            outputs = self._eval_node(node, grid_inputs).reshape(resolution, resolution)
            im = axes[i].imshow(
                outputs,
                extent=[x[0], x[-1], y[0], y[-1]],
                origin='lower',
                aspect='equal',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axes[i].set(xlabel="ERN Protein Amount", ylabel="mRNA Target Amount")
            axes[i].set_title(f"ERN\n({node.name})", fontweight='bold', fontsize=12)

        cbar = plt.colorbar(im, cax=axes[-1])
        cbar.set_label("Output")

    def _plot_ern_1d_row(self, subfig, ern_nodes: list[NodeInfo]):
        if not ern_nodes:
            return

        subfig.suptitle("ERN Repression Curves", fontsize=16, fontweight='bold', y=1.05)
        axes = subfig.subplots(1, 3, width_ratios=[1, 1, 1.1], gridspec_kw={'wspace': 0.3})

        neg_range = np.linspace(0.01, 0.8, 100)
        pos_slices = [0.25, 0.75]
        linestyles = ['-', '--']
        random_var = 0.5
        cmap = self._get_cmap(truncate=True)
        colors = {n.name: cmap(i / max(1, len(ern_nodes) - 1)) for i, n in enumerate(ern_nodes)}

        # Plot 1: Output vs Repressor with 2 pos slices
        ax = axes[0]
        for node in ern_nodes:
            for j, pos in enumerate(pos_slices):
                outputs = np.array([
                    node.apply_fn(neg, pos, node_id=node.node_id, random_var=random_var)
                    for neg in neg_range
                ])
                label = node.name if j == 0 else None
                ax.plot(neg_range, outputs, color=colors[node.name], linewidth=2.5,
                        label=label, linestyle=linestyles[j])
        # add legend entries for line styles
        ax.plot([], [], color='gray', linestyle='-', label=f'pos={pos_slices[0]}')
        ax.plot([], [], color='gray', linestyle='--', label=f'pos={pos_slices[1]}')
        ax.set(xlabel="ERN Protein Amount", ylabel="Output")
        ax.set_title("Output vs ERN Protein", fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 2: Relative Repression with 2 pos slices
        ax = axes[1]
        for node in ern_nodes:
            for j, pos in enumerate(pos_slices):
                baseline = node.apply_fn(0.01, pos, node_id=node.node_id, random_var=random_var)
                outputs = np.array([
                    node.apply_fn(neg, pos, node_id=node.node_id, random_var=random_var) / baseline
                    for neg in neg_range
                ])
                label = node.name if j == 0 else None
                ax.plot(neg_range, outputs, color=colors[node.name], linewidth=2.5,
                        label=label, linestyle=linestyles[j])
        ax.plot([], [], color='gray', linestyle='-', label=f'pos={pos_slices[0]}')
        ax.plot([], [], color='gray', linestyle='--', label=f'pos={pos_slices[1]}')
        ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
        ax.set(xlabel="ERN Protein Amount", ylabel="Output / Baseline", ylim=(0.3, 1.05))
        ax.set_title("Relative Repression", fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.3)

        # Plot 3: ERN embedding heatmap (use middle of pos slices)
        pos_heatmap = sum(pos_slices) / len(pos_slices)
        ax = axes[2]
        emb_vals = np.array([n.emb_val or 0 for n in ern_nodes])
        emb_range = np.linspace(emb_vals.min() - 0.1, emb_vals.max() + 0.1, 100)
        neg_range_2d = np.linspace(0.01, 0.8, 100)

        # build heatmap by interpolating between ERN nodes based on embedding
        heatmap = np.zeros((len(emb_range), len(neg_range_2d)))
        for j, neg in enumerate(neg_range_2d):
            node_outputs = np.array([
                node.apply_fn(neg, pos_heatmap, node_id=node.node_id, random_var=random_var)
                for node in ern_nodes
            ])
            heatmap[:, j] = np.interp(emb_range, np.sort(emb_vals), node_outputs[np.argsort(emb_vals)])

        im = ax.imshow(
            heatmap,
            extent=[neg_range_2d[0], neg_range_2d[-1], emb_range[0], emb_range[-1]],
            origin='lower',
            aspect='auto',
            cmap=self._get_cmap(truncate=False),
        )
        # horizontal dashed lines at each ERN embedding value
        for node in ern_nodes:
            if node.emb_val is not None:
                ax.axhline(node.emb_val, color='black', linestyle='--', alpha=0.7, linewidth=1)
                ax.text(neg_range_2d[-1] * 0.5, node.emb_val, node.name, color='black',
                        fontsize=9, ha='center', va='center', fontweight='bold')
        ax.set(xlabel="ERN Protein Amount", ylabel="ERN Embedding")
        ax.set_title(f"Output vs ERN Embedding (pos={pos_heatmap})", fontweight='bold', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Output")

    def _plot_translation_heatmap(self, ax, uorf_nodes: list[NodeInfo]):
        """Plot 2D translation heatmap: x=input, y=uORF embedding, color=output."""
        if not uorf_nodes:
            return

        # extract embedding values and sort nodes by embedding
        sorted_nodes = sorted(uorf_nodes, key=lambda n: n.emb_val or 0)
        emb_vals = np.array([n.emb_val or 0 for n in sorted_nodes])

        input_range = np.linspace(0.01, 0.8, 100)
        emb_range = np.linspace(emb_vals.min() - 0.1, emb_vals.max() + 0.1, 100)

        # build heatmap by interpolating between uORF nodes based on embedding
        heatmap = np.zeros((len(emb_range), len(input_range)))
        for j, inp in enumerate(input_range):
            node_outputs = np.array([
                node.apply_fn(inp, node_id=node.node_id, random_var=0.5)
                for node in sorted_nodes
            ])
            # linear interpolation over embedding values
            heatmap[:, j] = np.interp(emb_range, emb_vals, node_outputs)

        im = ax.imshow(
            heatmap,
            extent=[input_range[0], input_range[-1], emb_range[0], emb_range[-1]],
            origin='lower',
            aspect='auto',
            cmap=self._get_cmap(truncate=False),
        )
        # horizontal dashed lines at each uORF embedding value
        for node in sorted_nodes:
            if node.emb_val is not None:
                ax.axhline(node.emb_val, color='black', linestyle='--', alpha=0.7, linewidth=1)
                ax.text(input_range[-1] * 0.5, node.emb_val, node.name, color='black',
                        fontsize=9, ha='center', va='center', fontweight='bold')
        ax.set(xlabel="Input (mRNA)", ylabel="uORF Embedding")
        ax.set_title("Translation\n(input vs uORF emb)", fontweight='bold', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Output (PRT)")

    def _plot_multi_curve(
        self, ax, nodes: list[NodeInfo], inputs_1d: np.ndarray, cmap, title: str, legend_title: str
    ):
        step = max(1, len(nodes) // 8)
        for j, node in enumerate(nodes[::step]):
            col = cmap(j / max(1, len(nodes[::step]) - 1))
            outputs = self._eval_node(node, inputs_1d)
            ax.scatter(inputs_1d[:, 0], outputs, s=4, alpha=0.01, linewidth=0, color=col)
            xq = np.linspace(inputs_1d.min(), inputs_1d.max(), 200).reshape(-1, 1)
            z = knn_stats(
                xq,
                outputs.reshape(-1, 1),
                tree=build_tree(inputs_1d),
                stats=["mean"],
                k=min(500, len(inputs_1d) // 2),
            )
            if z is not None:
                ax.plot(xq, z, linewidth=1.5, color='white', linestyle='solid')
                ax.plot(xq, z, linewidth=1.5, color=col, linestyle='dashed')
            ax.plot([], [], color=col, label=node.name, linewidth=2)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.legend(title=legend_title, loc="upper left", fontsize='x-small', frameon=False)
        ax.set(xlabel="Input", ylabel="Output")
        ax.set_box_aspect(1)

    def _plot_forward_row(
        self,
        subfig,
        basic_nodes: list[NodeInfo],
        uorf_nodes: list[NodeInfo],
        source_nodes: list[NodeInfo],
    ):
        if not basic_nodes and not uorf_nodes and not source_nodes:
            return

        subfig.suptitle("Forward Nodes", fontsize=16, fontweight='bold', y=1.05)

        np.random.seed(42)
        inputs_1d = np.random.uniform(0.01, 0.8, (self.n_samples, 1))
        cmap = self._get_cmap(truncate=True)

        has_source = len(source_nodes) > 1
        has_uorf = bool(uorf_nodes)
        # +1 for translation heatmap if we have uorf_nodes
        n_plots = len(basic_nodes) + (1 if has_source else 0) + (2 if has_uorf else 0)
        axes = subfig.subplots(1, n_plots, gridspec_kw={'wspace': 0.3})
        if n_plots == 1:
            axes = [axes]

        ax_idx = 0
        if has_source:
            self._plot_multi_curve(
                axes[ax_idx], source_nodes, inputs_1d, cmap, "Source\nplasmid → DNA", "position"
            )
            ax_idx += 1

        for node in basic_nodes:
            outputs = self._eval_node(node, inputs_1d)
            self._plot_scatter(axes[ax_idx], node, inputs_1d, outputs, cmap)
            axes[ax_idx].set_title(f"{node.node_type}\n{node.name}", fontweight='bold', fontsize=12)
            ax_idx += 1

        if has_uorf:
            self._plot_multi_curve(
                axes[ax_idx], uorf_nodes, inputs_1d, cmap, "Translation\nmRNA → PRT", "uORFs"
            )
            ax_idx += 1
            # translation heatmap
            self._plot_translation_heatmap(axes[ax_idx], uorf_nodes)

    def _plot_inverse_row(
        self, subfig, inverse_nodes: list[NodeInfo], inv_uorf_nodes: list[NodeInfo]
    ):
        if not inverse_nodes and not inv_uorf_nodes:
            return

        subfig.suptitle("Inverse Nodes", fontsize=16, fontweight='bold', y=1.05)

        np.random.seed(42)
        inputs_1d = np.random.uniform(0.01, 0.8, (self.n_samples, 1))
        cmap = self._get_cmap(truncate=True)

        has_inv_uorf = bool(inv_uorf_nodes)
        n_plots = len(inverse_nodes) + (1 if has_inv_uorf else 0)
        axes = subfig.subplots(1, n_plots, gridspec_kw={'wspace': 0.3})
        if n_plots == 1:
            axes = [axes]

        ax_idx = 0
        if has_inv_uorf:
            self._plot_multi_curve(
                axes[ax_idx],
                inv_uorf_nodes,
                inputs_1d,
                cmap,
                "Inv Translation\nFluo → mRNA",
                "uORFs",
            )
            ax_idx += 1

        for node in inverse_nodes:
            outputs = self._eval_node(node, inputs_1d)
            self._plot_scatter(axes[ax_idx], node, inputs_1d, outputs, cmap)
            axes[ax_idx].set_title(f"{node.node_type}\n{node.name}", fontweight='bold', fontsize=12)
            ax_idx += 1

    def create_figure(self) -> MplFigure:
        ern_nodes = sorted(self._build_ern_probes(), key=lambda n: n.emb_val or 0, reverse=True)
        uorf_nodes = self._build_uorf_probes()
        inv_uorf_nodes = self._build_inv_uorf_probes()
        source_nodes = self._build_source_probes()
        basic_nodes, inverse_nodes = self._build_basic_probes()

        if uorf_nodes:
            basic_nodes = [n for n in basic_nodes if n.node_type != "Translation"]
        if len(source_nodes) > 1:
            basic_nodes = [n for n in basic_nodes if n.node_type != "Source"]
        if inv_uorf_nodes:
            inverse_nodes = [n for n in inverse_nodes if n.node_type != "Inv Translation"]

        rows = []
        if ern_nodes:
            rows.append(("ern", ern_nodes))
            rows.append(("ern_1d", ern_nodes))
        if basic_nodes or uorf_nodes or source_nodes:
            rows.append(("forward", (basic_nodes, uorf_nodes, source_nodes)))
        if (inverse_nodes or inv_uorf_nodes) and SHOW_INVERSE:
            rows.append(("inverse", (inverse_nodes, inv_uorf_nodes)))

        if not rows:
            fig = plt.figure(figsize=(10, 5))
            fig.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=16)
            return fig

        fig = plt.figure(figsize=(20, 5 * len(rows)))
        subfigs = fig.subfigures(len(rows), 1, hspace=0.1)
        if len(rows) == 1:
            subfigs = [subfigs]

        for subfig, (row_type, data) in zip(subfigs, rows):
            if row_type == "ern":
                self._plot_ern_row(subfig, data)
            elif row_type == "ern_1d":
                self._plot_ern_1d_row(subfig, data)
            elif row_type == "forward":
                self._plot_forward_row(subfig, *data)
            elif row_type == "inverse":
                self._plot_inverse_row(subfig, *data)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
        return fig

    def run(self, overwrite: bool = True, finalize: bool = True):
        self.figure_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = self.create_figure()
        fig.savefig(self.figure_spec.output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


InnerNodesFigureSpec = type("InnerNodesFigureSpec", (FigureSpec,), {})
