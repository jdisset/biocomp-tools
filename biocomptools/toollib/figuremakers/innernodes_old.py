from __future__ import annotations

from biocomptools.toollib.plot import Figure, PlotConfig, load_default_plotconf
from biocomptools.modelmodel import BiocompModel, NetworkModel, NodeSpec, load_model
from biocomptools.toollib.networkprediction import NetworkPrediction, reconstruct_from_flat
from biocomptools.logging_config import get_logger
from biocomp.plotutils import FigureSpec, PlotData
from biocomp.plotting.plotting_core import knn_stats, DEFAULT_CMAP_NAME, build_tree
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit as Unit
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext, load_lib
import biocomp.biorules as br
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure as MplFigure
from typing import Annotated, Any
from pydantic import Field, BeforeValidator
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

logger = get_logger(__name__)

N_SAMPLES = 100_000
SHOW_INVERSE = True
MINCOLOR = 0
CMAP_TRUNCATE_MIN = 0.4


@dataclass(frozen=True)
class NodeInfo:
    name: str
    type: str
    data: PlotData
    emb_name: str | None
    emb_val: float | None


class InnerNodesFigure(Figure):
    model: Annotated[BiocompModel, BeforeValidator(load_model)]
    n_samples: int = N_SAMPLES
    figure_spec: FigureSpec = Field(default_factory=FigureSpec)
    plot_config: PlotConfig = Field(default_factory=load_default_plotconf)
    uorf_names: list[str] | None = None
    print_summary: bool = True
    print_only: bool = False

    def _extract_emb(self, path: str, names: list[str]) -> dict[str, float]:
        try:
            vals: Any = self.model.shared_params
            for key in path.split('/'):
                vals = vals[key]
            return {n: float(v[0]) for n, v in zip(names, vals)}
        except (KeyError, IndexError, TypeError):
            return {}

    def _build_ern_probe_stack(self) -> tuple[Any, Any, list[str]]:
        """Build a compute stack with ERN probe networks and return (apply_fn, params, ern_names)."""
        from biocomptools.modelmodel import NetworkModel

        cc = self.model.compute_config
        if cc is None or cc.node_functions is None:
            return None, None, []

        affinity_names = cc.node_functions["sequestron_ERN"].kwargs.get("affinity_names", [])
        ern_names = [n.split("::")[1].split("#")[0] for n in affinity_names]
        if not ern_names:
            return None, None, []

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
            return None, None, []

        net_model = NetworkModel(model=self.model, network=networks)
        ern_layer = next(
            (layer for layer in net_model.stack.layers if "ERN" in layer.type_str()), None
        )
        if ern_layer is None:
            return None, None, []

        return ern_layer.f_apply, net_model.params, ern_names

    def _create_ern_forward_fn(self):
        """Create ERN forward function using the actual ERN layer's apply function."""
        import jax
        import jax.numpy as jnp

        ern_apply, params, ern_names = self._build_ern_probe_stack()
        if ern_apply is None:
            return None

        key = jax.random.PRNGKey(0)
        self._ern_names = ern_names
        self._ern_params = params

        def ern_forward(neg, pos, node_id, random_var):
            neg_arr = jnp.atleast_1d(jnp.asarray(neg))
            pos_arr = jnp.atleast_1d(jnp.asarray(pos))
            random_vars = jnp.atleast_1d(jnp.asarray(random_var))
            result, _ = ern_apply(
                neg_arr,
                pos_arr,
                random_vars=random_vars,
                params=params,
                node_id=jnp.array(node_id),
                key=key,
            )
            return float(result[0])

        return ern_forward

    def _get_ern_data_standalone(self) -> list[NodeInfo]:
        """Get ERN node data using the actual ERN layer's apply function."""
        ern_fn = self._create_ern_forward_fn()
        if ern_fn is None:
            return []

        ern_names = getattr(self, '_ern_names', [])
        if not ern_names:
            return []

        ern_vals = self._extract_emb("shared/ERN_5p/affinities", ern_names)

        np.random.seed(42)
        inputs = np.random.uniform(0.01, 0.8, (self.n_samples, 2))
        random_var = 0.5

        results = []
        for node_id, name in enumerate(ern_names):
            outputs = np.array(
                [
                    ern_fn(inputs[i, 0], inputs[i, 1], node_id, random_var)
                    for i in range(self.n_samples)
                ]
            )

            plot_data = PlotData(
                xval=inputs,
                yval=outputs.reshape(-1, 1),
                metadata={"input_shapes": [(1,), (1,)], "output_shapes": [(1,)]},
            )

            results.append(
                NodeInfo(
                    name=name,
                    type="ERN",
                    data=plot_data,
                    emb_name="affinity",
                    emb_val=ern_vals.get(name, 0.5),
                )
            )

        return results

    def _plot_ern_1d_summary(self, subfig, ern_nodes: list[NodeInfo], cmap):
        if not ern_nodes:
            return

        subfig.suptitle("ERN Repression Curves", fontsize=16, fontweight='bold', y=1.05)
        ern_fn = self._create_ern_forward_fn()
        if ern_fn is None:
            return

        ern_names = [n.name for n in ern_nodes]
        name_to_node_id = {name: i for i, name in enumerate(ern_names)}
        neg_range = np.linspace(0.01, 0.8, 100)
        pos, random_var = 0.5, 0.5
        trunc_cmap = LinearSegmentedColormap.from_list(
            "trunc", cmap(np.linspace(CMAP_TRUNCATE_MIN, 1, 256))
        )
        colors = {
            name: trunc_cmap(i / max(1, len(ern_names) - 1)) for i, name in enumerate(ern_names)
        }
        axes = subfig.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.3})

        ax = axes[0]
        for name in ern_names:
            node_id = name_to_node_id[name]
            outputs = np.array([ern_fn(neg, pos, node_id, random_var) for neg in neg_range])
            ax.plot(neg_range, outputs, label=name, color=colors[name], linewidth=2.5)
        ax.set(
            xlabel="Repressor Amount", ylabel="Output", title=f"Output vs Repressor\n(pos={pos})"
        )
        ax.set_title(ax.get_title(), fontweight='bold', fontsize=12)
        ax.legend(title="ERN Type", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for name in ern_names:
            node_id = name_to_node_id[name]
            baseline = ern_fn(0.01, pos, node_id, random_var)
            outputs = np.array(
                [ern_fn(neg, pos, node_id, random_var) / baseline for neg in neg_range]
            )
            ax.plot(neg_range, outputs, label=name, color=colors[name], linewidth=2.5)
        ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
        ax.set(xlabel="Repressor Amount", ylabel="Output / Baseline", ylim=(0.3, 1.05))
        ax.set_title(
            f"Relative Repression (pos={pos})\n(lower = stronger)", fontweight='bold', fontsize=12
        )
        ax.legend(title="ERN Type", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        neg_levels = [0.2, 0.4, 0.6]
        neg_colors = [trunc_cmap(i / (len(neg_levels) - 1)) for i in range(len(neg_levels))]
        x_positions = np.arange(len(ern_names))
        width = 0.25
        for i, (neg, col) in enumerate(zip(neg_levels, neg_colors)):
            outputs = [ern_fn(neg, pos, name_to_node_id[name], random_var) for name in ern_names]
            ax.bar(
                x_positions + i * width, outputs, width, label=f'neg={neg}', color=col, alpha=0.85
            )
        ax.set_xticks(x_positions + width)
        ax.set_xticklabels(ern_names)
        ax.set_ylabel("Output", fontsize=11)
        ax.set_title(f"ERN Output by Type (pos={pos})", fontweight='bold', fontsize=12)
        ax.legend(title="neg", fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    def _get_ern_probes(self):
        try:
            cc = self.model.compute_config
            if cc is None or cc.node_functions is None:
                return
            ern_names = [
                n.split("::")[1].split("#")[0]
                for n in cc.node_functions["sequestron_ERN"].kwargs["affinity_names"]
            ]
            ern_vals = self._extract_emb("shared/ERN_5p/affinities", ern_names)
        except (KeyError, AttributeError, TypeError):
            return

        with LibraryContext.with_library(load_lib()):
            for name in ern_names:
                all_nets = recipe_to_networks(
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
                net = all_nets[0] if all_nets else None
                if net and net.compute_graph:
                    ern_nodes = net.compute_graph.get_nodes_by_type("sequestron_ERN")
                    if ern_nodes:
                        print('found ERN node for', name)
                        print(ern_nodes)
                        yield (
                            net,
                            ("ERN", name, "affinity", ern_vals.get(name)),
                            ern_nodes[0].node_id,
                        )

    def _get_uorf_probes(self):
        """Generate probes for uORF/translation nodes."""
        try:
            cc = self.model.compute_config
            if cc is None or cc.node_functions is None:
                raise KeyError("No compute config")
            uorf_raw = cc.node_functions["translation"].kwargs["quantization_names"]
        except (KeyError, AttributeError, TypeError):
            uorf_raw = self.uorf_names if self.uorf_names else []
        if not uorf_raw:
            return
        uorf_vals = self._extract_emb("shared/quantization/values/tl_rate", uorf_raw)
        uorf_clean = [(r.strip("_uORF") if r != "00_empty_tc" else "none", r) for r in uorf_raw]

        with LibraryContext.with_library(load_lib()):
            for clean, raw in uorf_clean:
                all_nets = recipe_to_networks(
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
                if not all_nets:
                    continue
                # select the network where the non-uorf tu (eBFP2) is inverted
                net = None
                for candidate in all_nets:
                    if not candidate.compute_graph:
                        continue
                    inv_trans = [
                        n
                        for n in candidate.compute_graph.nodes.values()
                        if n.node_type == "inv_translation"
                    ]
                    if inv_trans:
                        inv_node = inv_trans[0]
                        if inv_node.is_inverse_of is None:
                            continue
                        orig_edges = candidate.compute_graph.get_incoming_edges(
                            inv_node.is_inverse_of.node_id
                        )
                        for edge in orig_edges:
                            if 'tl_rate' in edge.content_embedding_names:
                                if '00_empty_tc' in edge.content_embedding_names['tl_rate']:
                                    net = candidate
                                    break
                        if net:
                            break
                if net is None:
                    net = all_nets[0]

                if not net.compute_graph:
                    continue

                tlnodes = net.compute_graph.get_nodes_by_type("translation")
                if not tlnodes:
                    continue
                target_tlnode = None
                for tlnode in tlnodes:
                    edges = net.compute_graph.get_incoming_edges(tlnode.node_id)
                    for edge in edges:
                        if raw in edge.content_embedding_names.get('tl_rate', ()):
                            target_tlnode = tlnode
                            break
                    if target_tlnode:
                        break
                if target_tlnode is None:
                    target_tlnode = tlnodes[0]

                yield (
                    net,
                    ("Translation", clean, "tl_rate", uorf_vals.get(raw)),
                    target_tlnode.node_id,
                )

    def _get_basic_probes(self, include_translation: bool = True):
        """Generate probes for basic nodes (source, transcription, output, inverse)."""
        with LibraryContext.with_library(load_lib()):
            basic_nets = recipe_to_networks(
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
            if not basic_nets:
                return
            basic = basic_nets[0]
            g = basic.compute_graph
            if not g:
                return
            basic_node_types = [
                ("source", "plasmid → DNA"),
                ("transcription", "DNA → mRNA"),
                ("output", "PRT → fluo"),
            ]
            if include_translation:
                basic_node_types.append(("translation", "mRNA → PRT"))
            for t, label in basic_node_types:
                nodes = g.get_nodes_by_type(t)
                if nodes:
                    yield basic, (t.title(), label, None, None), nodes[0].node_id
            if SHOW_INVERSE:
                inv_map = {
                    "inv_source": "DNA → plasmid",
                    "inv_transcription": "mRNA → DNA",
                    "inv_translation": "Fluo → mRNA",
                }
                for nid, node in g.nodes.items():
                    if node.is_inverse_of and node.node_type in inv_map:
                        yield (
                            basic,
                            (
                                node.node_type.replace("_", " ").title(),
                                inv_map[node.node_type],
                                None,
                                None,
                            ),
                            nid,
                        )

    def _get_data(self, probes) -> list[NodeInfo]:
        """Convert probes to NodeInfo list via NetworkPrediction."""
        networks, specs, inputs, net_map = [], [], [], {}
        np.random.seed(42)
        shared_inputs = {
            'ERN': np.random.uniform(0.01, 0.8, (self.n_samples, 2)),
            'Translation': np.random.uniform(0.01, 0.8, (self.n_samples, 1)),
            'default': np.random.uniform(0.01, 0.8, (self.n_samples, 1)),
        }

        for net, (typ, name, emb_name, emb_val), nid in probes:
            if id(net) not in net_map:
                net_map[id(net)] = len(networks)
                networks.append(net)
                input_key = typ if typ in shared_inputs else 'default'
                inputs.append(shared_inputs[input_key].copy())
                print(
                    f"Added network {len(networks) - 1} for type {typ} with input shape {inputs[-1].shape}"
                )

            specs.append(
                NodeSpec(
                    node_id=nid,
                    network_id=net_map[id(net)],
                    extra_info={
                        "type": typ,
                        "name": name,
                        "emb_name": emb_name,
                        "emb_val": emb_val,
                    },
                )
            )

        if not networks:
            return []

        pred = NetworkPrediction(
            predict_at=inputs,
            network_model=NetworkModel(model=self.model, network=networks),
            collection_points=specs,
            disable_variational=True,
            z_value="uniform",
            already_latent=True,
        )
        return [
            NodeInfo(
                name=pd.metadata["collection_point_nodespec"].extra_info["name"],
                type=pd.metadata["collection_point_nodespec"].extra_info["type"],
                data=pd,
                emb_name=pd.metadata["collection_point_nodespec"].extra_info["emb_name"],
                emb_val=pd.metadata["collection_point_nodespec"].extra_info["emb_val"],
            )
            for pd in pred.get_data()
        ]

    def _smart_scatter(
        self, ax, node: NodeInfo, size=30000, cbar=False, trend=True, full_cmap=False, **kw
    ):
        x, y = node.data.x, node.data.y
        base_cmap = plt.get_cmap(DEFAULT_CMAP_NAME)
        cmap_min = MINCOLOR if full_cmap else CMAP_TRUNCATE_MIN
        cmap = LinearSegmentedColormap.from_list("trunc", base_cmap(np.linspace(cmap_min, 1, 256)))
        ax.set_box_aspect(1)

        if "input_shapes" in node.data.metadata:
            x_list = reconstruct_from_flat(x, node.data.metadata["input_shapes"])
            y_list = reconstruct_from_flat(y, node.data.metadata["output_shapes"])
            if len(x_list) > 1 and len(y_list) == 1:
                x_2d = np.column_stack(x_list)
                idx = (
                    np.random.choice(len(x_2d), min(size, len(x_2d)), replace=False)
                    if len(x_2d) > size
                    else slice(None)
                )
                sc = ax.scatter(
                    x_2d[idx, 0],
                    x_2d[idx, 1],
                    c=y_list[0][idx].flatten(),
                    cmap=cmap,
                    s=10,
                    alpha=1,
                    linewidths=0,
                )
                ax.set(xlabel="Protein Amount", ylabel="mRNA Amount")
                if cbar:
                    plt.colorbar(
                        sc, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                    ).set_label("Output", fontsize=10)
                return
            pairs_x = [x_list[0]] * len(y_list) if len(y_list) > 1 and len(x_list) == 1 else x_list
            pairs_y = (
                y_list
                if len(y_list) > 1
                else [y_list[0]] * len(x_list)
                if len(x_list) > 1
                else y_list
            )
            labels = (
                [f"out {i}" for i in range(len(pairs_y))]
                if len(pairs_y) > 1
                else ([f"in {i}" for i in range(len(pairs_x))] if len(pairs_x) > 1 else [""])
            )
            colors = base_cmap(np.linspace(MINCOLOR, 1.0, len(pairs_x)))
        else:
            pairs_x, pairs_y = [x.reshape(-1, 1)], [y.reshape(-1, 1)]
            labels, colors = [""], [kw.get('color', 'blue')]

        n_lines = len(pairs_x)
        use_markers = kw.get('markers', (1 < n_lines < 4))
        s_alpha = kw.get('alpha', (0.01 if n_lines > 1 else 0.05))

        for j, (X, Y, col, lbl) in enumerate(zip(pairs_x, pairs_y, colors, labels)):
            X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
            idx = (
                np.random.choice(len(X), min(size, len(X)), replace=False)
                if len(X) > size
                else slice(None)
            )
            ax.scatter(X[idx].flatten(), Y[idx].flatten(), s=4, alpha=s_alpha, linewidth=0, c=col)
            if trend:
                xq = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
                z = knn_stats(xq, Y, tree=build_tree(X), stats=["mean"], k=min(500, len(X) // 2))
                if z is None:
                    continue
                trend_lw = kw.get('lw', 1)
                trend_col = kw.get('trend_color', 'black' if n_lines == 1 else col)
                # white contour behind the dashed line
                if kw.get('trend_contour', False):
                    ax.plot(xq, z, linewidth=trend_lw + 2, color='white', linestyle='solid')
                ax.plot(
                    xq,
                    z,
                    linewidth=trend_lw,
                    color=trend_col,
                    linestyle='dashed',
                )
                if use_markers:
                    mk_idx = np.arange(0, 200, 20) + int((j / n_lines) * 100)
                    mk_idx = mk_idx[mk_idx < len(z)]
                    ax.plot(
                        xq[mk_idx],
                        z[mk_idx],
                        marker=["^", "s", "X", "v"][j % 4],
                        markersize=7,
                        color='black',
                        linestyle='None',
                        label=lbl,
                        markerfacecolor='none',
                    )
                elif lbl and n_lines > 1:
                    ax.plot([], [], color=col, label=lbl, linewidth=2)

        if any(labels) and n_lines > 1:
            ax.legend(loc="upper left", fontsize='x-small', frameon=False)
        ax.set(xlabel="Input (latent)", ylabel="Output (latent)")

    def _safe_get_data(self, probes_gen, section_name: str) -> list[NodeInfo]:
        """Safely get data for a section, returning empty list on error."""
        try:
            probes = list(probes_gen) if probes_gen else []
            if not probes:
                return []
            return self._get_data(probes)
        except Exception as e:
            logger.warning(f"Skipping {section_name} section due to error: {e}")
            return []

    def _compute_translation_summary(self, uorf_nodes: list[NodeInfo]) -> dict:
        """Compute summary statistics for translation nodes with different uORFs.

        Returns dict with:
        - baseline_name: name of baseline uORF (none/empty_tc)
        - comparisons: list of (name, pct_diff, mean_output) relative to baseline
        - embedding_range: (min, max) of embedding values
        - output_range: (min_mean, max_mean) of mean outputs
        """
        if not uorf_nodes:
            return {}

        # compute mean output for each uORF (averaged over input range)
        means = {}
        emb_vals = {}
        for node in uorf_nodes:
            y = node.data.y.flatten()
            means[node.name] = float(np.mean(y))
            if node.emb_val is not None:
                emb_vals[node.name] = node.emb_val

        # find baseline (none or empty_tc)
        baseline_name = None
        for candidate in ["none", "None", "empty", "00_empty_tc"]:
            if candidate in means:
                baseline_name = candidate
                break
        if baseline_name is None and means:
            # use the one with highest mean as baseline (likely "none")
            baseline_name = max(means, key=lambda k: means[k])

        if baseline_name is None:
            return {}

        baseline_mean = means[baseline_name]

        # compute percent differences from baseline
        comparisons = []
        for name, mean in sorted(means.items(), key=lambda x: x[1], reverse=True):
            if name == baseline_name:
                continue
            pct_diff = ((mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
            comparisons.append((name, pct_diff, mean, emb_vals.get(name)))

        return {
            "baseline_name": baseline_name,
            "baseline_mean": baseline_mean,
            "baseline_emb": emb_vals.get(baseline_name),
            "comparisons": comparisons,
            "embedding_range": (min(emb_vals.values()), max(emb_vals.values()))
            if emb_vals
            else None,
            "output_range": (min(means.values()), max(means.values())),
        }

    def _print_summary(self, uorf_summary: dict, ern_nodes: list[NodeInfo] | None = None):
        """Print human-readable summary of node behaviors."""
        lines = ["\n" + "=" * 60, "INNER NODES SUMMARY", "=" * 60]

        if uorf_summary:
            lines.append("\nTranslation Node (uORF effects):")
            lines.append(
                f"  Baseline: {uorf_summary['baseline_name']} "
                f"(mean output: {uorf_summary['baseline_mean']:.4f})"
            )
            if uorf_summary.get('baseline_emb') is not None:
                lines.append(f"  Baseline embedding value: {uorf_summary['baseline_emb']:.4f}")

            lines.append("\n  Relative to baseline:")
            for name, pct_diff, mean, emb in uorf_summary["comparisons"]:
                sign = "+" if pct_diff > 0 else ""
                emb_str = f" (emb={emb:.3f})" if emb is not None else ""
                lines.append(f"    {name:12s}: {sign}{pct_diff:6.1f}% (mean={mean:.4f}){emb_str}")

            if uorf_summary.get("embedding_range"):
                emin, emax = uorf_summary["embedding_range"]
                lines.append(f"\n  Embedding value range: [{emin:.4f}, {emax:.4f}]")
            omin, omax = uorf_summary["output_range"]
            lines.append(f"  Output mean range: [{omin:.4f}, {omax:.4f}]")
            dynamic_range = (omax - omin) / omax * 100 if omax != 0 else 0
            lines.append(f"  Dynamic range: {dynamic_range:.1f}%")

        if ern_nodes:
            lines.append("\nERN Nodes (affinity embeddings):")
            sorted_ern = sorted(ern_nodes, key=lambda n: n.emb_val or 0, reverse=True)
            for node in sorted_ern:
                emb_str = f"{node.emb_val:.4f}" if node.emb_val is not None else "N/A"
                lines.append(f"    {node.name:12s}: embedding={emb_str}")

        lines.append("=" * 60 + "\n")
        print("\n".join(lines))

    def _fetch_all_data(
        self,
    ) -> tuple[list[NodeInfo], list[NodeInfo], list[NodeInfo], list[NodeInfo]]:
        """Fetch all node data for ERN, uORF, basic, and inverse sections."""
        # Use standalone ERN evaluation to avoid network graph issues
        ern = sorted(
            self._get_ern_data_standalone(),
            key=lambda n: n.emb_val or 0,
            reverse=True,
        )
        uorf = self._safe_get_data(self._get_uorf_probes(), "uORF")
        basic_data = self._safe_get_data(
            self._get_basic_probes(include_translation=not uorf), "basic"
        )
        basic = [
            n for n in basic_data if n.type in ["Source", "Transcription", "Output", "Translation"]
        ]
        inverse = [n for n in basic_data if "Inv" in n.type]
        return ern, uorf, basic, inverse

    def print_summary_only(self) -> None:
        """Print summary without generating figure."""
        ern, uorf, _, _ = self._fetch_all_data()
        if uorf:
            uorf_summary = self._compute_translation_summary(uorf)
            self._print_summary(uorf_summary, ern if ern else None)
        elif ern:
            self._print_summary({}, ern)
        else:
            print("No uORF or ERN data available for summary.")

    def create_innernodes_figure(self) -> MplFigure:
        ern, uorf, basic, inverse = self._fetch_all_data()

        # compute and print summary if enabled
        if self.print_summary and uorf:
            uorf_summary = self._compute_translation_summary(uorf)
            self._print_summary(uorf_summary, ern if ern else None)

        base_cmap = plt.get_cmap(DEFAULT_CMAP_NAME)

        # determine which rows to show
        show_ern = bool(ern)
        show_ern_1d = show_ern  # Show 1D ERN summary plots alongside scatter
        show_forward = bool(basic) or bool(uorf)
        show_inverse = bool(inverse) and SHOW_INVERSE
        n_rows = sum([show_ern, show_ern_1d, show_forward, show_inverse])

        if n_rows == 0:
            logger.warning("No data available for inner nodes figure")
            fig = plt.figure(figsize=(10, 5))
            fig.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=16)
            return fig

        height_ratios = [1] * n_rows
        fig = plt.figure(figsize=(20, 5 * n_rows))
        subfigs = fig.subfigures(n_rows, 1, height_ratios=height_ratios, hspace=0.1)
        rows: list[Any] = [subfigs] if n_rows == 1 else list(subfigs)

        row_idx = 0

        if show_ern:
            rows[row_idx].suptitle(
                "ERN Nodes and Embeddings", fontsize=16, fontweight='bold', y=1.05
            )
            cols = rows[row_idx].subfigures(1, 2, width_ratios=[3, 1])
            n_ern = min(4, len(ern))
            axes = cols[0].subplots(
                1, n_ern, gridspec_kw={'width_ratios': [1] * n_ern, 'wspace': 0.3}
            )
            if n_ern == 1:
                axes = [axes]
            for i, node in enumerate(ern[:n_ern]):
                print(f"Plotting ERN node: {node.name} with emb {node.emb_val}, {node.data}")
                self._smart_scatter(
                    axes[i], node, size=100000, cbar=(i == n_ern - 1), full_cmap=True
                )
                axes[i].set_title(f"ERN\n({node.name})", fontweight='bold', fontsize=12)
            emb_ax = cols[1].subplots(1, 1)
            names, vals = zip(*[(n.name, n.emb_val) for n in ern])
            trunc_cmap = LinearSegmentedColormap.from_list(
                "trunc", base_cmap(np.linspace(CMAP_TRUNCATE_MIN, 1, 256))
            )
            emb_ax.barh(
                np.arange(len(names)), vals, color=trunc_cmap(np.linspace(0, 1, len(names)))
            )
            emb_ax.set(yticks=np.arange(len(names)), yticklabels=names, title="ERN Embeddings")
            emb_ax.invert_yaxis()
            emb_ax.set_box_aspect(2)
            row_idx += 1

        if show_ern_1d:
            self._plot_ern_1d_summary(rows[row_idx], ern, base_cmap)
            row_idx += 1

        if show_forward:
            rows[row_idx].suptitle("Forward Nodes", fontsize=16, fontweight='bold', y=1.05)
            n_fwd = len(basic) + (1 if uorf else 0)
            if n_fwd > 0:
                axes = rows[row_idx].subplots(
                    1, n_fwd, gridspec_kw={'width_ratios': [1] * n_fwd, 'wspace': 0.3}
                )
                if n_fwd == 1:
                    axes = [axes]
                for ax, node in zip(axes, basic):
                    self._smart_scatter(ax, node)
                    ax.set_title(f"{node.type}\n{node.name}", fontweight='bold', fontsize=12)
                if uorf:
                    step = max(1, len(uorf) // 8)
                    trunc_cmap = LinearSegmentedColormap.from_list(
                        "trunc", base_cmap(np.linspace(CMAP_TRUNCATE_MIN, 1, 256))
                    )
                    for node, col in zip(
                        uorf[::step], trunc_cmap(np.linspace(0, 1, len(uorf[::step])))
                    ):
                        self._smart_scatter(
                            axes[-1],
                            node,
                            alpha=0.01,
                            trend=True,
                            color=col,
                            trend_color=col,
                            lw=1.5,
                            markers=False,
                            trend_contour=True,
                        )
                        axes[-1].plot([], [], color=col, label=node.name, linewidth=2)
                    axes[-1].set_title("Translation\nmRNA → PRT", fontweight='bold', fontsize=12)
                    axes[-1].legend(
                        title="uORFs", loc="upper left", fontsize='x-small', frameon=False
                    )
            row_idx += 1

        if show_inverse:
            rows[row_idx].suptitle("Inverse Nodes", fontsize=16, fontweight='bold', y=1.05)
            axes = rows[row_idx].subplots(
                1, len(inverse), gridspec_kw={'width_ratios': [1] * len(inverse), 'wspace': 0.3}
            )
            if len(inverse) == 1:
                axes = [axes]
            for ax, node in zip(axes, inverse):
                self._smart_scatter(ax, node)
                ax.set_title(f"{node.type}\n{node.name}", fontweight='bold', fontsize=12)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
        return fig

    def run(self, overwrite: bool = True, finalize: bool = True):
        if self.print_only:
            self.print_summary_only()
            return
        self.figure_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = self.create_innernodes_figure()
        fig.savefig(self.figure_spec.output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


InnerNodesFigureSpec = type('InnerNodesFigureSpec', (FigureSpec,), {})
