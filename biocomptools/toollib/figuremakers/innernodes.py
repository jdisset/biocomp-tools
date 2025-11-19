from biocomptools.toollib.plot import Figure, PlotConfig, load_default_plotconf
from biocomptools.modelmodel import BiocompModel, NetworkModel, NodeSpec, load_model
from biocomptools.toollib.networkprediction import NetworkPrediction, reconstruct_from_flat
from biocomptools.logging_config import get_logger
from biocomp.plotutils import FigureSpec, FigureLayout, FigAx, PlotData
from biocomp.plotting.plotting_core import knn_stats, DEFAULT_CMAP_NAME, build_tree
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit as Unit, Slot
from biocomp.network import recipe_to_networks, Network
from biocomp.library import LibraryContext, load_lib
import biocomp.biorules as br
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from typing import Annotated
from pydantic import Field, BeforeValidator
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

logger = get_logger(__name__)

N_SAMPLES = 100_000
SHOW_INVERSE = True


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

    def _extract_emb(self, path: str, names: list[str]) -> dict[str, float]:
        try:
            vals = self.model.shared_params
            for key in path.split('/'):
                vals = vals[key]
            return {n: float(v[0]) for n, v in zip(names, vals)}
        except (KeyError, IndexError):
            return {}

    def _get_probes(self):
        try:
            ern_names = [
                n.split("::")[1].split("#")[0]
                for n in self.model.compute_config.node_functions["sequestron_ERN"].kwargs[
                    "affinity_names"
                ]
            ]
            ern_vals = self._extract_emb("shared/ERN_5p/affinities", ern_names)
        except (KeyError, AttributeError):
            ern_names, ern_vals = [], {}
        uorf_raw = self.model.compute_config.node_functions["translation"].kwargs[
            "quantization_names"
        ]
        uorf_vals = self._extract_emb("shared/quantization/values/tl_rate", uorf_raw)
        uorf_clean = [(r.strip("_uORF") if r != "00_empty_tc" else "none", r) for r in uorf_raw]

        with LibraryContext.with_library(load_lib()):
            for name in ern_names:
                net = recipe_to_networks(
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
                )[0]
                yield (
                    net,
                    ("ERN", name, "affinity", ern_vals.get(name)),
                    net.compute_graph.get_nodes_by_type("sequestron_ERN")[0].node_id,
                )
            for clean, raw in uorf_clean:
                print(clean, raw)
                net = recipe_to_networks(
                    Recipe(
                        content=[
                            CoTransfection(
                                units=[
                                    Unit(slots=["hEF1a", raw, "mKO2"]),
                                    Unit(slots=["hEF1a", "eBFP2"]),
                                ]
                            )
                        ]
                    ),
                    br.ALL_RULES,
                    invert=True,
                )[0]
                tlnodes = net.compute_graph.get_nodes_by_type("translation")
                yield (
                    net,
                    ("Translation", clean, "tl_rate", uorf_vals.get(raw)),
                    tlnodes[0].node_id,
                )
            basic = recipe_to_networks(
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
            )[0]
            g = basic.compute_graph
            for t, label in [
                ("source", "plasmid → DNA"),
                ("transcription", "DNA → mRNA"),
                ("output", "PRT → fluo"),
            ]:
                yield basic, (t.title(), label, None, None), g.get_nodes_by_type(t)[0].node_id
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

    def _get_data(self) -> list[NodeInfo]:
        networks, specs, inputs, net_map = [], [], [], {}
        for net, (typ, name, emb_name, emb_val), nid in self._get_probes():
            if id(net) not in net_map:
                net_map[id(net)] = len(networks)
                networks.append(net)
                inputs.append(
                    np.random.uniform(0.01, 0.8, (self.n_samples, 2 if typ == "ERN" else 1))
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

    def _smart_scatter(self, ax, node: NodeInfo, size=30000, cbar=False, trend=True, **kw):
        x, y = node.data.x, node.data.y
        base_cmap = plt.get_cmap(DEFAULT_CMAP_NAME)
        cmap = LinearSegmentedColormap.from_list("trunc", base_cmap(np.linspace(0.4, 1, 256)))
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
            colors = base_cmap(np.linspace(0.4, 1.0, len(pairs_x)))
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
                ax.plot(
                    xq,
                    z,
                    linewidth=kw.get('lw', 1),
                    color=kw.get('trend_color', 'black' if n_lines == 1 else col),
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

    def create_innernodes_figure(self) -> plt.Figure:
        data = self._get_data()
        ern = sorted(
            [n for n in data if n.emb_name == "affinity"],
            key=lambda n: n.emb_val or 0,
            reverse=True,
        )
        uorf = [n for n in data if n.emb_name == "tl_rate"]
        basic = [
            n
            for n in data
            if n.emb_name is None and n.type in ["Source", "Transcription", "Output"]
        ]
        inverse = [n for n in data if n.emb_name is None and "Inv" in n.type]
        base_cmap = plt.get_cmap(DEFAULT_CMAP_NAME)
        fig = plt.figure(figsize=(20, 15))
        rows = fig.subfigures(3, 1, height_ratios=[1, 1, 1], hspace=0.1)

        if ern:
            rows[0].suptitle("ERN Nodes and Embeddings", fontsize=16, fontweight='bold', y=1.05)
            cols = rows[0].subfigures(1, 2, width_ratios=[3, 1])
            n_ern = min(4, len(ern))
            axes = cols[0].subplots(
                1, n_ern, gridspec_kw={'width_ratios': [1] * n_ern, 'wspace': 0.3}
            )
            if n_ern == 1:
                axes = [axes]
            for i, node in enumerate(ern[:n_ern]):
                self._smart_scatter(axes[i], node, size=100000, cbar=(i == n_ern - 1))
                axes[i].set_title(f"ERN\n({node.name})", fontweight='bold', fontsize=12)
            emb_ax = cols[1].subplots(1, 1)
            names, vals = zip(*[(n.name, n.emb_val) for n in ern])
            emb_ax.barh(
                np.arange(len(names)), vals, color=base_cmap(np.linspace(0.4, 1, len(names)))
            )
            emb_ax.set(yticks=np.arange(len(names)), yticklabels=names, title="ERN Embeddings")
            emb_ax.invert_yaxis()
            emb_ax.set_box_aspect(2)

        rows[1].suptitle("Forward Nodes", fontsize=16, fontweight='bold', y=1.05)
        n_fwd = len(basic) + 1
        axes = rows[1].subplots(1, n_fwd, gridspec_kw={'width_ratios': [1] * n_fwd, 'wspace': 0.3})
        for ax, node in zip(axes, basic):
            self._smart_scatter(ax, node)
            ax.set_title(f"{node.type}\n{node.name}", fontweight='bold', fontsize=12)
        step = max(1, len(uorf) // 8)
        for node, col in zip(uorf[::step], base_cmap(np.linspace(0.4, 1, len(uorf[::step])))):
            self._smart_scatter(
                axes[-1],
                node,
                alpha=0.01,
                trend=True,
                color=col,
                trend_color=col,
                lw=2,
                markers=False,
            )
            axes[-1].plot([], [], color=col, label=node.name, linewidth=2)
        axes[-1].set_title("Translation\nmRNA → PRT", fontweight='bold', fontsize=12)
        axes[-1].legend(title="uORFs", loc="upper left", fontsize='x-small', frameon=False)

        if inverse and SHOW_INVERSE:
            rows[2].suptitle("Inverse Nodes", fontsize=16, fontweight='bold', y=1.05)
            axes = rows[2].subplots(
                1, len(inverse), gridspec_kw={'width_ratios': [1] * len(inverse), 'wspace': 0.3}
            )
            if len(inverse) == 1:
                axes = [axes]
            for ax, node in zip(axes, inverse):
                self._smart_scatter(ax, node)
                ax.set_title(f"{node.type}\n{node.name}", fontweight='bold', fontsize=12)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
        return fig

    def run(self, overwrite=True):
        self.figure_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = self.create_innernodes_figure()
        fig.savefig(self.figure_spec.output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


InnerNodesFigureSpec = type('InnerNodesFigureSpec', (FigureSpec,), {})
