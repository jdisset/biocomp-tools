from biocomptools.toollib.plot import Figure, PlotConfig, load_default_plotconf
from biocomptools.modelmodel import BiocompModel, NetworkModel, NodeSpec, load_model
from biocomptools.toollib.networkprediction import NetworkPrediction, reconstruct_from_flat
from biocomptools.logging_config import get_logger
from biocomp.plotutils import FigureSpec, FigureLayout, FigAx, PlotData
from biocomp.plotting.plotting_core import knn_stats, DEFAULT_CMAP_NAME, build_tree
from biocomp.old_network.network import Network, CoTransfection, Unit

from typing import Optional, List, Dict, Tuple, Union, Literal, TypeVar, TypeAlias, Annotated, Any
from pydantic import Field, ConfigDict, BaseModel, BeforeValidator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import KDTree
import jax.numpy as jnp

logger = get_logger(__name__)

NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]

ERN_ASPECT_RATIO = 1.0  # square plots for ERN scatter plots
OTHER_ASPECT_RATIO = 0.8  # slightly taller plots for everything else

SCATTER_ALPHA = 0.2
CMAP_MIN = 0.3  # lower bound for colormap truncation
CMAP_MAX = 1.0  # upper bound for colormap truncation

ERN_SCATTER_ALPHA = 1
ERN_CMAP_MIN = 0.0
ERN_CMAP_MAX = 1.0


ERN_SAMPLE_SIZE = 100_000  # sample size for ERN scatter plots
NODE_SAMPLE_SIZE = 30_000  # sample size for other node plots
N_PREDICT_SAMPLES = 100_000
TREND_K = 1000


SHOW_INVERSE_NODES = True
LABEL_FONT_SIZE = 12

ERN_COLORBAR_HEIGHT_RATIO = 0.75
ERN_EMBEDDINGS_HEIGHT_RATIO = 0.75
ERN_COLORBAR_WIDTH = 0.1

LABELS: Dict[str, Dict[str, str]] = {
    "ERN": {"x2d": "Protein Amount", "y2d": "mRNA Amount", "out": "Surviving mRNA"},
    "Translation": {
        "x1d": "Input mRNA (latent)",
        "y1d": "Output Protein (latent)",
        "out": "Protein Level",
    },
    "Transcription": {
        "x1d": "Input DNA (latent)",
        "y1d": "Output mRNA (latent)",
        "out": "mRNA Level",
    },
    "Source": {
        "x1d": "Input (latent)",
        "y1d": "Output DNA (latent)",
        "out": "DNA Level",
    },
    "Output": {
        "x1d": "Input Protein (latent)",
        "y1d": "Fluorescence (latent)",
        "out": "Fluorescence",
    },
}


class NodeData(BaseModel):
    node_name: str
    node_type: str
    plot_data: PlotData
    embedding_name: str | None = None
    embedding_value: float | None = None

    class Config:
        arbitrary_types_allowed = True


class RowSubFigureLayout(FigureLayout):
    figsize: Tuple[int, int] = (20, 15)
    nrows: int = 3
    ncols: int = 1
    hspace: float = 0.1
    wspace: float = 0.4
    height_ratios: List[float] = [1, 1, 1]

    def make_figure(self):
        fig = plt.figure(figsize=self.figsize, constrained_layout=False)
        subfigs = fig.subfigures(
            self.nrows,
            self.ncols,
            height_ratios=self.height_ratios,
            hspace=self.hspace,
        )
        return FigAx(figure=fig, subfigs=subfigs)


class InnerNodesFigureSpec(FigureSpec):
    title: Optional[str] = 'Inner Nodes and Embeddings'
    output_file: Optional[str] = 'inner_nodes.png'
    layout: FigureLayout = Field(default_factory=RowSubFigureLayout)
    dpi: int = 150
    title_kwargs: Dict[str, Any] = {'fontsize': 16, 'fontweight': 'bold', 'y': 1.05}

    def save_figure(self, figax: FigAx) -> None:
        assert self.output_file is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        figax.figure.savefig(self.output_path, bbox_inches="tight", dpi=self.dpi)


class InnerNodesFigure(Figure):
    """
    Figure for visualizing inner node functions of a BiocompModel.

    Visualizes ERN embeddings, basic node functions, translation with different uORFs,
    and shows embeddings of different node types.
    """

    model: Annotated[BiocompModel, BeforeValidator(load_model)]
    n_samples: int = NODE_SAMPLE_SIZE
    figure_spec: FigureSpec = Field(default_factory=InnerNodesFigureSpec)

    plot_config: PlotConfig = Field(default_factory=load_default_plotconf)

    def _cmap(self, cmin: float = CMAP_MIN, cmax: float = CMAP_MAX) -> mpl.colors.Colormap:
        """Truncated colormap – keeps lower part light for readability."""
        return mpl.colors.LinearSegmentedColormap.from_list(
            f"trunc_{DEFAULT_CMAP_NAME}",
            plt.colormaps.get_cmap(DEFAULT_CMAP_NAME)(np.linspace(cmin, cmax, 256)),
            N=256,
        )

    def _set_plot_aspect(self, ax, ratio):
        """Set a fixed aspect ratio on matplotlib plots regardless of axis units"""
        xvals, yvals = ax.get_xlim(), ax.axes.get_ylim()
        xrange = xvals[1] - xvals[0]
        yrange = yvals[1] - yvals[0]
        try:
            ax.set_aspect(ratio * (xrange / yrange), adjustable='box')
        except Exception as e:
            ax.set_aspect(ratio, adjustable='box')

    def _sample(self, x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly subsample *n* rows (without replacement)."""
        if y.shape[0] <= n:
            return x, y
        idx = np.random.choice(y.shape[0], n, replace=False)
        return x[idx], y[idx]

    def _trend(self, ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, k: int = 500):
        """Plot a dashed k‑NN smoothed trend line and return (xq, z)."""
        xq = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
        _, z = knn_stats(
            xq,
            y,
            tree=build_tree(x),
            stats=["std", "mean"],
            k=k,
            radius=0.1,
            min_points=200,
        )
        ax.plot(xq, z, linewidth=1, color="black", linestyle="dashed")
        return xq, z

    def _plot(
        self,
        ax: mpl.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        labels: Dict[str, str],
        size: int = NODE_SAMPLE_SIZE,
        scatter_alpha: float = SCATTER_ALPHA,
        *,
        add_colorbar: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        color: Any | None = None,
        add_trend: bool = True,
        cmap_min: float = CMAP_MIN,
        cmap_max: float = CMAP_MAX,
    ) -> mpl.collections.PathCollection | None:
        """Generic scatter / line plot helper – returns the scatter for colour‑bar handling."""
        x, y = self._sample(x, y, size)

        # 2‑D input – coloured by output value
        if x.shape[1] == 2:
            sc = ax.scatter(
                x[:, 0],
                x[:, 1],
                c=y.flatten(),
                cmap=self._cmap(cmin=cmap_min, cmax=cmap_max),
                s=10,
                alpha=scatter_alpha,
                linewidths=0,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel(labels.get("x2d", "Input 1"))
            ax.set_ylabel(labels.get("y2d", "Input 2"))

            if add_colorbar:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(labels.get("out", "Output"), fontsize=10)
                # Force *opaque* colour bar
                if hasattr(cbar, "solids"):
                    cbar.solids.set(alpha=1)
                for coll in cbar.ax.collections:
                    coll.set_alpha(1)
                for patch in cbar.ax.patches:
                    patch.set_alpha(1)
            return sc

        # 1‑D input – optionally add trend
        ax.scatter(
            x.flatten(),
            y.flatten(),
            s=4,
            alpha=0.05,
            linewidth=0,
            c=color if color is not None else "blue",
        )
        if add_trend:
            self._trend(ax, x, y)
        ax.set_xlabel(labels.get("x1d", "Input (latent)"))
        ax.set_ylabel(labels.get("y1d", "Output (latent)"))
        return None

    def _bar(
        self,
        ax: mpl.axes.Axes,
        data: Dict[str, float] | List[Tuple[str, float]],
        title: str,
        *,
        orientation: str = "horizontal",
        inset: bool = False,
        labelsize: int = 6,
        ratio=None,
    ) -> None:
        """Draw either a horizontal or vertical coloured bar chart."""
        items = sorted(data.items(), key=lambda x: x[1]) if isinstance(data, dict) else list(data)
        names, values = zip(*items)
        colors = self._cmap()(np.linspace(0, 1, len(names)))

        if orientation == "horizontal":
            y_pos = np.arange(len(names))
            ax.barh(y_pos, values, color=colors, height=0.7, edgecolor="black")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=labelsize if not inset else 6)
            ax.invert_yaxis()
        else:  # vertical
            x_pos = np.arange(len(names))
            ax.bar(x_pos, values, color=colors, width=0.7, edgecolor="black")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(names, fontsize=labelsize if inset else labelsize, rotation=90)

        if not inset:
            ax.set_title(title, fontsize=11, fontweight="bold")

        if ratio is not None:
            self._set_plot_aspect(ax, ratio)

    def extract_ern_embeddings(self) -> Dict[str, float]:
        try:
            ern_names = self.model.compute_config.node_functions["sequestron_ERN"].kwargs[
                "affinity_names"
            ]
            ern_names = [n.split("::")[1].split("#")[0] for n in ern_names]
            ern_embedding_values = self.model.shared_params["shared"]["ERN_5p"]["affinities"]
            return {k: float(v[0]) for k, v in zip(ern_names, ern_embedding_values)}
        except (KeyError, AttributeError) as e:
            logger.warning(f"Could not extract ERN embeddings: {e}")
            return {}

    def extract_uorf_embeddings(self) -> Dict[str, float]:
        uorf_names = self.model.compute_config.node_functions["translation"].kwargs[
            "quantization_names"
        ]
        uorf_values = self.model.shared_params["shared"]["quantization"]["values"]["tl_rate"]
        cleaned_names: List[str] = []
        for name in uorf_names:
            name = name.strip("_uORF")
            if name == "00_empty_tc":
                name = "none"
            cleaned_names.append(name)
        return {k: float(v[0]) for k, v in zip(cleaned_names, uorf_values)}

    def prepare_all_networks_and_specs(
        self, *, n_samples: int = N_PREDICT_SAMPLES
    ) -> Tuple[List[Network], List[NodeSpec], List[np.ndarray]]:
        all_networks: List[Network] = []
        all_node_specs: List[NodeSpec] = []
        all_inputs: List[np.ndarray] = []

        # ERN nodes ----------------------------------------------------------------
        ern_embeddings = self.extract_ern_embeddings()
        for name in ern_embeddings.keys():
            net = Network(
                cotx=[
                    CoTransfection(
                        units=[Unit(slots=["hEF1a", name]), Unit(slots=["hEF1a", "mKO2"])]
                    ),
                    CoTransfection(
                        units=[
                            Unit(slots=["hEF1a", f"{name}_rec", "eYFP"]),
                            Unit(slots=["hEF1a", "eBFP2"]),
                        ]
                    ),
                ],
                invert_on_build=True,
            )

            all_networks.append(net)
            all_node_specs.append(
                NodeSpec(
                    node_id=1,
                    network_id=len(all_networks) - 1,
                    extra_info={
                        "node_type": "ERN",
                        "node_name": name,
                        "embedding_name": "affinity",
                        "embedding_value": ern_embeddings[name],
                        "display_name": name,
                    },
                )
            )
            all_inputs.append(np.random.uniform(0.01, 0.8, (n_samples, 2)))

        # uORF translation nodes ----------------------------------------------------
        uorf_embeddings = self.extract_uorf_embeddings()
        uorf_names_raw = self.model.compute_config.node_functions["translation"].kwargs[
            "quantization_names"
        ]
        uorf_names_clean = list(uorf_embeddings.keys())
        for name_raw, name_clean in zip(uorf_names_raw, uorf_names_clean):
            net = Network(
                cotx=[
                    CoTransfection(
                        units=[Unit(slots=["hEF1a", name_raw]), Unit(slots=["hEF1a", "mKO2"])]
                    )
                ],
                invert_on_build=True,
            )
            all_networks.append(net)
            all_node_specs.append(
                NodeSpec(
                    node_id=2,
                    network_id=len(all_networks) - 1,
                    extra_info={
                        "node_type": "Translation",
                        "node_name": name_clean,
                        "embedding_name": "tl_rate",
                        "embedding_value": uorf_embeddings[name_clean],
                        "display_name": name_clean,
                    },
                )
            )
            all_inputs.append(np.random.uniform(0.01, 0.8, (n_samples, 1)))

        # Core source/transcription/output nodes -----------------------------------
        basic_network = Network(
            cotx=[
                CoTransfection(
                    units=[
                        Unit(slots=["hEF1a", "eYFP"], source="plsmd0"),
                        Unit(slots=["hEF1a", "eBFP2"], source="plsmd0"),
                    ]
                )
            ],
            invert_on_build=True,
        )
        basic_node_configs: List[Dict[str, Any]] = [
            {"name": "Source", "node_id": 7, "display_name": "plasmid count → DNA"},
            {"name": "Transcription", "node_id": 3, "display_name": "DNA → mRNA"},
            {"name": "Output", "node_id": 0, "display_name": "PRT → fluo"},
        ]
        basic_network.build()

        # Add inverse nodes when requested ----------------------------------------
        if SHOW_INVERSE_NODES:
            for row_id, row in basic_network.compute_graph.iterrows():
                if getattr(row, "is_inverse_of", None) is not None:
                    type_map = {
                        "inv_source": "DNA → plasmid count",
                        "inv_transcription": "mRNA → DNA",
                        "inv_translation": "Fluo → mRNA",
                    }
                    display_name = type_map.get(row.type, f"Inverse {row.type}")
                    basic_node_configs.append(
                        {
                            "name": row.type.title().replace("_", " "),
                            "node_id": row_id,
                            "display_name": display_name,
                        }
                    )

        basic_network_id = len(all_networks)
        all_networks.append(basic_network)
        all_inputs.append(np.random.uniform(0.01, 0.8, (n_samples, 1)))
        for cfg in basic_node_configs:
            all_node_specs.append(
                NodeSpec(
                    node_id=cfg["node_id"],
                    network_id=basic_network_id,
                    extra_info={
                        "node_type": cfg["name"],
                        "node_name": cfg["display_name"],
                        "embedding_name": None,
                        "embedding_value": None,
                        "display_name": cfg["display_name"],
                    },
                )
            )

        return all_networks, all_node_specs, all_inputs

    def get_all_node_data_unified(self, *, n_samples: int = N_PREDICT_SAMPLES) -> List[NodeData]:
        """Get all node data using a single, unified NetworkPrediction call."""
        networks, node_specs, inputs = self.prepare_all_networks_and_specs(n_samples=n_samples)
        nmod = NetworkModel(model=self.model, network=networks)
        pred = NetworkPrediction(
            predict_at=inputs,
            network_model=nmod,
            collection_points=node_specs,
            disable_variational=True,
            z_value="uniform",
            already_latent=True,
        )
        all_plot_data = pred.get_data()
        node_data_list: List[NodeData] = []
        for plot_data in all_plot_data:
            extra_info = plot_data.metadata["collection_point_nodespec"].extra_info
            node_data_list.append(
                NodeData(
                    node_name=extra_info["node_name"],
                    node_type=extra_info["node_type"],
                    plot_data=plot_data,
                    embedding_name=extra_info["embedding_name"],
                    embedding_value=extra_info["embedding_value"],
                )
            )
        return node_data_list

    def plot_node_data(
        self, ax: mpl.axes.Axes, node: NodeData, *, size: int = NODE_SAMPLE_SIZE
    ) -> None:
        """Dispatch plotting by node type and manage aspect ratio."""
        labels = LABELS.get(node.node_type, {})
        aspect_ratio = ERN_ASPECT_RATIO if node.node_type == "ERN" else OTHER_ASPECT_RATIO

        plot_data = node.plot_data
        x, y = plot_data.x, plot_data.y

        title = (
            f"{node.node_type}\n({node.node_name})"
            if node.embedding_name
            else (
                f"{node.node_type}\n{node.node_name}"
                if node.node_name != node.node_type
                else node.node_type
            )
        )

        if "input_shapes" in plot_data.metadata:
            # Reconstruct flattened inputs/outputs when needed
            x_r = reconstruct_from_flat(x, plot_data.metadata["input_shapes"])
            y_r = reconstruct_from_flat(y, plot_data.metadata["output_shapes"])
            n_in, n_out = len(x_r), len(y_r)

            # 1 input ‑> many outputs or vice‑versa
            if n_in > 1 and n_out == 1:
                self._plot(
                    ax,
                    np.column_stack(x_r),
                    y_r[0],
                    labels,
                    size,
                )
                self._set_plot_aspect(ax, aspect_ratio)
                ax.set_title(title, fontweight="bold", fontsize=LABEL_FONT_SIZE)
                return

            # General case – overlay plots with unique colours
            allX = [x_r[0]] * n_out if n_out > 1 and n_in == 1 else x_r if n_in > 1 else [x_r[0]]
            allY = y_r if n_out > 1 else [y_r[0]] * n_in if n_in > 1 else y_r
            legends = (
                [f"out {i}" for i in range(n_out)]
                if n_out > 1
                else ([f"in {i}" for i in range(n_in)] if n_in > 1 else [""])
            )
            colors = self._cmap()(np.linspace(0, 0.7, len(allX)))
            for j, (X, Y, col, leg) in enumerate(zip(allX, allY, colors, legends)):
                X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
                self._plot(ax, X, Y, labels, size, color=col, add_trend=True)
                if len(legends) > 1 and leg:
                    xq, z = self._trend(ax, X, Y)
                    idx = np.arange(0, 200, 20) + int((j / len(allX)) * 100)
                    idx = idx[idx < len(z)]
                    ax.plot(
                        xq[idx],
                        z[idx],
                        marker=["^", "s", "X", "v", "D", "P", "o", "p", "h", "d"][j % 10],
                        markersize=7,
                        color="black",
                        linestyle="None",
                        label=leg,
                        markerfacecolor="none",
                        markeredgewidth=1,
                    )
            if len(legends) > 1 and any(legends):
                ax.legend(loc="upper left", fontsize="x-small", frameon=False)
        else:
            self._plot(ax, x, y, labels, size)

        ax.set_title(title, fontweight="bold", fontsize=LABEL_FONT_SIZE)
        self._set_plot_aspect(ax, aspect_ratio)

    def _plot_ern_row(self, row_fig: mpl.figure.SubFigure, ern_nodes: List[NodeData]) -> int:
        if not ern_nodes:
            # If no ERN nodes, create empty subplot or placeholder
            row_fig.suptitle(
                "ERN Nodes and Embeddings (No ERN data available)",
                fontsize=16,
                y=1.0,
                fontweight="bold",
            )
            ax = row_fig.add_subplot(1, 1, 1)
            ax.text(
                0.5,
                0.5,
                "No ERN nodes found in this model",
                ha='center',
                va='center',
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return

        row_fig.suptitle("ERN Nodes and Embeddings", fontsize=16, y=1.0, fontweight="bold")
        row_subfigs = row_fig.subfigures(1, 2, width_ratios=[4, 1], wspace=-0.05)

        # ── Scatter plots ─────────────────────────────────────────────────────────
        scatter_sub = row_subfigs[0]

        num_ern = min(4, len(ern_nodes))
        # Create a gridspec that allows for configurable height colorbar
        cbar_empty_ratio = (
            1.0 - ERN_COLORBAR_HEIGHT_RATIO
        ) / 2  # split remaining space above/below
        ern_gs = scatter_sub.add_gridspec(
            3,
            num_ern + 1,
            width_ratios=[1] * num_ern + [ERN_COLORBAR_WIDTH],
            height_ratios=[cbar_empty_ratio, ERN_COLORBAR_HEIGHT_RATIO, cbar_empty_ratio],
            wspace=0.3,
        )
        vmin = min(node.plot_data.y.min() for node in ern_nodes[:num_ern])
        vmax = max(node.plot_data.y.max() for node in ern_nodes[:num_ern])

        sc: mpl.collections.PathCollection | None = None
        for i in range(num_ern):
            # Span all 3 rows for the scatter plots
            ax = scatter_sub.add_subplot(ern_gs[:, i])
            sc = self._plot(
                ax,
                ern_nodes[i].plot_data.x,
                ern_nodes[i].plot_data.y,
                LABELS["ERN"],
                scatter_alpha=ERN_SCATTER_ALPHA,
                vmin=vmin,
                vmax=vmax,
                size=ERN_SAMPLE_SIZE,
                cmap_min=ERN_CMAP_MIN,
                cmap_max=ERN_CMAP_MAX,
            )
            self._set_plot_aspect(ax, ERN_ASPECT_RATIO)
            ax.set_title(
                f"ERN\n({ern_nodes[i].node_name})", fontweight="bold", fontsize=LABEL_FONT_SIZE
            )

        if sc is not None:
            cbar_ax = scatter_sub.add_subplot(ern_gs[1, num_ern])
            cbar = plt.colorbar(sc, cax=cbar_ax)
            cbar.set_label("Surviving mRNA", fontsize=10)
            if hasattr(cbar, "solids"):
                cbar.solids.set(alpha=1)
            for coll in cbar.ax.collections:
                coll.set_alpha(1)
            for patch in cbar.ax.patches:
                patch.set_alpha(1)

        emb_sub = row_subfigs[1]
        # Create configurable height for ERN embeddings
        emb_empty_ratio = (
            1.0 - ERN_EMBEDDINGS_HEIGHT_RATIO
        ) / 2  # split remaining space above/below
        emb_gs = emb_sub.add_gridspec(
            3, 1, height_ratios=[emb_empty_ratio, ERN_EMBEDDINGS_HEIGHT_RATIO, emb_empty_ratio]
        )
        emb_ax = emb_sub.add_subplot(emb_gs[1, 0])  # Use middle row with configurable height
        self._bar(
            emb_ax,
            {node.node_name: node.embedding_value for node in ern_nodes},
            "ERN Embeddings",
            orientation="horizontal",
            labelsize=12,
            ratio=0.45,
        )

    def _plot_forward_row(
        self,
        row_fig: mpl.figure.SubFigure,
        basic_nodes: List[NodeData],
        uorf_nodes: List[NodeData],
    ) -> int:
        row_fig.suptitle(f"Forward Nodes", fontsize=16, y=1.0, fontweight="bold")

        num_forward = len(basic_nodes) + 1  # +1 for the aggregated translation panel
        forward_gs = row_fig.add_gridspec(
            1, num_forward, width_ratios=[1] * num_forward, wspace=0.3
        )

        # Reorder nodes: Source, Transcription, Translation, Output
        source_transcription = [
            node for node in basic_nodes if node.node_type in ["Source", "Transcription"]
        ]
        output_nodes = [node for node in basic_nodes if node.node_type == "Output"]

        # Plot Source and Transcription first
        col_idx = 0
        for node in source_transcription:
            ax = row_fig.add_subplot(forward_gs[0, col_idx])
            self.plot_node_data(ax, node)
            col_idx += 1

        # Aggregated Translation panel with trend lines per uORF -------------------
        ax_trans = row_fig.add_subplot(forward_gs[0, col_idx])
        step = max(1, len(uorf_nodes) // 8)
        selected_uorfs = uorf_nodes[::step][:8]
        colors = self._cmap()(np.linspace(0, 1, len(selected_uorfs)))
        for node, col in zip(selected_uorfs, colors):
            self._plot(
                ax_trans,
                node.plot_data.x,
                node.plot_data.y,
                LABELS["Translation"],
                size=NODE_SAMPLE_SIZE,
                color=col,
                scatter_alpha=0.1,
                add_trend=False,
            )
            xq, z = self._trend(ax_trans, node.plot_data.x, node.plot_data.y, k=TREND_K)
            ax_trans.plot(xq, z, linewidth=2, color=col, label=node.node_name)

        ax_trans.set_title("Translation\nmRNA → PRT", fontsize=LABEL_FONT_SIZE, fontweight="bold")
        ax_trans.set_xlabel("Input mRNA (latent)", fontsize=10)
        ax_trans.set_ylabel("Output Protein (latent)", fontsize=10)
        ax_trans.legend(
            title="uORFs",
            loc="upper left",
            fontsize="x-small",
            title_fontsize="small",
            frameon=False,
        )
        self._set_plot_aspect(ax_trans, OTHER_ASPECT_RATIO)

        # Inset with uORF embeddings ----------------------------------------------
        ax_inset = inset_axes(
            ax_trans,
            width="35%",
            height="25%",
            loc="lower right",
            bbox_to_anchor=(-0.01, 0.12, 1, 1),
            bbox_transform=ax_trans.transAxes,
        )
        uorf_embs = {node.node_name: node.embedding_value for node in uorf_nodes}
        names, values = list(uorf_embs.keys()), list(uorf_embs.values())
        if len(names) > 8:
            step = len(names) // 8
            names, values = names[::step][:8], values[::step][:8]
        self._bar(ax_inset, list(zip(names, values)), "", orientation="vertical", inset=True)
        ax_inset.set_title("uORF Embeddings", fontsize=8)

        col_idx += 1

        # Plot Output nodes last
        for node in output_nodes:
            ax = row_fig.add_subplot(forward_gs[0, col_idx])
            self.plot_node_data(ax, node)
            col_idx += 1

    def _plot_inverse_row(self, row_fig: mpl.figure.SubFigure, inverse_nodes: List[NodeData]):
        if not inverse_nodes:
            return

        row_fig.suptitle(f"Inverse Nodes", fontsize=16, y=1.05, fontweight="bold")

        num_inverse = len(inverse_nodes)
        inverse_gs = row_fig.add_gridspec(
            1, num_inverse, width_ratios=[1] * num_inverse, wspace=0.3
        )

        for i, node in enumerate(inverse_nodes):
            ax = row_fig.add_subplot(inverse_gs[0, i])
            self.plot_node_data(ax, node)

    def create_innernodes_figure(self, *, figsize: Tuple[int, int] = (20, 15)) -> mpl.figure.Figure:
        # Gather all node data -----------------------------------------------------
        all_node_data = self.get_all_node_data_unified(n_samples=self.n_samples)

        ern_nodes = sorted(
            [nd for nd in all_node_data if nd.embedding_name == "affinity"],
            key=lambda n: n.embedding_value,
            reverse=True,
        )

        uorf_nodes = [nd for nd in all_node_data if nd.embedding_name == "tl_rate"]

        basic_nodes = [nd for nd in all_node_data if nd.embedding_name is None]

        basic_core = [
            nd for nd in basic_nodes if nd.node_type in ["Source", "Transcription", "Output"]
        ]
        inverse_nodes = [
            nd
            for nd in basic_nodes
            if nd.node_type not in ["Source", "Transcription", "Output"] and SHOW_INVERSE_NODES
        ]

        # Main figure skeleton -----------------------------------------------------
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        main_subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 1], hspace=0.1)

        # Populate rows ------------------------------------------------------------
        try:
            self._plot_ern_row(main_subfigs[0], ern_nodes)
        except Exception as e:
            logger.warning(f"Failed to plot ERN row: {e}")
            # Create a placeholder for the ERN row
            ax = main_subfigs[0].add_subplot(1, 1, 1)
            ax.text(
                0.5,
                0.5,
                f"ERN plotting failed: {str(e)}",
                ha='center',
                va='center',
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        self._plot_forward_row(main_subfigs[1], basic_core, uorf_nodes)
        if SHOW_INVERSE_NODES:
            self._plot_inverse_row(main_subfigs[2], inverse_nodes)

        # Final figure adjustments
        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
        return fig

    def run(self, overwrite=True):
        """Execute the figure creation process"""
        logger.debug("=== InnerNodesFigure.run() STARTED ===")
        logger.debug(f"Model signature: {self.model.signature}")
        logger.debug(f"Figure spec output file: {self.figure_spec.output_file}")
        logger.debug(f"N samples: {self.n_samples}")

        try:
            logger.debug("Creating figure layout...")
            self._figax = self.figure_spec.make_figure()

            logger.debug("Generating inner nodes figure...")
            # Replace the figure with our new implementation
            new_fig = self.create_innernodes_figure(figsize=self.figure_spec.layout.figsize)

            logger.debug("Replacing figure in figax...")
            # Replace the figure in figax
            self._figax.figure = new_fig

            logger.debug("Finalizing figure...")
            self.figure_spec.finalize(self._figax)
            logger.debug("=== InnerNodesFigure.run() COMPLETED SUCCESSFULLY ===")

        except Exception as e:
            logger.error(f"Error creating inner nodes figure: {e}")
            import traceback

            logger.error(traceback.format_exc())
            logger.error("=== InnerNodesFigure.run() FAILED ===")
            raise
