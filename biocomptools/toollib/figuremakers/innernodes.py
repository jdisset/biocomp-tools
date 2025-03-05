from biocomptools.toollib.plot import Figure, PlotConfig, load_default_plotconf
from biocomptools.modelmodel import BiocompModel, NetworkModel, NodeSpec, load_model
from biocomptools.toollib.networkprediction import NetworkPrediction
from biocomptools.logging_config import get_logger
from biocomp.plotutils import FigureSpec, FigureLayout, FigAx
from biocomp.plotting.plotting_core import knn_avg, DEFAULT_CMAP_NAME
from biocomp.network import Network, CoTransfection, Unit
import biocomptools.toollib.models as md
import biocomp.plotutils as pu
from pathlib import Path

from typing import Optional, List, Dict, Tuple, Union, Literal, TypeVar, TypeAlias, Annotated
from pydantic import Field, ConfigDict, BaseModel, BeforeValidator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import KDTree
import jax
import jax.numpy as jnp
from pathlib import Path

logger = get_logger(__name__)

NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]


class RowSubFigureLayout(FigureLayout):
    figsize: Tuple[int, int] = (20, 10)
    nrows: int = 2
    ncols: int = 1
    hspace: float = 0.4
    wspace: float = 0.4
    height_ratios: List[float] = [1.25, 1]

    def make_figure(self):
        fig = plt.figure(figsize=self.figsize)
        subfigs = fig.subfigures(
            self.nrows,
            self.ncols,
            height_ratios=self.height_ratios,
            hspace=self.hspace,
            wspace=self.wspace,
        )
        return FigAx(figure=fig, subfigs=subfigs)


class InnerNodesFigureSpec(FigureSpec):
    title: Optional[str] = 'Inner Nodes and Embeddings'
    output_file: Optional[str] = 'inner_nodes.png'
    layout: FigureLayout = Field(default_factory=RowSubFigureLayout)
    dpi: int = 200

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Annotated[BiocompModel, BeforeValidator(load_model)]
    n_samples: int = 150000
    figure_spec: FigureSpec = Field(default_factory=InnerNodesFigureSpec)

    plot_config: PlotConfig = Field(default_factory=load_default_plotconf)

    def extract_ern_embeddings(self) -> Dict[str, List[float]]:
        """Extract ERN embeddings from model"""
        ern_names = self.model.compute_config.node_functions['sequestron_ERN'].kwargs[
            'affinity_names'
        ]
        ern_names = [n.split('::')[1].split('#')[0] for n in ern_names]
        ern_embedding_values = self.model.shared_params["shared"]["ERN_5p"]["affinities"]
        return {k: v for k, v in zip(ern_names, ern_embedding_values)}

    def extract_uorf_embeddings(self) -> Dict[str, List[float]]:
        """Extract uORF embeddings from model"""
        uorf_names = self.model.compute_config.node_functions['translation'].kwargs[
            'quantization_names'
        ]
        uorf_values = self.model.shared_params["shared"]["quantization"]["values"]["tl_rate"]

        # Clean up uORF names for better display
        cleaned_names = []
        for name in uorf_names:
            name = name.strip('_uORF')
            if name == '00_empty_tc':
                name = 'none'
            cleaned_names.append(name)

        return {k: v for k, v in zip(cleaned_names, uorf_values)}

    def create_ern_networks(self, ern_names: List[str]) -> List[md.Network]:
        """Create networks for ERN visualization"""
        networks = []
        for name in ern_names:
            net = md.Network.from_network(
                Network(
                    cotx=[
                        CoTransfection(
                            units=[
                                Unit(slots=["hEF1a", name]),
                                Unit(slots=["hEF1a", "mKO2"]),
                            ]
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
            )
            networks.append(net)
        return networks

    def create_uorf_networks(self, uorf_names: List[str]) -> List[md.Network]:
        """Create networks for uORF visualization"""
        uorf_nets = [
            md.Network.from_network(
                Network(
                    cotx=[
                        CoTransfection(
                            units=[
                                Unit(slots=["hEF1a", uorf]),
                                Unit(slots=["hEF1a", "mKO2"]),
                            ]
                        ),
                    ],
                    invert_on_build=True,
                )
            )
            for uorf in uorf_names
        ]
        return uorf_nets

    def create_basic_network(self) -> md.Network:
        """Create a simple network for basic nodes"""
        n = md.Network.from_network(
            Network(
                cotx=[
                    CoTransfection(
                        units=[
                            Unit(slots=["hEF1a", "eYFP"], source='plsmd0'),
                            Unit(slots=["hEF1a", "eBFP2"], source='plsmd0'),
                        ]
                    ),
                ],
                invert_on_build=True,
            )
        )
        return n

    def collect_node_data(self) -> Dict:
        """Collect data from all interesting nodes"""
        result_data = {}

        # --- ERN NETWORKS ---
        ern_embeddings = self.extract_ern_embeddings()
        ern_names = list(ern_embeddings.keys())
        ern_networks = self.create_ern_networks(ern_names)
        ern_nmod = NetworkModel(model=self.model, network=ern_networks)

        # Create collection points for ERN nodes
        ern_nodes = [
            NodeSpec(
                node_id=1,
                network_id=i,
            )
            for i in range(len(ern_networks))
        ]

        # Generate random inputs for ERN networks
        ern_inputs = np.random.uniform(0.01, 0.8, (self.n_samples, 2))
        inputs = [ern_inputs] * len(ern_networks)

        ern_pred = NetworkPrediction(
            predict_at=inputs,
            network_model=ern_nmod,
            collection_points=ern_nodes,
            disable_variational=True,
            z_value=0.5,
            already_latent=True,
        )

        # Get ERN node data
        ern_data = ern_pred.get_data()
        result_data['ern_data'] = ern_data
        result_data['ern_names'] = ern_names
        result_data['ern_embeddings'] = ern_embeddings

        # --- UORF NETWORKS ---
        uorf_embeddings = self.extract_uorf_embeddings()
        uorf_names_raw = self.model.compute_config.node_functions['translation'].kwargs[
            'quantization_names'
        ]
        uorf_names = list(uorf_embeddings.keys())
        uorf_networks = self.create_uorf_networks(uorf_names_raw)
        uorf_nmod = NetworkModel(model=self.model, network=uorf_networks)

        logger.info(f"Created {len(uorf_networks)} uORF networks")

        translation_node_id = 2

        translation_nodes = [
            NodeSpec(network_id=i, node_id=translation_node_id) for i in range(len(uorf_networks))
        ]

        # Generate inputs for uORF networks
        uorf_inputs = np.random.uniform(0.01, 0.8, (self.n_samples, 1))
        uorf_inputs = [uorf_inputs.copy() for _ in range(len(uorf_networks))]

        uorf_pred = NetworkPrediction(
            predict_at=uorf_inputs,
            network_model=uorf_nmod,
            collection_points=translation_nodes,
            disable_variational=True,
            already_latent=True,
            z_value='uniform',
        )

        # Get translation node data
        translation_data = uorf_pred.get_data()
        result_data['translation_data'] = translation_data
        result_data['uorf_names'] = uorf_names
        result_data['uorf_embeddings'] = uorf_embeddings

        # --- BASIC NETWORK NODES ---
        basic_network = self.create_basic_network()
        basic_nmod = NetworkModel(model=self.model, network=basic_network)

        # Node IDs for basic nodes - from the log output
        basic_node_ids = {
            "Source\n$\\mathrm{a.k.a. plasmid, DNA \\rightarrow DNA}$": 7,
            "Transcription\n$\\mathrm{DNA \\rightarrow mRNA}$": 3,
            "Output\n$\\mathrm{PRT \\rightarrow fluo}$": 0,
        }

        # Create collection points for basic nodes
        basic_nodes = [
            NodeSpec(network_id=0, node_id=node_id) for node_type, node_id in basic_node_ids.items()
        ]

        # Generate inputs for basic network
        basic_inputs = np.random.uniform(0.01, 0.8, (self.n_samples, 1))

        basic_pred = NetworkPrediction(
            predict_at=[basic_inputs],
            network_model=basic_nmod,
            collection_points=basic_nodes,
            disable_variational=True,
            already_latent=True,
            z_value='uniform',
        )

        # Get basic node data
        basic_data = basic_pred.get_data()
        result_data['basic_data'] = basic_data
        result_data['basic_node_types'] = list(basic_node_ids.keys())

        return result_data

    def add_subplot_letter(self, ax, letter, xpos=-0.0, ypos=1.22):
        """Add a letter annotation to the top-left of a subplot."""
        ax.text(
            xpos,
            ypos,
            letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='right',
        )

    def create_visualization(self, data, figax, subsample_size=20000):
        """Create a visualization with organized subfigures layout."""
        pconf = self.plot_config

        with mpl.rc_context(pconf.rc_context):
            # Create truncated colormap for embeddings and translation plot
            base_cmap = plt.cm.get_cmap(DEFAULT_CMAP_NAME)
            truncated_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                f'trunc_{DEFAULT_CMAP_NAME}', base_cmap(np.linspace(0.3, 1, 256)), N=256
            )

            fig = figax.figure
            subfigs = figax.subfigs

            # Row 1: ERN nodes and embeddings
            row1_fig = subfigs[0]

            # Number of ERN nodes to display
            ern_data = data['ern_data']
            ern_names = data['ern_names']
            ern_embeddings = data['ern_embeddings']
            num_ern_to_show = min(4, len(ern_names))

            # Create equal columns for all plots in row 1 (ERN nodes + embeddings)
            row1_gs = row1_fig.add_gridspec(
                1, num_ern_to_show + 2, width_ratios=[1] * num_ern_to_show + [0.1, 0.5]
            )

            # Plot ERN nodes
            vlims = [
                np.min([np.min(data.y) for data in ern_data]),
                np.max([np.max(data.y) for data in ern_data]),
            ]
            sc = None
            for i in range(num_ern_to_show):
                ax = row1_fig.add_subplot(row1_gs[0, i])
                name = ern_names[i]
                ern_data_item = ern_data[i]

                self.add_subplot_letter(ax, chr(65 + i))

                sc = ax.scatter(
                    ern_data_item.x[:, 0],
                    ern_data_item.x[:, 1],
                    c=ern_data_item.y.flatten(),
                    cmap=DEFAULT_CMAP_NAME,
                    s=10,
                    alpha=0.2,
                    linewidths=0,
                    vmin=vlims[0],
                    vmax=vlims[1],
                )

                ax.set_title(f'ERN: {name}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Protein Amount', fontsize=10)
                ax.set_ylabel('mRNA Amount', fontsize=10)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_color('gray')
                ax.tick_params(width=0.8, length=4, color='gray')

            row1_fig.subplots_adjust(wspace=self.wspace, right=0.9)
            if sc is not None:
                cbar_ax = row1_fig.add_subplot(row1_gs[0, num_ern_to_show])
                cbar = plt.colorbar(sc, cax=cbar_ax)
                cbar.solids.set(alpha=1)
                cbar.set_label('Surviving mRNA', fontsize=10, labelpad=-60)

            # ERN embeddings plot
            ax_ern_emb = row1_fig.add_subplot(row1_gs[0, -1])
            self.add_subplot_letter(ax_ern_emb, chr(65 + num_ern_to_show))

            # Sort by embedding value
            sorted_ern = sorted(ern_embeddings.items(), key=lambda x: x[1][0])
            names = [item[0] for item in sorted_ern]
            values = [item[1][0] for item in sorted_ern]

            norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values))

            y_pos = np.arange(len(names))
            bars = ax_ern_emb.barh(
                y_pos,
                values,
                color=[truncated_cmap(norm(val)) for val in values],
                edgecolor='black',
                height=0.7,
            )

            ax_ern_emb.set_yticks(y_pos)
            ax_ern_emb.set_yticklabels(names, fontsize=8)
            ax_ern_emb.set_title('ERN embeddings', fontsize=11, fontweight='bold')
            ax_ern_emb.set_xlabel('Value', fontsize=9)

            for spine in ax_ern_emb.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('gray')
            ax_ern_emb.tick_params(width=0.8, length=4, color='gray')
            ax_ern_emb.invert_yaxis()

            # Row 2: simple inner nodes, translation, and uORF embeddings
            row2_fig = subfigs[1]

            basic_data = data['basic_data']
            basic_node_types = data['basic_node_types']
            num_basic_nodes = len(basic_node_types)

            row2_gs = row2_fig.add_gridspec(
                1, num_basic_nodes + 2, width_ratios=[1] * (num_basic_nodes + 1) + [0.5]
            )

            letter_idx = 65 + num_ern_to_show + 1

            # Plot basic nodes
            for i, (node_type, node_data) in enumerate(zip(basic_node_types, basic_data)):
                ax = row2_fig.add_subplot(row2_gs[0, i])

                self.add_subplot_letter(ax, chr(letter_idx))
                letter_idx += 1

                n_outputs = node_data.metadata['full_y'].shape[1]
                n_inputs = node_data.x.shape[1]

                if n_outputs > 1:
                    if n_inputs == 1:  # single input, multiple outputs
                        allX = [node_data.x.flatten()[:, None]] * n_outputs
                    else:
                        assert (
                            n_inputs == n_outputs
                        ), f"n_inputs: {n_inputs}, n_outputs: {n_outputs}"
                        allX = [
                            node_data.x[:, k].flatten()[:, None]
                            for k in range(node_data.x.shape[1])
                        ]
                    allY = [
                        node_data.metadata['full_y'][:, k].flatten()[:, None]
                        for k in range(n_outputs)
                    ]
                    legends = [f'out {i}' for i in range(n_outputs)]
                else:
                    if n_inputs > 1:  # multiple inputs, single output
                        allX = [node_data.x[:, k].flatten()[:, None] for k in range(n_inputs)]
                        allY = [node_data.y.flatten()[:, None]] * n_inputs
                        legends = [f'in {i}' for i in range(n_inputs)]

                    else:  # single input, single output
                        allX = [node_data.x.flatten()[:, None]]
                        allY = [node_data.y.flatten()[:, None]]
                        legends = ['']

                colors = truncated_cmap(np.linspace(0, 0.7, len(allX)))
                npoints = len(allX[0])
                sample_idx = np.random.choice(npoints, min(subsample_size, npoints), replace=False)
                for j, (X, Y, c, legend) in enumerate(zip(allX, allY, colors, legends)):
                    ax.scatter(X[sample_idx], Y[sample_idx], s=4, alpha=0.05, linewidth=0, c=c)

                    # calculate trend line using KNN
                    tree = KDTree(X)
                    xquery_min, xquery_max = X.min(), X.max()
                    res = 200
                    xquery = np.linspace(xquery_min, xquery_max, res).reshape(-1, 1)

                    z, _ = knn_avg(
                        xquery, Y, tree, avg_method="mean", min_points=500, k=1000, radius=0.1
                    )

                    ax.plot(
                        xquery,
                        z,
                        linewidth=1,
                        color='black',
                        linestyle='dashed',
                    )

                    if len(legends) > 1:
                        marker = ['^', 's', 'X', 'v', 'D', 'P', 'o', 'p', 'h', 'd'][j]
                        nmarkers = 10
                        interval_size = res // nmarkers
                        offset = (j / len(allX)) * (res * 0.5 / nmarkers)
                        marker_idx = np.arange(0, res, interval_size) + int(offset)
                        ax.plot(
                            xquery[marker_idx],
                            z[marker_idx],
                            marker=marker,
                            markersize=7,
                            color='black',
                            linestyle='None',
                            label=legend,
                            markerfacecolor='none',
                            markeredgewidth=1,
                        )

                ax.set_title(f'{node_type}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Input (latent)', fontsize=10)
                # show labels if more than one plot
                if len(legends) > 1:
                    ax.legend(
                        loc='upper left',
                        fontsize='x-small',
                        frameon=False,
                        facecolor='white',
                    )

                if i == 0:
                    ax.set_ylabel('Output (latent)', fontsize=10)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_color('gray')
                ax.tick_params(width=0.8, length=4, color='gray')

            translation_data = data['translation_data']
            uorf_names = data['uorf_names']
            uorf_embeddings = data['uorf_embeddings']

            ax_trans = row2_fig.add_subplot(row2_gs[0, num_basic_nodes])

            self.add_subplot_letter(ax_trans, chr(letter_idx))
            letter_idx += 1

            uorf_list = [(name, float(uorf_embeddings[name][0])) for name in uorf_names]

            plot_uorfs = []
            for i in range(0, len(uorf_list)):
                if i < len(uorf_list):
                    name, value = uorf_list[i]
                    orig_idx = uorf_names.index(name)
                    plot_uorfs.append((orig_idx, name, value))

            colors = truncated_cmap(np.linspace(0, 1, len(plot_uorfs)))

            for i, (idx, name, _) in enumerate(plot_uorfs):
                color = colors[i]
                plot_data = translation_data[idx]

                X = plot_data.x.flatten()[:, None]
                Y = plot_data.y.flatten()[:, None]

                tree = KDTree(X)
                xquery_min, xquery_max = X.min(), X.max()
                res = 200
                xquery = np.linspace(xquery_min, xquery_max, res).reshape(-1, 1)
                z, _ = knn_avg(
                    xquery, Y, tree, avg_method="mean", min_points=500, k=1000, radius=0.1
                )

                ax_trans.plot(xquery, z, linewidth=2, color=color, label=f"{name}")

            ax_trans.set_title(
                "Translation\n$\\mathrm{mRNA \\rightarrow PRT}$", fontsize=12, fontweight='bold'
            )
            ax_trans.set_xlabel('Input mRNA (latent)', fontsize=10)

            ax_trans.legend(
                title="uORFs",
                loc='upper left',
                fontsize='x-small',
                title_fontsize='small',
                frameon=False,
                facecolor='white',
            )

            for spine in ax_trans.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('gray')
            ax_trans.tick_params(width=0.8, length=4, color='gray')

            # uORF embeddings on right of row 2
            ax_uorf_emb = row2_fig.add_subplot(row2_gs[0, -1])

            self.add_subplot_letter(ax_uorf_emb, chr(letter_idx))

            names = list(uorf_embeddings.keys())
            values = [float(uorf_embeddings[name][0]) for name in names]

            n_uorfs = len(names)
            position_colors = truncated_cmap(np.linspace(0, 1, n_uorfs))

            y_pos = np.arange(len(names))
            bars = ax_uorf_emb.barh(
                y_pos,
                values,
                color=position_colors,
                height=0.7,
                edgecolor='black',
            )

            ax_uorf_emb.set_yticks(y_pos)
            ax_uorf_emb.set_yticklabels(names, fontsize=8)
            ax_uorf_emb.set_title('uORF Embeddings', fontsize=11, fontweight='bold')
            ax_uorf_emb.set_xlabel('Value', fontsize=9)

            for spine in ax_uorf_emb.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('gray')
            ax_uorf_emb.tick_params(width=0.8, length=4, color='gray')

            row1_fig.subplots_adjust(wspace=self.wspace, right=0.9)
            row2_fig.subplots_adjust(wspace=self.wspace, right=0.9)

            return fig

    def run(self, overwrite=True):
        """Execute the figure creation process"""

        logger.debug("Collecting node data from model")
        try:
            all_data = self.collect_node_data()

            figax = self.figure_spec.make_figure()
            self.create_visualization(all_data, figax)
            self.figure_spec.finalize(figax)

        except Exception as e:
            logger.error(f"Error creating inner nodes figure: {e}")
            import traceback

            logger.error(traceback.format_exc())
