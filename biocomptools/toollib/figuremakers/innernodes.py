from biocomptools.toollib.plot import Figure, PlotConfig, load_default_plotconf
from biocomptools.modelmodel import BiocompModel, NetworkModel, NodeSpec, load_model
from biocomptools.toollib.networkprediction import NetworkPrediction
from biocomptools.logging_config import get_logger
from biocomp.plotutils import FigureSpec, FigureLayout, FigAx
from biocomp.plotting.plotting_core import knn_stats, DEFAULT_CMAP_NAME, build_tree
from biocomp.network import Network, CoTransfection, Unit

from typing import Optional, List, Dict, Tuple, Union, Literal, TypeVar, TypeAlias, Annotated, Any
from pydantic import Field, ConfigDict, BaseModel, BeforeValidator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import jax.numpy as jnp

logger = get_logger(__name__)

NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]


class RowSubFigureLayout(FigureLayout):
    figsize: Tuple[int, int] = (25, 13)
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
    n_samples: int = 100000
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

    def create_ern_networks(self, ern_names: List[str]) -> List[Network]:
        """Create networks for ERN visualization"""
        networks = []
        for name in ern_names:
            net = Network(
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
            networks.append(net)
        return networks

    def create_uorf_networks(self, uorf_names: List[str]) -> List[Network]:
        """Create networks for uORF visualization"""
        uorf_nets = [
            Network(
                cotx=[
                    CoTransfection(
                        units=[
                            Unit(slots=["hEF1a", uorf, "eBFP2"], source='plsmd0'),
                            Unit(slots=["hEF1a", "mKO2"], source='plsmd0'),
                        ]
                    ),
                ],
                invert_on_build=True,
            )
            for uorf in uorf_names
        ]
        return uorf_nets

    def collect_node_data(self) -> Dict:
        """Collect data from all interesting nodes"""
        result_data = {}

        # --- ERN NETWORKS ---
        try:
            ern_embeddings = self.extract_ern_embeddings()
        except Exception as e:
            logger.warning(f"Couldn't get ERN embeddings from model: {e}")
            ern_embeddings = {}

        if ern_embeddings:
            logger.info(f"Extracted {len(ern_embeddings)} ERN embeddings")
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
                max_evals=self.n_samples,
                disable_variational=True,
                already_latent=True,
                z_value=0.5,
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
        logger.debug(f"Created {len(uorf_networks)} uORF networks")
        logger.debug(f"UORF embeddings: {uorf_embeddings}")

        import pandas as pd

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.debug(f"uorf network comp g:\n{uorf_networks[2].compute_graph}")
            logger.debug(f"uorf network cdg:\n{uorf_networks[2].central_dogma_graph}")

        uorf_nmod = NetworkModel(model=self.model, network=uorf_networks)

        logger.info(f"Created {len(uorf_networks)} uORF networks")

        translation_node_id = 1
        translation_nodes = [
            NodeSpec(network_id=i, node_id=translation_node_id) for i in range(len(uorf_networks))
        ]
        logger.info(
            f"Created {len(translation_nodes)} translation nodes specs: {translation_nodes}"
        )

        uorf_inputs = np.random.uniform(0.0, 0.8, (self.n_samples, 1))
        uorf_inputs = [uorf_inputs.copy() for _ in range(len(uorf_networks))]

        basic_node_ids = {
            "Source\n$\\mathrm{a.k.a. plasmid, DNA \\rightarrow DNA}$": 7,
            "Transcription\n$\\mathrm{DNA \\rightarrow mRNA}$": 3,
            "Output\n$\\mathrm{PRT \\rightarrow fluo}$": 0,
        }

        basic_nodes = [
            NodeSpec(network_id=0, node_id=node_id) for node_type, node_id in basic_node_ids.items()
        ]

        uorf_pred = NetworkPrediction(
            predict_at=uorf_inputs,
            network_model=uorf_nmod,
            max_evals=self.n_samples,
            collection_points=translation_nodes + basic_nodes,
            disable_variational=True,
            already_latent=True,
            z_value='uniform',
        )

        vnodes = [
            uorf_pred.network_model._stack.get_node_from_net_and_compute_id(i, translation_node_id)
            for i in range(len(uorf_networks))
        ]
        signatures = [vnode.type_signature for vnode in vnodes]
        assert all([s == "translation_1_1" for s in signatures]), f"Signatures: {signatures}"
        logger.info(f"Extracted {len(uorf_networks)} translation nodes: {vnodes}")

        # print all signature of basic plotted nodes:
        for k, node_id in basic_node_ids.items():
            vnode = uorf_pred.network_model._stack.get_node_from_net_and_compute_id(0, node_id)
            logger.debug(f"Node {node_id} signature ({k}): {vnode.type_signature}")

        all_data = uorf_pred.get_data()
        translation_data = all_data[: len(uorf_networks)]
        basic_data = all_data[len(uorf_networks) :]

        logger.info(f"Extracted {len(translation_data)} translation nodes: {translation_data}")

        result_data['translation_data'] = translation_data
        result_data['uorf_names'] = uorf_names
        result_data['uorf_embeddings'] = uorf_embeddings
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

    def create_visualization(self, data, figax):
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

            ern_data = data.get('ern_data', [])
            ern_names = data.get('ern_names', [])
            ern_embeddings = data.get('ern_embeddings', {})
            num_ern_to_show = min(4, len(ern_names))

            if len(ern_data) > 0:
                # create equal columns for all plots in row 1 (ERN nodes + embeddings)
                row1_gs = row1_fig.add_gridspec(
                    1, num_ern_to_show + 2, width_ratios=[1] * num_ern_to_show + [0.1, 0.5]
                )

                assert len(ern_data) == len(ern_names)
                assert isinstance(ern_data[0].metadata['full_y'], list)

                vlims = [
                    np.min([np.min(data.metadata['full_y'][0]) for data in ern_data]),
                    np.max([np.max(data.metadata['full_y'][0]) for data in ern_data]),
                ]

                sc = None
                for i in range(num_ern_to_show):
                    ax = row1_fig.add_subplot(row1_gs[0, i])
                    name = ern_names[i]
                    ern_data_item = ern_data[i]

                    X = ern_data_item.metadata['full_x']
                    assert isinstance(X, list)

                    x0 = X[0]
                    assert isinstance(x0, np.ndarray)
                    assert x0.shape[1] == 1, f'First ERN input is not 1D: {x0.shape}'

                    x1 = X[1]
                    assert isinstance(x1, np.ndarray)
                    assert x1.shape[1] == 1, f'Second ERN input is not 1D: {x1.shape}'

                    Y = ern_data_item.metadata['full_y']
                    assert isinstance(Y, list)
                    assert len(Y) == 1, f'ERN has more than one output: {len(Y)}'
                    y = Y[0]
                    assert isinstance(y, np.ndarray)
                    assert y.shape[1] == 1, f'ERN output is not 1D: {y.shape}'
                    assert y.shape == x0.shape, (
                        f'ERN output shape mismatch: {y.shape} != {x0.shape}'
                    )

                    self.add_subplot_letter(ax, chr(65 + i))

                    sc = ax.scatter(
                        x0.flatten(),
                        x1.flatten(),
                        c=y.flatten(),
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

                wspace = self.figure_spec.layout.wspace
                row1_fig.subplots_adjust(wspace=wspace, right=0.9)
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

            # row 2: simple inner nodes, translation, and uORF embeddings
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

                full_x = node_data.metadata['full_x']
                full_y = node_data.metadata['full_y']
                n_outputs = len(full_y)
                n_inputs = len(full_x)
                assert all([x.shape[1] == 1 for x in full_x]), 'Inputs are not 1D'
                assert all([y.shape[1] == 1 for y in full_y]), 'Outputs are not 1D'

                if n_outputs > 1:
                    if n_inputs == 1:  # single input, multiple outputs
                        allX = [full_x[0].flatten()[:, None]] * n_outputs
                    else:
                        assert n_inputs == n_outputs, (
                            f"n_inputs: {n_inputs}, n_outputs: {n_outputs}"
                        )
                        # multiple inputs, multiple outputs, we treat them as separate pairs
                        allX = [x.flatten()[:, None] for x in full_x]
                    allY = [y.flatten()[:, None] for y in full_y]
                    legends = [f'out {i}' for i in range(n_outputs)]
                else:
                    if n_inputs > 1:  # multiple inputs, single output
                        raise ValueError(
                            f'Multiple inputs, single output not supported: node {i} of {node_type} has {n_inputs} inputs and {n_outputs} outputs'
                        )

                    else:  # single input, single output
                        allX = [full_x[0].flatten()[:, None]]
                        allY = [full_y[0].flatten()[:, None]]
                        legends = ['']

                colors = truncated_cmap(np.linspace(0, 0.7, len(allX)))
                xshapes = [x.shape for x in allX]
                yshapes = [y.shape for y in allY]
                npoints = len(allX[0])
                logger.debug(
                    f'Plotting {len(allX)} inputs and {len(allY)} outputs, {npoints} points'
                )
                logger.debug(f'X shapes: {xshapes}, Y shapes: {yshapes}')
                for j, (X, Y, c, legend) in enumerate(zip(allX, allY, colors, legends)):
                    ax.scatter(X, Y, s=4, alpha=0.05, linewidth=0, color=c)

                    logger.debug(
                        f'Plotting inputs with shape {X.shape} and outputs with shape {Y.shape}'
                    )

                    # calculate trend line using KNN
                    xquery_min, xquery_max = X.min(), X.max()
                    res = 200
                    xquery = np.linspace(xquery_min, xquery_max, res).reshape(-1, 1)

                    z, _ = knn_stats(
                        xquery,
                        Y,
                        tree=build_tree(X),
                        avg_method="mean",
                        min_points=500,
                        k=1000,
                        radius=0.1,
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

            logger.debug(f"Plotting translation nodes with {len(translation_data)} uORFs")

            ax_trans = row2_fig.add_subplot(row2_gs[0, num_basic_nodes])

            self.add_subplot_letter(ax_trans, chr(letter_idx))
            letter_idx += 1

            colors = truncated_cmap(np.linspace(0, 1, len(translation_data)))

            for i in range(len(translation_data)):
                color = colors[i]
                full_x = translation_data[i].metadata['full_x']
                full_y = translation_data[i].metadata['full_y']
                n_outputs = len(full_y)
                n_inputs = len(full_x)
                assert all([x.shape[1] == 1 for x in full_x]), 'Inputs are not 1D'
                assert all([y.shape[1] == 1 for y in full_y]), 'Outputs are not 1D'
                assert n_inputs == 1, f"Translation node has {n_inputs} inputs"
                assert n_outputs == 1, f"Translation node has {n_outputs} outputs"

                X = full_x[0]
                Y = full_y[0]

                logger.debug(f'Plotting translation node {uorf_names[i]} with {X.shape[0]} points')
                ax_trans.scatter(X, Y, s=1, alpha=0.05, linewidth=0, color=color)

                xquery_min, xquery_max = X.min(), X.max()
                res = 200
                xquery = np.linspace(xquery_min, xquery_max, res).reshape(-1, 1)
                z = knn_stats(
                    xquery, Y, tree=build_tree(X), stats="mean", min_points=500, k=5000, radius=0.05
                )

                ax_trans.plot(xquery, z, linewidth=2, color=color, label=f"{uorf_names[i]}")

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

            wspace = self.figure_spec.layout.wspace
            row1_fig.subplots_adjust(wspace=wspace, right=0.9)
            row2_fig.subplots_adjust(wspace=wspace, right=0.9)

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
