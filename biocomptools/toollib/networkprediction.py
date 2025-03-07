from pydantic.functional_validators import BeforeValidator
from typing import Any, Optional, List, Union, Dict, Annotated, Literal, Tuple, TypeAlias, Callable
import numpy as np
import jax.numpy as jnp
from biocomp.plotutils import PlotData, LazyPlotData, get_reordered_protein_names
from biocomptools.toollib.common import make_pretty_input_names
from biocomptools.modelmodel import NetworkModel, BiocompModel, NodeSpec
from biocomptools.logging_config import get_logger
from biocomptools.toollib.datasources import DataSource
from pathlib import Path


logger = get_logger(__name__)

NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]


def validate_predict_at(v: Any) -> List[np.ndarray]:
    """convert single numpy array to list of arrays"""
    if isinstance(v, np.ndarray):
        return [v]
    return v


def validate_ground_truth(v: Any) -> Optional[List[Optional[np.ndarray]]]:
    """convert single numpy array to list of arrays or none to list of none"""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return [v]
    return v


def validate_input_order(v: Any) -> Optional[List[List[int]]]:
    """convert list of ints to list of lists of ints"""
    if v is None:
        return None
    if isinstance(v, list) and all(isinstance(x, int) for x in v):
        return [v]
    return v


class NetworkPrediction(DataSource):
    """
    Performs predictions using a networkmodel and prepares data for plotting

    responsible for:
    - preparing input data for prediction
    - making predictions using the network model
    - comparing predictions to ground truth
    - generating plot data for visualization
    """

    predict_at: Annotated[Union[np.ndarray, List[np.ndarray]], BeforeValidator(validate_predict_at)]
    network_model: NetworkModel
    input_order: Annotated[Optional[List[List[int]]], BeforeValidator(validate_input_order)] = None
    ground_truth: Annotated[
        Optional[Union[np.ndarray, List[Optional[np.ndarray]]]],
        BeforeValidator(validate_ground_truth),
    ] = None

    collection_points: Optional[List[NodeSpec]] = None  # collection points to examine inner nodes

    seed: int = 0
    max_evals: int = 300000
    use_output_as_input: bool = False
    z_value: Union[Literal['uniform'], float] = 'uniform'
    disable_variational: bool = True

    save_csv_to: Optional[str] = None  # save prediction statistics to a CSV file

    already_latent: bool = False  # no need to rescale if the input data is already in latent space

    metadata: Dict[str, Any] = {}

    _yhats: Optional[List[np.ndarray]] = None
    _x: Optional[List[np.ndarray]] = None
    _gtruths: Optional[List[Optional[np.ndarray]]] = None
    _aligned_x: Optional[List[np.ndarray]] = None
    _aligned_ground_truth: Optional[List[Optional[np.ndarray]]] = None
    _collection_in: Optional[np.ndarray] = None
    _collection_out: Optional[np.ndarray] = None
    _collection_input_offsets: Optional[List[int]] = None
    _collection_output_offsets: Optional[List[int]] = None
    _network_stats: Optional[List[Dict[str, Any]]] = None

    def model_post_init(self, *args, **kwargs):
        """initialize the model after validation"""
        super().model_post_init(*args, **kwargs)

        logger.debug(
            f"predict_at: {len(self.predict_at)} arrays, shapes: {[x.shape for x in self.predict_at]}"
        )

        assert isinstance(self.network_model.network, list)

        # validate number of networks matches input data
        if len(self.predict_at) != len(self.network_model.network):
            raise ValueError(
                f"number of predict_at arrays ({len(self.predict_at)}) "
                f"does not match number of networks ({len(self.network_model.network)})"
            )

        # normalize ground truth
        if self.ground_truth is None:
            self.ground_truth = [None] * len(self.predict_at)
        else:
            if len(self.ground_truth) != len(self.predict_at):
                raise ValueError(
                    f"number of ground truth arrays ({len(self.ground_truth)}) "
                    f"does not match number of predict_at arrays ({len(self.predict_at)})"
                )

        # shuffle inputs with fixed random seed
        self._shuffle_inputs()

        # prepare aligned inputs
        self._aligned_x, self._aligned_ground_truth = self._prepare_inputs()
        logger.debug(f"aligned_x shapes: {[x.shape for x in self._aligned_x]}")
        logger.debug(
            f"aligned_ground_truth shapes: {[None if gt is None else gt.shape for gt in self._aligned_ground_truth]}"
        )

    def with_shared_from_model(self, model: BiocompModel) -> 'NetworkPrediction':
        """create a new networkprediction with a different model"""
        logger.debug(f"creating new networkprediction with model {model.signature()=}")

        # create new instance with updated network model
        new = self.model_copy(update={'network_model': self.network_model.with_model(model)})
        new._yhats = None  # clear the cache
        new._network_stats = None  # clear the stats cache

        logger.debug(
            f"created networkprediction with model signature {new.network_model.model.signature()=}"
        )
        return new

    def with_csv_output_path(self, path: str) -> 'NetworkPrediction':
        """set path to save prediction statistics to a CSV file"""
        return self.model_copy(update={'save_csv_to': path})

    def _shuffle_inputs(self):
        """shuffle inputs and ground truth with the same random order"""
        # set random seed for reproducibility
        rng = np.random.RandomState(self.seed)

        new_predict_at = []
        new_gt = []
        for i, (x, gt) in enumerate(zip(self.predict_at, self.ground_truth)):
            order = rng.permutation(len(x))
            new_predict_at.append(x[order])
            new_gt.append(gt[order] if gt is not None else None)
        self.predict_at = new_predict_at
        self.ground_truth = new_gt

    def _prepare_inputs(self) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        """prepare inputs by padding or truncating to the same length"""
        max_prediction_length = max(len(x) for x in self.predict_at)
        logger.debug(f"max_prediction_length across networks: {max_prediction_length}")

        effective_max_evals = min(
            self.max_evals if self.max_evals > 0 else max_prediction_length, max_prediction_length
        )
        logger.debug(f"effective_max_evals: {effective_max_evals}")

        aligned_predict_at = []
        aligned_ground_truth = []

        for i, (x, gt) in enumerate(zip(self.predict_at, self.ground_truth)):
            logger.debug(
                f"aligning network {i}: x.shape={x.shape}, gt={None if gt is None else gt.shape}"
            )

            if len(x) < effective_max_evals:
                # pad with zeros if shorter than desired length
                f"padding prediction queries for network {i} from {len(x)} to {effective_max_evals} points"
                zeros = np.zeros((effective_max_evals - len(x), x.shape[1]))
                padded_x = np.vstack([x, zeros])
                aligned_predict_at.append(padded_x)

                if gt is not None:
                    gtzeros = np.zeros((effective_max_evals - len(x), gt.shape[1]))
                    padded_gt = np.vstack([gt, gtzeros])
                    aligned_ground_truth.append(padded_gt)
                else:
                    aligned_ground_truth.append(None)
            else:
                # truncate if longer than desired length
                logger.debug(
                    f"truncating prediction queries for network {i} from {len(x)} to {effective_max_evals} points"
                )
                truncated_x = x[:effective_max_evals]
                aligned_predict_at.append(truncated_x)

                if gt is not None:
                    truncated_gt = gt[:effective_max_evals]
                    aligned_ground_truth.append(truncated_gt)
                else:
                    aligned_ground_truth.append(None)

        return aligned_predict_at, aligned_ground_truth

    def _compute_collection_indices(self):
        """compute index offsets for each collection point"""
        input_offsets = [0]
        output_offsets = [0]
        assert self.collection_points is not None

        for point in self.collection_points:
            input_indices, output_indices = self.network_model.get_node_indices(
                point.network_id, point.node_id
            )

            total_input_indices = sum(len(idx) for idx in input_indices)
            total_output_indices = sum(len(idx) for idx in output_indices)

            input_offsets.append(input_offsets[-1] + total_input_indices)
            output_offsets.append(output_offsets[-1] + total_output_indices)

        return input_offsets, output_offsets

    def compute_all_network_predictions(self):
        """compute predictions for all networks"""
        logger.debug(f"computing predictions with model {self.network_model.signature()}")
        logger.debug(
            f"prediction params: seed={self.seed}, disable_variational={self.disable_variational}, z_value={self.z_value}"
        )

        # stack inputs from all networks
        assert isinstance(self._aligned_x, list)
        stacked_x = np.column_stack(self._aligned_x)

        effective_max_evals = len(self._aligned_x[0])
        logger.debug(f"effective_max_evals: {effective_max_evals}")

        predict_f = (
            self.network_model.predict
            if self.already_latent
            else self.network_model.predict_unscaled
        )

        stacked_yhats, collection_values = predict_f(
            stacked_x,
            key=self.seed,
            collection_points=self.collection_points,
            z_value=self.z_value,
            disable_variational=self.disable_variational,
        )

        self._collection_in, self._collection_out = collection_values

        if self.collection_points and self._collection_in is not None:
            self._collection_input_offsets, self._collection_output_offsets = (
                self._compute_collection_indices()
            )

        # split the outputs by network
        network_outputs = self.network_model.split_outputs_per_network(
            stacked_yhats, effective_max_evals
        )

        self._process_prediction_results(network_outputs, effective_max_evals)

        # calculate and store network statistics
        self._network_stats = self._calculate_all_network_stats()

    def _create_collection_xy_function(
        self,
        collection_idx: int,
        input_start: int,
        input_end: int,
        output_start: int,
        output_end: int,
    ) -> Callable[[PlotData], Tuple[np.ndarray, np.ndarray]]:
        """create a function to get x and y data for a specific collection point"""

        def get_xy(pdata: PlotData) -> Tuple[np.ndarray, np.ndarray]:
            logger.debug(f"getting xy for collection point {collection_idx}")

            # compute predictions if not already computed
            if not hasattr(self, '_collection_in') or self._collection_in is None:
                self.compute_all_network_predictions()

            if self._collection_in is None or self._collection_out is None:
                raise ValueError("No collection data available")

            # extract the specific collection point data
            x = self._collection_in[:, input_start:input_end]

            all_y = self._collection_out[:, output_start:output_end]

            # for compatibility with PlotData, we need to pick just one column from y
            y = all_y[:, 0:1]

            logger.debug(f"x shape: {x.shape}, y shape: {y.shape}, all_y shape: {all_y.shape}")

            pdata.metadata['full_y'] = all_y

            return x, y

        return get_xy

    def _process_prediction_results(self, network_outputs: List[np.ndarray], max_evals: int):
        """process and store prediction results"""
        self._x = []
        self._yhats = []
        self._gtruths = []
        assert isinstance(self.ground_truth, list)
        assert isinstance(self._aligned_ground_truth, list)

        for i, (network_output, x) in enumerate(zip(network_outputs, self.predict_at)):
            effective_max_evals = min(max_evals, len(x))
            logger.debug(f"processing network {i}, effective_max_evals: {effective_max_evals}")

            # store prediction results
            truncated_output = network_output[:effective_max_evals]
            self._yhats.append(truncated_output)
            logger.debug(f"stored yhat shape: {truncated_output.shape}")

            # store ground truth if available
            if self.ground_truth[i] is not None:
                truncated_gt = self._aligned_ground_truth[i][:effective_max_evals]
                self._gtruths.append(truncated_gt)
                logger.debug(f"stored ground truth shape: {truncated_gt.shape}")
            else:
                self._gtruths.append(None)
                logger.debug("stored ground truth: None")

            # store inputs
            truncated_x = x[:effective_max_evals]
            self._x.append(truncated_x)
            logger.debug(f"stored x shape: {truncated_x.shape}")

        # validate results
        assert len(self._x) == len(self.predict_at)
        assert len(self._x) == len(self.network_model.network)

    def _create_xy_function(
        self, network_idx: int
    ) -> Callable[[PlotData], Tuple[np.ndarray, np.ndarray]]:
        """create a function to get x and y data for a specific network"""

        def get_xy(pdata: PlotData) -> Tuple[np.ndarray, np.ndarray]:
            logger.debug(
                f"getting xy for network {network_idx} with model {self.network_model.signature()}"
            )

            # compute predictions if not already computed
            if not hasattr(self, '_yhats') or self._yhats is None:
                logger.debug("computing all network predictions")
                self.compute_all_network_predictions()
                if self.save_csv_to:
                    logger.debug("saving prediction statistics to CSV")
                    self.save_csv()
                else:
                    logger.debug("no save_csv_to path provided")

            assert isinstance(self._x, list)
            assert isinstance(self._yhats, list)
            assert isinstance(self._gtruths, list)

            x = self._x[network_idx]
            yhat = self._yhats[network_idx]
            gt = self._gtruths[network_idx]

            logger.debug(
                f"x shape: {x.shape}, yhat shape: {yhat.shape}, gt: {None if gt is None else gt.shape}"
            )

            assert isinstance(self.network_model.network, list)

            network = self.network_model.network[network_idx].network

            # get output position for this network
            _, output_pos, _, _ = get_reordered_protein_names(network)
            assert isinstance(output_pos, int)
            logger.debug(f"output_pos: {output_pos}")

            stats = self.get_network_stats()
            assert isinstance(stats, list) and (len(stats) == len(self.network_model.network))
            pdata.metadata['prediction_stats'] = stats[network_idx].copy()
            pdata.metadata.update(self.metadata)

            if self.use_output_as_input:
                logger.debug("using output as input, returning yhat as both x and y")
                return yhat, yhat
            else:
                logger.debug("using original input, returning x and yhat")
                return x, yhat

        return get_xy

    def _log_stats(
        self,
        network_idx: int,
        prediction_stats: Dict[str, Any],
        latent_gt: np.ndarray,
        latent_yhat: np.ndarray,
        latent_x: np.ndarray,
    ):
        n_samples = 15

        sample_ids = np.random.choice(len(latent_gt), n_samples, replace=False)
        gt_sample = latent_gt[sample_ids]
        yhat_sample = latent_yhat[sample_ids]
        latentx_sample = latent_x[sample_ids]
        se_sample = (yhat_sample - gt_sample) ** 2

        original_yhat = self.network_model.model.rescaler.inv(yhat_sample)
        original_gt = self.network_model.model.rescaler.inv(gt_sample)
        unscaledx_sample = self.network_model.model.rescaler.inv(latentx_sample)
        unscaledx_sample = [tuple(np.round(x, 3) for x in xs) for xs in unscaledx_sample]
        latentx_sample = [tuple(np.round(x, 3) for x in xs) for xs in latentx_sample]
        import pandas as pd

        df = pd.DataFrame(
            {
                'unscaled X': unscaledx_sample,
                'unscaled gt': original_gt,
                'unscaled yhat': original_yhat,
                'latent X': latentx_sample,
                'latent gt': gt_sample,
                'latent yhat': yhat_sample,
                'l2 error': se_sample,
            }
        )

        logger.debug(
            f"""network {network_idx} evaluated over {prediction_stats['samples']} samples:
                - mse: {prediction_stats['mse']:.3f}
                - rmse: {prediction_stats['rmse']:.3f}
                - mean: {prediction_stats['latent_mean']:.3f}
                - std: {prediction_stats['latent_std']:.3f}
                - min: {prediction_stats['latent_min']:.3f}
                - max: {prediction_stats['latent_max']:.3f}
                """
        )

        logger.debug(f"Random samples:\n{df.round(3).to_string()}")

    def _calculate_network_stats(
        self, network_idx, network, yhat, gt, output_pos, nb_points_in_eval
    ):
        """calculate statistics for a single network"""
        # transform to latent space for statistics
        rescaler = self.network_model.model.rescaler
        latent_yhats = np.asarray(rescaler.fwd(yhat))
        latent_yhat = latent_yhats[:, output_pos]

        # calculate basic statistics
        xp_name = None
        try:
            xp_name = network.recipe.experiment.name
        except AttributeError:
            pass
        recipe_name = None
        try:
            recipe_name = network.recipe.name
        except AttributeError:
            pass
        network_name = getattr(network, 'name', f"Network_{network_idx}")
        network_stats = {
            'xp_name': xp_name,
            'recipe_name': recipe_name,
            'network_name': network_name,
            'eval_npoints': nb_points_in_eval,
            'samples': yhat.shape[0],  # number of actual prediction points used
            'mse': None,
            'rmse': None,
            'latent_mean': float(latent_yhat.mean()),
            'latent_std': float(latent_yhat.std()),
            'latent_min': float(latent_yhat.min()),
            'latent_max': float(latent_yhat.max()),
        }

        # add comparison stats if ground truth available
        if gt is not None:
            latent_gt = np.asarray(rescaler.fwd(gt).flatten())
            mse = float(np.mean((latent_yhat - latent_gt) ** 2))
            network_stats['mse'] = mse
            network_stats['rmse'] = float(np.sqrt(mse))

            # log sample comparisons for debugging
            assert self._x is not None and len(self._x) > network_idx
            x = self._x[network_idx]
            latent_x = rescaler.fwd(x)
            self._log_stats(network_idx, network_stats, latent_gt, latent_yhat, latent_x)

        return network_stats

    def _calculate_all_network_stats(self) -> List[Dict[str, Any]]:
        """calculate statistics for all networks and return as a list of dictionaries"""
        all_stats = []

        for i, network in enumerate(self.network_model.network):
            # extract network info
            network_name = getattr(network, 'name', f"Network_{i}")
            nb_points_in_eval = len(self.predict_at[i])

            # get prediction data
            yhat = self._yhats[i]
            gt = self._gtruths[i]

            # get output position
            output_pos = 0
            try:
                _, output_pos, _, _ = get_reordered_protein_names(network.network)
            except ValueError:
                continue

            # calculate statistics in latent space
            network_stats = self._calculate_network_stats(
                i, network, yhat, gt, output_pos, nb_points_in_eval
            )
            all_stats.append(network_stats)

        return all_stats

    def get_network_stats(self):
        """get statistics for all networks, computing if necessary"""
        if not hasattr(self, '_network_stats') or self._network_stats is None:
            if not hasattr(self, '_yhats') or self._yhats is None:
                self.compute_all_network_predictions()
            else:
                # just calculate stats if predictions already exist
                self._network_stats = self._calculate_all_network_stats()

        return self._network_stats

    def save_csv(self):
        """save prediction statistics to a CSV file"""
        logger.debug("saving prediction statistics to CSV")
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is required to save prediction statistics to CSV")
            return

        stats = self.get_network_stats()
        df = pd.DataFrame(stats)

        # update df with content of self.extra_metadata
        for key, value in self.metadata.items():
            df[key] = value

        logger.debug(f"prediction statistics DataFrame:\n{df.to_string()}")

        try:
            save_to = Path(self.save_csv_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_to, index=False)
            logger.info(f"saved prediction statistics to {save_to}")
        except Exception as e:
            logger.error(f"failed to save CSV to {self.save_csv_to}: {e}")

    def _normalize_input_order(self) -> List[Optional[List[int]]]:
        """normalize input_order to ensure it's consistent across networks"""
        assert self.network_model is not None
        assert isinstance(self.network_model.network, list)

        if self.input_order is None:
            logger.debug(
                f"input_order is None, using None for all {len(self.network_model.network)} networks"
            )
            return [None] * len(self.network_model.network)

        if isinstance(self.input_order, list):
            # single input order for all networks
            if all(isinstance(x, int) for x in self.input_order):
                logger.debug(f"using same input_order {self.input_order} for all networks")
                return [self.input_order] * len(self.network_model.network)

            if len(self.input_order) != len(self.network_model.network):
                raise ValueError(
                    f"input order list has {len(self.input_order)} items but there are "
                    f"{len(self.network_model.network)} networks"
                )
            return self.input_order

        raise ValueError(f"unexpected input_order type: {type(self.input_order)}")

    def get_data_lazy(self) -> List[LazyPlotData]:
        """get plot data in lazy evaluation mode"""
        logger.debug(f"getting data lazily for model {self.network_model.signature()}")

        # handle collection points if provided
        if self.collection_points is not None and len(self.collection_points) > 0:
            return self._get_collection_data_lazy()

        # otherwise, continue with normal network data
        plot_data_list = []
        input_order = self._normalize_input_order()
        logger.debug(f"normalized input_order: {input_order}")

        # create plot data for each network
        for i, network in enumerate(self.network_model.network):
            metadata = self._create_network_metadata(i, network)
            plot_data = self._extract_plot_data(i, network, input_order[i], metadata)

            network_info = metadata['network_info']
            pretty_inputs = make_pretty_input_names(
                network_info['cotx'],
                plot_data.input_names,
            )
            plot_data.metadata['pretty_inputs'] = pretty_inputs

            plot_data_list.append(plot_data)

        logger.debug(f"returning {len(plot_data_list)} plot data objects")
        return plot_data_list

    def _get_collection_data_lazy(self) -> List[LazyPlotData]:
        """get plot data for collection points in lazy evaluation mode"""
        logger.debug(f"getting collection data lazily for model {self.network_model.signature()}")

        plot_data_list = []

        # compute index offsets for each collection point if needed
        if not hasattr(self, '_collection_input_offsets') or self._collection_input_offsets is None:
            if not hasattr(self, '_collection_in') or self._collection_in is None:
                self.compute_all_network_predictions()

        if self._collection_in is None or self._collection_out is None:
            logger.warning("No collection data available, returning empty list")
            return []

        # create plot data for each collection point
        for i, collection_point in enumerate(self.collection_points):
            metadata = {
                'source_type': 'collection',
                'seed': self.seed,
                'model_signature': self.network_model.model.signature(),
                'collection_point_index': i,
                'network_id': collection_point.network_id,
                'node_id': collection_point.node_id,
            }

            assert isinstance(self._collection_input_offsets, list)
            assert isinstance(self._collection_output_offsets, list)

            input_dim = self._collection_input_offsets[i + 1] - self._collection_input_offsets[i]

            # create function to get data for this collection point
            get_xy_fn = self._create_collection_xy_function(
                i,
                self._collection_input_offsets[i],
                self._collection_input_offsets[i + 1],
                self._collection_output_offsets[i],
                self._collection_output_offsets[i + 1],
            )

            # create plot data with placeholder input and output names
            input_names = [
                f"node_{collection_point.network_id}_{collection_point.node_id}_{i}"
                for i in range(input_dim)
            ]
            output_name = f"node_{collection_point.network_id}_{collection_point.node_id}"

            plot_data = LazyPlotData(
                get_xy=get_xy_fn,
                input_names=input_names,
                output_name=output_name,
                metadata=metadata,
            )

            plot_data_list.append(plot_data)

        logger.debug(f"returning {len(plot_data_list)} collection plot data objects")
        return plot_data_list

    def _create_network_metadata(self, network_idx: int, network) -> Dict[str, Any]:
        """create metadata dictionary for a network"""
        metadata = self.metadata.copy()

        import biocomp.network

        network_info = biocomp.network.generate_network_info(network.network)

        metadata.update(
            {
                'source_type': 'prediction',
                'seed': self.seed,
                'model_signature': self.network_model.model.signature(),
                'network': network,
                'network_info': network_info,
                'n_predictions': len(self.predict_at[network_idx]),
                'network_index': network_idx,
            }
        )

        logger.debug(f"created metadata for network {network_idx}")
        return metadata

    def _extract_plot_data(
        self, network_idx: int, network, input_order: Optional[List[int]], metadata: Dict[str, Any]
    ) -> LazyPlotData:
        """extract plot data from a network"""
        import biocomp.plotutils as pu

        # create function to get data for this network
        get_xy_fn = self._create_xy_function(network_idx)

        # extract plot data
        plot_data = pu.extract_lazy_plot_data_from_network(
            network.network,
            get_xy_fn,
            input_order=input_order,
            metadata=metadata,
        )

        logger.debug(
            f"extracted plot data for network {network_idx}: {plot_data.input_names=}, {plot_data.output_name=}"
        )
        return plot_data

    def get_data(self) -> List[PlotData]:
        """get fully-evaluated plot data"""
        lazy_data = self.get_data_lazy()

        # evaluate all plot data
        for i, data in enumerate(lazy_data):
            data.set_xy()

        return lazy_data
