## {{{                          --     imports     --
from pydantic.functional_validators import BeforeValidator
from pydantic import BaseModel, model_validator
from typing import Any, Optional, List, Union, Dict, Annotated, Literal

import numpy as np

from biocomp.utils import (
    ArbitraryModel,
    load_lib,
)
from pathlib import Path
import biocomp as bc
from biocomp.plotutils import PlotData, LazyPlotData, get_reordered_protein_names
import biocomp.plotutils as pu

from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector, NetworkDataId

import biocomptools.toollib.common as cm
from biocomptools.toollib.common import maybetqdm
import biocomptools.toollib.models as md
from biocomptools.modelmodel import NetworkModel, BiocompModel
from biocomptools.logging_config import get_logger

from sqlmodel import Session, text
from sqlalchemy.sql.elements import TextClause

config = cm.config


def truncated_path(path: str | Path, max_len=50) -> str:
    if isinstance(path, Path):
        path = path.as_posix()
    if len(path) > max_len:
        return '...' + path[-max_len:]
    return path


def to_str(data: Any) -> Any:
    if not isinstance(data, str) and data is not None:
        return str(data)
    return data


logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     datasource     --


class DataSource(ArbitraryModel):
    metadata: dict = {}

    def get_data(self) -> List[PlotData]:
        raise NotImplementedError('Subclasses must implement get_data')


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     DB   --

config = cm.config

DEFAULT_NAME_LOOKUP = {
    'mNeonGreen': 'mNG',
    'PgU': 'Pgu',
}


def make_pretty_input_names(
    ratios,
    ordered_input_names,
    name_lookup: Optional[dict] = DEFAULT_NAME_LOOKUP,
):
    fluo_markers = [p[0][-1].upper() for p in ratios]

    names = []

    for _, p in enumerate(ordered_input_names):
        # x = rf"$X_{i+1}$ ({p})"
        x = ''
        if p.upper() in fluo_markers:
            idx = fluo_markers.index(p.upper())
            content = ' + '.join(ratios[idx][0][:-1])
            if content:
                x += rf"${content}$"
        else:
            # print(
            #     f"Fluo marker: {p}, not found in ratios {fluo_markers}. {ordered_input_names=}, {ratios=}, {name_lookup=}, {fluo_markers=}"
            # )
            ...

        if name_lookup is not None:
            for k, v in name_lookup.items():
                x = x.replace(k, v)

        names.append(x)

    return names


def to_text_clause(data: Any) -> TextClause:
    if isinstance(data, str):
        return text(data)
    return data


class DBSource(DataSource, NetworkSet):
    input_order: Optional[List[int]] = None

    @model_validator(mode='before')
    def validate_content(cls, values):
        content = values.get('content')
        if isinstance(content, (NetworkSelector, NetworkDataId)):
            values['content'] = [content]
        elif isinstance(content, NetworkSet):
            values['content'] = content.content
        return values

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._lib = load_lib()
        self._engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path)
        with Session(self._engine) as session:
            self.run_selectors(session)

    @property
    def db_session(self):
        return Session(self._engine)

    @property
    def path_prefix(self):
        return Path(config.paths.root).expanduser().resolve()

    def data_from_network(self, network: md.Network, datafile: md.DataFile) -> Optional[PlotData]:
        try:
            network.build(self._lib)
        except Exception as e:
            logger.error(f"Error building network {network.name}: {e}")
            return None

        actual_network = network._network
        assert isinstance(actual_network, bc.network.Network)

        datafile_path = Path(self.path_prefix / datafile.file).expanduser().resolve()
        metadata = self.metadata
        metadata['network'] = network
        metadata['built_network'] = actual_network
        metadata['network_info'] = network.network_info
        metadata['source_type'] = 'DB'

        metadata = {**metadata, **network.model_dump()}
        metadata['datafile'] = datafile
        metadata['file_stem'] = datafile_path.stem

        if not datafile_path.exists():
            raise ValueError(f'Data file {datafile_path} does not exist for network {network.name}')

        def get_XY(_):
            logger.debug(f"DBSource: getting XY data for network {network.name}")
            try:
                X, Y = bc.recipe.get_network_XY(actual_network, datafile_path)
            except Exception as e:
                logger.error(f"Error getting XY data for network {network.name}: {e}")
                return None, None
            assert isinstance(X, np.ndarray)
            assert isinstance(Y, np.ndarray)
            return X, Y

        try:
            pdata = pu.extract_lazy_plot_data_from_network(
                actual_network,
                get_XY,
                input_order=self.input_order,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Error extracting data from network {network.name}: {e}")
            return None

        pdata.metadata['pretty_inputs'] = make_pretty_input_names(
            metadata['network_info']['cotx'],
            pdata.input_names,
        )

        return pdata

    def get_data(self) -> List[PlotData]:
        with self.db_session as session:
            data = self.get_networks_and_data(session)
            if not data:
                msg = f'No data found for {self.content}'
                raise ValueError(msg)
            networks, datafiles = zip(*data)

            res = []

            for n, f in maybetqdm(zip(networks, datafiles), desc='Loading recipes'):
                try:
                    d = self.data_from_network(n, f)
                except Exception as e:
                    logger.error(f"Error loading data for {n.name}: {e}")
                    d = None

                if d is not None:
                    res.append(d)

            return res


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                     --     NetworkPrediction     --


def validate_predict_at(v):
    if isinstance(v, np.ndarray):
        return [v]
    return v


def validate_ground_truth(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return [v]
    return v

def validate_input_order(v):
    # if it's just a list of ints, turn it into a list of lists of ints
    if isinstance(v, list) and all(isinstance(x, int) for x in v):
        return [v]

class NetworkPrediction(DataSource):
    predict_at: Annotated[Union[np.ndarray, List[np.ndarray]], BeforeValidator(validate_predict_at)]
    network_model: NetworkModel
    input_order: Annotated[Optional[List[List[int]]], BeforeValidator(validate_input_order)] = None
    ground_truth: Annotated[
        Optional[np.ndarray | list[Optional[np.ndarray]]], BeforeValidator(validate_ground_truth)
    ] = None
    seed: int = 0

    max_evals: int = 300000

    # Networks always output their input (after they've been through inverse transform + forward again).
    # Setting this to True will use the outputed values as input in the plot data, instead of the original inputs.
    use_output_as_input: bool = False

    _yhats: Optional[np.ndarray] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)

        if len(self.predict_at) != len(self.network_model.network):
            raise ValueError(
                f"Number of predict_at arrays ({len(self.predict_at)}) "
                f"does not match number of networks ({len(self.network_model.network)})"
            )

        if self.ground_truth is not None:
            if len(self.ground_truth) != len(self.predict_at):
                raise ValueError(
                    f"Number of ground truth arrays ({len(self.ground_truth)}) "
                    f"does not match number of predict_at arrays ({len(self.predict_at)})"
                )
        else:
            self.ground_truth = [None] * len(self.predict_at)

        # shuffle predict_at (and ground_truth)
        new_predict_at = []
        new_gt = []
        for x, gt in zip(self.predict_at, self.ground_truth):
            order = np.random.permutation(len(x))
            new_predict_at.append(x[order])
            new_gt.append(gt[order] if gt is not None else None)
        self.predict_at = new_predict_at
        self.ground_truth = new_gt

        self._aligned_x, self._aligned_ground_truth = self._prepare_inputs()

    def with_shared_from_model(self, model: BiocompModel) -> 'NetworkPrediction':
        logger.debug(f"Creating new NetworkPrediction with model {model.signature()=}")
        logger.debug(f"Old model signature: {self.network_model.model.signature()=}")
        new = self.model_copy(update={'network_model': self.network_model.with_model(model)})
        new._yhats = None  # Clear the cache
        logger.debug(
            f"Created NetworkPrediction with model signature {new.network_model.model.signature()=}"
        )
        return new

    def _prepare_inputs(self):
        """pad or truncate inputs to the same length"""

        max_prediction_length = max(len(x) for x in self.predict_at)

        if self.max_evals < 0:
            self.max_evals = max_prediction_length

        self.max_evals = min(self.max_evals, max_prediction_length)

        aligned_predict_at = []
        aligned_ground_truth = []

        for x, gt in zip(self.predict_at, self.ground_truth):  # type: ignore
            if len(x) < self.max_evals:
                zeros = np.zeros((self.max_evals - len(x), x.shape[1]))
                aligned_predict_at.append(np.vstack([x, zeros]))
                gtzeros = np.zeros((self.max_evals - len(x), gt.shape[1]))
                aligned_ground_truth.append(np.vstack([gt, gtzeros]) if gt is not None else None)
            else:  # If input is larger than resample_to, truncate it
                aligned_predict_at.append(x[: self.max_evals])
                aligned_ground_truth.append(gt[: self.max_evals] if gt is not None else None)

        return aligned_predict_at, aligned_ground_truth

    def _split_yhat_per_network(self, yhat: np.ndarray):
        """Takes the whole stack output and returns a list of per-network outputs"""
        self._x = []
        self._yhats = []
        self._gtruths = []

        logger.debug(f"Going to split a full yhat of shape {yhat.shape}")

        output_start_id = 0
        for i, x in enumerate(self.predict_at):
            _, output_shapes = self.network_model._stack.get_network_output_indices(i)
            assert isinstance(output_shapes, list)
            outputs = []
            for output_shape in output_shapes:
                nout = np.prod(output_shape)
                output = yhat[:, output_start_id : output_start_id + nout].reshape(
                    -1, *output_shape
                )
                outputs.append(output)
                output_start_id += nout

            # assumes all outputs are of same shape
            assert all(output.shape == outputs[0].shape for output in outputs)
            network_i_outputs = np.concatenate(outputs, axis=1)

            assert (
                network_i_outputs.shape[0] == self.max_evals
            ), f"Expected {self.max_evals} but got {len(network_i_outputs)}"

            # truncate to remove padding
            network_i_outputs = network_i_outputs[: min(len(x), self.max_evals)]

            self._yhats.append(network_i_outputs)
            if self.ground_truth is not None and self.ground_truth[i] is not None:
                self._gtruths.append(self._aligned_ground_truth[i][: len(network_i_outputs)])
            else:
                self._gtruths.append(None)

            self._x.append(x[: min(len(x), self.max_evals)])

        assert len(self._x) == len(self.predict_at)
        assert len(self._x) == len(self.network_model.network)

    def compute_all_network_predictions(self):
        # self._aligned_x is just all self.predict_at
        # but with proper padding/truncation so that they all have the same length
        # TODO:
        # we need to make sure the predict_at are given in the correct input order!!!!!!

        stacked_x = np.column_stack(self._aligned_x)
        stacked_yhats = self.network_model.predict_unscaled(
            X=stacked_x,
            key=self.seed,
        )

        self._split_yhat_per_network(stacked_yhats)

    def get_data_lazy(self) -> List[LazyPlotData]:
        logger.debug(f"Getting data lazily for model {self.network_model.signature()}")

        def make_get_XY(network_idx):
            logger.debug(
                f"Making get_XY for network {network_idx} with model {self.network_model.signature()}"
            )

            def get_XY(pdata: PlotData):
                logger.debug(
                    f"Getting XY for network {network_idx} with model {self.network_model.signature()}"
                )
                if not hasattr(self, '_yhats') or self._yhats is None:
                    logger.debug(
                        f"Computing all network predictions with model {self.network_model.signature()}"
                    )
                    self.compute_all_network_predictions()

                logger.debug(f"Getting data for network {network_idx}. {len(self._x)=}")

                x = self._x[network_idx]
                yhat = self._yhats[network_idx]
                gt = self._gtruths[network_idx]

                network = self.network_model.network[network_idx].network

                _, output_pos, _, _ = get_reordered_protein_names(network)

                latent_yhats = np.asarray(self.network_model.model.rescaler.fwd(yhat))
                latent_yhat = latent_yhats[:, output_pos]

                latent_gt = (
                    np.asarray(self.network_model.model.rescaler.fwd(gt).flatten())
                    if gt is not None
                    else None
                )
                latent_x = np.asarray(self.network_model.model.rescaler.fwd(x))

                prediction_stats = {
                    'samples': yhat.shape[0],
                    'mean': float(latent_yhat.mean()),
                    'std': float(latent_yhat.std()),
                    'min': float(latent_yhat.min()),
                    'max': float(latent_yhat.max()),
                }

                if latent_gt is not None:
                    prediction_stats['mse'] = np.mean((latent_yhat - latent_gt) ** 2)
                    prediction_stats['rmse'] = float(np.sqrt(prediction_stats['mse']))

                    if prediction_stats['rmse'] > 0.18:
                        # dump to file
                        with open('/tmp/prediction_latent_gt.npy', 'wb') as f:
                            np.save(f, latent_gt)
                        with open('/tmp/prediction_latent_yhats.npy', 'wb') as f:
                            np.save(f, latent_yhats)
                        with open('/tmp/prediction_latent_yhat.npy', 'wb') as f:
                            np.save(f, latent_yhat)
                        with open('/tmp/prediction_latent_x.npy', 'wb') as f:
                            np.save(f, latent_x)

                for metric, value in prediction_stats.items():
                    logger.debug(f"Network {network_idx} predicted {metric}: {value}")

                pdata.metadata['prediction_stats'] = prediction_stats

                logger.debug(f"Returning data for network {network_idx}")
                return x, yhat
                # return yhat, yhat # in case we want to plot at predicted X

            return get_XY

        plot_data_list = []

        input_order = self.input_order
        if input_order is None:
            input_order = [None] * len(self.network_model.network)
        elif isinstance(input_order, (list, tuple, np.ndarray)):
            input_order = np.asarray(input_order)
            # check dimensions
            if input_order.ndim == 1:  # repeat for all networks
                input_order = np.tile(input_order, (len(self.network_model.network), 1))
            elif input_order.ndim == 2:  # use as is
                if input_order.shape[0] != len(self.network_model.network):
                    raise ValueError(
                        f"Input order has {input_order.shape[0]} rows but there are {len(self.network_model.network)} networks"
                    )
            input_order = input_order.tolist()

        assert isinstance(input_order, list)

        for i, network in enumerate(self.network_model.network):
            metadata = self.metadata.copy()
            metadata.update(
                {
                    'source_type': 'prediction',
                    'seed': self.seed,
                    'model_signature': self.network_model.model.signature(),
                    'network': network,
                    'network_info': network.network_info,
                    'built_network': network.network,
                    'n_predictions': len(self.predict_at[i]),
                    'network_index': i,
                }
            )

            plot_data = pu.extract_lazy_plot_data_from_network(
                network.network,
                make_get_XY(i),
                input_order=input_order[i],
                metadata=metadata,
            )

            plot_data_list.append(plot_data)

        return plot_data_list

    def get_data(self):
        return self.get_data_lazy()

        alld = self.get_data_lazy()
        for d in alld:
            d.set_xy()
        return alld
        # raise NotImplementedError('Use get_data_lazy instead')


##────────────────────────────────────────────────────────────────────────────}}}
