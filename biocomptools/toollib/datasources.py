## {{{                          --     imports     --
from pydantic.functional_validators import BeforeValidator
from pydantic import BaseModel, model_validator
from typing import Any, Optional, List, Union, Dict, Annotated
import time

import numpy as np

from biocomp.utils import (
    ArbitraryModel,
    load_lib,
)
from pathlib import Path
import biocomp as bc
from biocomp.plotutils import PlotData
import biocomp.plotutils as pu

from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector, NetworkDataId
import biocomptools.toollib.common as cm
from biocomptools.toollib.common import maybetqdm
import biocomptools.toollib.models as md
from biocomptools.modelmodel import NetworkModel
from biocomptools.logging_config import get_logger

from sqlmodel import Session, text
from sqlalchemy.sql.elements import TextClause


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
config = cm.config

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
    ratios, ordered_input_names, name_lookup: Optional[dict] = DEFAULT_NAME_LOOKUP
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
            print(
                f"Fluo marker: {p}, not found in ratios {fluo_markers}. {ordered_input_names=}, {ratios=}, {name_lookup=}, {fluo_markers=}"
            )

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
            try:
                X, Y = bc.recipe.get_network_XY(actual_network, datafile_path)
            except Exception as e:
                logger.error(f"Error getting XY data for network {network.name}: {e}")
                return None
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


class NetworkPrediction(DataSource):
    predict_at: Annotated[Union[np.ndarray, List[np.ndarray]], BeforeValidator(validate_predict_at)]
    network_model: NetworkModel
    input_order: Optional[list[int]] = None
    ground_truth: Annotated[
        Optional[np.ndarray | list[np.ndarray]], BeforeValidator(validate_ground_truth)
    ] = None
    seed: int = 0
    resample_to: Optional[int] = 50000

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

    def _align_inputs(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Align all predict_at arrays by either resampling to resample_to (or minimum length)"""
        rng = np.random.RandomState(self.seed)

        resample_to = self.resample_to or min(len(x) for x in self.predict_at)

        aligned_predict = []
        indices = []
        for x in self.predict_at:
            ids = rng.choice(len(x), size=resample_to, replace=(len(x) > resample_to))
            aligned_predict.append(x[ids])
            indices.append(ids)
        aligned_predict = np.column_stack(aligned_predict)

        aligned_ground_truth = None
        if self.ground_truth is not None:
            assert len(self.ground_truth) == len(self.predict_at) == len(indices)
            aligned_ground_truth = np.column_stack(
                [gt[ids] for gt, ids in zip(self.ground_truth, indices)]
            )

        logger.info(f"Resampled all arrays to size {resample_to}")

        return aligned_predict, aligned_ground_truth

    def _split_predictions(self, yhat: np.ndarray) -> List[np.ndarray]:
        """Split concatenated predictions back into separate arrays for each network."""
        outputs_per_network = [net.network.get_nb_outputs() for net in self.network_model.network]
        splits = []
        start_idx = 0
        for n_outputs in outputs_per_network:
            end_idx = start_idx + n_outputs
            splits.append(yhat[:, start_idx:end_idx])
            start_idx = end_idx
        return splits

    def get_data(self) -> List[PlotData]:
        logger.info(f"Predicting for {len(self.network_model.network)} networks")
        for i, predict in enumerate(self.predict_at):
            logger.info(f"Network {i}: {len(predict)} samples")

        # Align and predict
        aligned_x, aligned_ground_truth = self._align_inputs()
        t0 = time.time()
        all_yhats, overall_mse = self.network_model.predict_unscaled(
            aligned_x, self.seed, ground_truth=aligned_ground_truth
        )
        t1 = time.time()
        prediction_time = t1 - t0

        # Split predictions for each network
        yhats = self._split_predictions(all_yhats)

        plot_data_list = []

        # Create plot data for each network
        for i, (network, yhat, predict_at) in enumerate(
            zip(self.network_model.network, yhats, self.predict_at)
        ):
            metadata = self.metadata.copy()
            metadata.update(
                {
                    'source_type': 'prediction',
                    'seed': self.seed,
                    'model': self.network_model.model,
                    'network': network,
                    'network_info': network.network_info,
                    'built_network': network.network,
                    'n_predictions': len(predict_at),
                    'network_index': i,
                    'prediction_time': prediction_time,
                    'prediction_stats': {
                        'samples': yhat.shape[0],
                        'mean': float(yhat.mean()),
                        'std': float(yhat.std()),
                        'min': float(yhat.min()),
                        'max': float(yhat.max()),
                    },
                }
            )

            # Calculate network-specific MSE if ground truth is available
            if aligned_ground_truth is not None:
                network_gt = aligned_ground_truth[:, i : i + 1]
                network_mse = float(np.mean((yhat - network_gt) ** 2))
                metadata['mse'] = network_mse
                metadata['rmse'] = float(np.sqrt(network_mse))
                logger.info(f"Network {i} RMSE: {metadata['rmse']}")

            logger.info(f"Network {i} prediction shape: {yhat.shape}")
            logger.info(f"Network {i} stats: {metadata['prediction_stats']}")

            plot_data = pu.extract_plot_data_from_network(
                network.network,
                predict_at[: yhat.shape[0]],  # Ensure lengths match
                yhat,
                input_order=self.input_order,
                metadata=metadata,
            )
            plot_data_list.append(plot_data)

        logger.info(f"Total prediction time: {prediction_time}")
        return plot_data_list

    def get_data_lazy(self) -> List[PlotData]:
        def make_get_XY(network_idx):
            def get_XY(pdata):
                aligned_x, aligned_ground_truth = self._align_inputs()

                t0 = time.time()
                all_yhats, mse = self.network_model.predict_unscaled(
                    aligned_x, self.seed, ground_truth=aligned_ground_truth
                )
                t1 = time.time()

                # Split predictions for each network
                yhats = self._split_predictions(all_yhats)
                yhat = yhats[network_idx]

                prediction_stats = {
                    'samples': yhat.shape[0],
                    'mean': float(yhat.mean()),
                    'std': float(yhat.std()),
                    'min': float(yhat.min()),
                    'max': float(yhat.max()),
                }

                for metric, value in prediction_stats.items():
                    logger.info(f"Network {network_idx} predicted {metric}: {value}")

                pdata.metadata['prediction_time'] = t1 - t0
                pdata.metadata['prediction_stats'] = prediction_stats

                if aligned_ground_truth is not None:
                    network_mse = float(
                        np.mean(
                            (yhat - aligned_ground_truth[:, network_idx : network_idx + 1]) ** 2
                        )
                    )
                    pdata.metadata['mse'] = network_mse
                    pdata.metadata['rmse'] = float(np.sqrt(network_mse))
                    logger.info(f"Network {network_idx} RMSE: {pdata.metadata['rmse']}")

                return self.predict_at[network_idx][: yhat.shape[0]], yhat

            return get_XY

        plot_data_list = []

        for i, network in enumerate(self.network_model.network):
            metadata = self.metadata.copy()
            metadata.update(
                {
                    'source_type': 'prediction',
                    'seed': self.seed,
                    'model': self.network_model.model,
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
                input_order=self.input_order,
                metadata=metadata,
            )
            plot_data_list.append(plot_data)

        return plot_data_list


##────────────────────────────────────────────────────────────────────────────}}}
