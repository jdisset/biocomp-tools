from biocomp.library import PartsLibrary
from biocomp.recipe import get_network_XY
from tqdm import tqdm
from biocomp.datautils import DataConfig, DEFAULT_DATA_CONFIG, DataManager
from pydantic import BaseModel, Field, BeforeValidator, model_validator, ConfigDict
from sqlalchemy.orm import selectinload
from sqlalchemy import func
from sqlmodel import select, Session, col
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Annotated
import biocomptools.toollib.models as md
from biocomptools.logging_config import get_logger
from biocomptools.toollib.common import config
from sqlalchemy.exc import SQLAlchemyError


logger = get_logger(__name__)


## {{{                           --     utils     --
class NetworkDataId(BaseModel):
    """
    A network name and datafile path pair.
    The point is to keep it simple and explicit for repeatable training.
    """

    network_name: str
    file_path: str

    def __hash__(self):
        return hash((self.network_name, self.file_path))

    def fetch_network_and_datafile(self, session) -> Tuple[md.Network, md.DataFile]:
        logger.debug(f"Fetching network and datafile for {self.network_name} - {self.file_path}")
        network = session.exec(
            select(md.Network)
            .where(md.Network.name == self.network_name)
            .options(
                selectinload(md.Network.recipe).selectinload(md.Recipe.data_files),
                selectinload(md.Network.recipe).selectinload(md.Recipe.experiment),
                selectinload(md.Network.recipe).selectinload(md.Recipe.networks),
            )
        )
        networks = network.all()
        if len(networks) > 1:
            logger.error(f"Found multiple networks with name {self.network_name}. Taking first.")
        network = networks[0] if networks else None

        datafile = session.exec(
            select(md.DataFile)
            .where(md.DataFile.file == self.file_path)
            .options(selectinload(md.DataFile.calibration))
        )
        datafiles = datafile.all()
        if len(datafiles) > 1:
            logger.error(f"Found multiple datafiles with path {self.file_path}. Taking first.")
        datafile = datafiles[0] if datafiles else None

        assert network, f"No network found for {self.network_name}"
        assert datafile, f"No datafile found for {self.file_path}"

        return network, datafile


class Regex(str):
    def __new__(cls, string):
        return super().__new__(cls, string)


def maybe_regex(s: str) -> str:
    if isinstance(s, Regex):
        return s
    else:
        import re

        return re.escape(s)


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                         --     selector     --
class NetworkSelector(BaseModel):
    """
    Manually writing a NetworkAndData can be very annoying and verbose and error-prone.
    This class allows to batch select networks based on their names, recipes, experiments, etc.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    experiment_name: Optional[str | Regex] = None
    recipe_name: Optional[str | Regex] = None
    calibration_name: Optional[str | Regex] = None
    output_name: Optional[str] = None

    def get_networkdata_ids(self, session) -> List[NetworkDataId]:
        """
        Retrieve network data IDs based on specified filters.

        Args:
            session: The database session

        Returns:
            List[NetworkDataId]: List of network data identifiers

        Raises:
            ValueError: If no networks are found or if query execution fails
            SQLAlchemyError: For database-related errors
        """

        try:
            # Build the base query
            query = (
                select(md.Network)
                .join(md.Recipe)
                .join(md.Experiment)
                .options(
                    selectinload(md.Network.recipe).selectinload(md.Recipe.data_files),
                    selectinload(md.Network.recipe).selectinload(md.Recipe.experiment),
                    selectinload(md.Network.recipe).selectinload(md.Recipe.networks),
                )
            )

            # Apply filters
            if self.experiment_name:
                logger.debug(f"Applying experiment filter: {self.experiment_name}")
                if isinstance(self.experiment_name, Regex):
                    query = query.where(col(md.Experiment.name).regexp_match(self.experiment_name))
                else:
                    query = query.where(md.Experiment.name == self.experiment_name)

            if self.recipe_name:
                if isinstance(self.recipe_name, Regex):
                    # For regex, we'll just use the pattern directly since it might include custom matching
                    query = query.where(col(md.Recipe.name).regexp_match(self.recipe_name))
                else:
                    # For exact string match, we need to find the recipe with format "{exp_name}_{recipe_name}"
                    if self.experiment_name:
                        if isinstance(self.experiment_name, Regex):
                            # If experiment_name is a regex, we can't construct the exact recipe name
                            # So we'll need to use a regex that matches the end of the string
                            query = query.where(
                                col(md.Recipe.name).regexp_match(f".*_{self.recipe_name}$")
                            )
                        else:
                            # If we have exact experiment name, we can construct the full recipe name
                            query = query.where(
                                md.Recipe.name == f"{self.experiment_name}_{self.recipe_name}"
                            )
                    else:
                        # If no experiment name is provided, match any experiment prefix
                        query = query.where(
                            col(md.Recipe.name).regexp_match(f".*_{self.recipe_name}$")
                        )

            logger.debug(f"Final network query: {query}")

            # Execute query
            try:
                networks = session.exec(query).all()
            except SQLAlchemyError as e:
                logger.error(f"Database error while executing network query: {str(e)}")
                raise ValueError(f"Failed to execute network query: {str(e)}") from e

            if not networks:
                msg = f"No networks found for experiment '{self.experiment_name}', recipe '{self.recipe_name}. Query: {query}. Ignoring."
                logger.error(msg)

            # Filter by output name if specified
            if self.output_name is not None:
                original_count = len(networks)
                networks = [
                    network
                    for network in networks
                    if network.network_info['dependent_outputs'][0].upper()
                    == self.output_name.upper()
                ]

            # Process networks and collect data
            network_and_data = []

            for network in networks:
                logger.debug(f"Processing network: {network.name}")

                try:
                    if self.calibration_name:
                        logger.debug(f"Applying calibration filter: {self.calibration_name}")
                        datafile_query = (
                            select(md.DataFile)
                            .options(selectinload(md.DataFile.calibration))
                            .where(md.DataFile.recipe_name == network.recipe_name)
                        )

                        # Add calibration name filter based on type
                        if isinstance(self.calibration_name, Regex):
                            datafile_query = datafile_query.where(
                                col(md.DataFile.calibration_name).regexp_match(
                                    self.calibration_name
                                )
                            )

                            # # case insensitive version:
                            # datafile_query = datafile_query.where(
                            #     func.upper(col(md.DataFile.calibration_name)).regexp_match(
                            #         self.calibration_name
                            #     )
                            # )

                        else:
                            datafile_query = datafile_query.where(
                                md.DataFile.calibration_name == self.calibration_name
                            )

                        datafile_query = datafile_query.order_by(col(md.DataFile.priority).desc())
                        logger.debug(f"Datafile query: {datafile_query}")

                        try:
                            datafiles = list(session.exec(datafile_query).all())
                            logger.debug(f"Found {len(datafiles)} datafiles for calibration")
                        except SQLAlchemyError as e:
                            logger.error(f"Database error while querying datafiles: {str(e)}")
                            continue
                    else:
                        logger.debug("No calibration filter, getting best datafile")
                        with session.no_autoflush:
                            try:
                                datafiles = [network.recipe.get_best_datafile()]
                                if datafiles[0] is None:
                                    logger.warning(
                                        f"No best datafile found for recipe: {network.recipe_name}"
                                    )
                                    continue
                            except Exception as e:
                                logger.error(
                                    f"Error getting best datafile for recipe {network.recipe_name}: {str(e)}"
                                )
                                continue

                    if not datafiles:
                        logger.error(f"No datafile for {network.recipe_name}. Skipping")
                        continue

                    logger.debug(
                        f"Processing {len(datafiles)} datafiles for network {network.name}"
                    )
                    for datafile in datafiles:
                        try:
                            network_and_data.append(
                                NetworkDataId(network_name=network.name, file_path=datafile.file)
                            )
                            logger.debug(f"Matching data: {datafile.file}")
                        except Exception as e:
                            logger.error(
                                f"Error creating NetworkDataId for {network.name}: {str(e)}"
                            )
                            continue

                except Exception as e:
                    logger.error(f"Error processing network {network.name}: {str(e)}")
                    continue

            logger.debug(
                f"Network data retrieval complete. Found {len(network_and_data)} network-data pairs"
            )
            return network_and_data

        except Exception as e:
            logger.error(f"Unexpected error in get_networkdata_ids: {str(e)}")
            raise ValueError(f"Failed to retrieve network data: {str(e)}") from e


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                       --     network sets     --


class NetworkSet(BaseModel):
    content: List[Union[NetworkDataId, NetworkSelector, "NetworkSet"]] = []

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode='before')
    def validate_content(cls, values):
        # Handle case where values is a list
        if isinstance(values, list):
            return {'content': values}
        # Handle case where content is single item
        content = values.get('content')
        if isinstance(content, (NetworkSelector, NetworkDataId, NetworkSet)):
            values['content'] = [content]
        elif isinstance(content, NetworkSet):
            values['content'] = content.content
        return values

    def run_selectors(self, session):
        new_content = []
        for n in self.content:
            if isinstance(n, NetworkSelector):
                logger.debug(f"Running selector: {n}")
                ncontent = n.get_networkdata_ids(session)
                new_content.extend(ncontent)
                logger.debug(f"Found {len(ncontent)} matching networks")

            elif isinstance(n, NetworkSet):
                # Recursively run selectors on nested NetworkSets
                n.run_selectors(session)
                new_content.extend(n.content)
            else:
                assert isinstance(n, NetworkDataId), f"Expected NetworkDataId but got {type(n)}"
                new_content.append(n)
        self.content = new_content

    def get_networks_and_data(self, session) -> List[Tuple[md.Network, md.DataFile]]:
        res = []
        for n in self.content:
            assert isinstance(n, NetworkDataId)

            try:
                r = n.fetch_network_and_datafile(session)
            except AssertionError as e:
                logger.error(f"Error fetching network and datafile: {e}. Skipping.")
                continue
            res.append(r)

        return res

    def model_dump(self, **kwargs):
        return super().model_dump(exclude={'recipe.networks'}, **kwargs)

    def __len__(self):
        return len(self.content)


class NetworkSetUnion(NetworkSet):
    """
    A union of multiple NetworkSets. (and itself)
    """

    sets: List[NetworkSet] = []
    # exclude from export:
    model_config = ConfigDict(exclude={'sets'})

    allow_duplicates: bool = False

    def run_selectors(self, session):
        for s in self.sets:
            s.run_selectors(session)

        new_content = []
        for s in self.sets:
            new_content.extend(s.content)
        new_content.extend(self.content)
        if not self.allow_duplicates:
            new_content = list(set(new_content))
        self.content = new_content


class NetworkSetIntersection(NetworkSet):
    """
    An intersection of multiple NetworkSets. (and itself)
    """

    sets: List[NetworkSet] = []

    def run_selectors(self, session):
        for s in self.sets:
            s.run_selectors(session)

        new_content = self.sets[0].content
        for s in self.sets[1:]:
            new_content = list(set(new_content) & set(s.content))
        new_content = list(set(new_content) & set(self.content))
        self.content = new_content


class NetworkSetDifference(NetworkSet):
    """
    A difference of two NetworkSets.
    """

    set1: NetworkSet
    set2: NetworkSet

    # # make sure content is empty
    # @model_validator(mode='before')
    # def check_content(self):
    #     assert not self.content, "content of a SetDifference should be empty"

    def run_selectors(self, session):
        self.set1.run_selectors(session)
        self.set2.run_selectors(session)

        new_content = list(set(self.set1.content) - set(self.set2.content))
        self.content = new_content


class NetworkFilter(NetworkSet):
    """Base class for filtering NetworkSets"""

    source_set: NetworkSet

    def run_selectors(self, session):
        self.source_set.run_selectors(session)

        self.content = [
            net_id for net_id in self.source_set.content if self.should_keep(net_id, session)
        ]

    def should_keep(self, network_id: NetworkDataId, session) -> bool:
        """
        Determine if a network should be kept in the filtered set.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement should_keep()")


class CustomFilter(NetworkFilter):
    filter_func: Callable[[md.Network, md.DataFile], bool]

    def should_keep(self, network_id: NetworkDataId, session) -> bool:
        try:
            network, datafile = network_id.fetch_network_and_datafile(session)
            return self.filter_func(network, datafile)
        except Exception as e:
            logger.error(f"Error checking custom filter for network {network_id}: {e}")
            return False


class CleanupFilter(NetworkFilter):
    """Filter NetworkSets based on specific cleanup values"""

    # - remove mismatch ERN - Recs
    # - remove splits, meaning multiple recs for same ERN
    # - anything with recombinase
    ...


class UorfFilter(NetworkFilter):
    """Filter NetworkSets based on specific uORF value pairs"""

    uorf_values: List[Tuple[int, int]] = []

    def should_keep(self, network_id: NetworkDataId, session) -> bool:
        try:
            network, _ = network_id.fetch_network_and_datafile(session)

            # get uORF values from network info
            network_info = network.network_info

            if not network_info or 'uorf_values' not in network_info:
                return False

            # network_info['uorf_values'] should be list[tuple[int, int]]
            # we want single ERN networks
            if len(network_info['uorf_values']) != 1:
                logger.warning(
                    f"Unexpected uORF values for {network_id}: {network_info['uorf_values']}. Is it a single ERN?"
                )
                return False

            uorf_values = tuple(network_info['uorf_values'][0])

            return uorf_values in self.uorf_values

        except Exception as e:
            logger.error(f"Error checking uORF values for network {network_id}: {e}")
            return False


##────────────────────────────────────────────────────────────────────────────}}}


def build_data_manager(
    lib: PartsLibrary,
    db_session,
    path_prefix,
    dataset: NetworkSet,
    data_conf: DataConfig = DEFAULT_DATA_CONFIG,
    network_cache=config.paths.cache.networks,
    data_cache=config.paths.cache.data,
) -> DataManager:
    assert all(
        [isinstance(n, NetworkDataId) for n in dataset.content]
    ), "By now, dataset should only contain NetworkDataId objects"

    networks, datafiles = zip(*dataset.get_networks_and_data(db_session))
    data = []
    actual_networks = []

    for n, f in tqdm(list(zip(networks, datafiles)), desc='Building networks & loading data'):
        n.build(lib, use_cache=network_cache)
        # n.build(lib)
        network = n._network
        if isinstance(network, list):
            logger.debug(
                f"Network {n.name} contains multiple networks due to multiple valid inversion"
            )
        else:
            network = [network]

        for net in network:
            data.append(get_network_XY(net, path_prefix / f.file))
            net.metadata['data_file'] = f.file
            net.metadata['calibration_name'] = f.calibration.name
            net.metadata['recipe_name'] = f.recipe_name
            actual_networks.append(net)

    X, Y = zip(*data)

    return DataManager(X, Y, actual_networks, data_cfg=data_conf, cache_location=data_cache)
