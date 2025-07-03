from biocomp.library import PartsLibrary
from biocomp.recipe import get_network_XY
from tqdm import tqdm
from biocomp.datautils import DataConfig, DEFAULT_DATA_CONFIG, DataManager
from pydantic import BaseModel, Field, BeforeValidator, model_validator, field_validator, ConfigDict
from sqlalchemy.orm import selectinload
from sqlalchemy import func
from sqlmodel import select, Session, col
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Annotated
from biocomptools.toollib.models import NetworkDataPair, Recipe, DataFile, Network, Experiment
from biocomptools.logging_config import get_logger
from biocomptools.toollib.common import config
from sqlalchemy.exc import SQLAlchemyError
from biocomp.utils import load_lib
from dracon.utils import ser_debug, list_like, dict_like
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from collections.abc import Mapping

logger = get_logger(__name__)


## {{{                         --     selector     --


class RegexString(str):
    """Base class for regex string types."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(cls),
                    ]
                ),
            ]
        )

    def __new__(cls, string):
        return super().__new__(cls, string)


class Regex(RegexString):
    """A string that should be treated as a regex pattern."""
    pass


class iRegex(RegexString):
    """A string that should be treated as a case-insensitive regex pattern."""
    pass


def maybe_regex(s: str) -> str:
    if isinstance(s, (Regex, iRegex)):
        return s
    else:
        import re

        return re.escape(s)


def apply_regex_filter(query, column, pattern: str | Regex | iRegex):
    """Apply regex or exact match filter to a query column."""
    if isinstance(pattern, iRegex):
        regex_pattern = f"(?i){pattern}"
        return query.where(column.regexp_match(regex_pattern))
    elif isinstance(pattern, Regex):
        return query.where(column.regexp_match(pattern))
    else:
        return query.where(column == pattern)


class NetworkSelector(BaseModel):
    """
    Manually writing a NetworkAndData can be very annoying and verbose and error-prone.
    This class allows to batch select networks based on their names, recipes, experiments, etc.
    """

    experiment_name: Optional[str | Regex | iRegex] = None
    recipe_name: Optional[str | Regex | iRegex] = None
    recipe_short_name: Optional[str | Regex | iRegex] = None
    calibration_name: Optional[str | Regex | iRegex] = None
    output_name: Optional[str] = None

    def get_networkdata_ids(self, session) -> List[NetworkDataPair]:
        """
        Retrieve network data IDs based on specified filters.

        Args:
            session: The database session

        Returns:
            List[NetworkDataPair]: List of network data pairs matching the filters.

        Raises:
            ValueError: If no networks are found or if query execution fails
            SQLAlchemyError: For database-related errors
        """

        try:
            query = (
                select(Network)
                .join(Recipe)
                .join(Experiment)
                .options(
                    selectinload(Network.recipe).selectinload(Recipe.data_files),
                    selectinload(Network.recipe).selectinload(Recipe.experiment),
                    selectinload(Network.recipe).selectinload(Recipe.networks),
                )
            )
            logger.debug(f"Initial network query: {query}")

            if self.experiment_name:
                logger.debug(f"Applying experiment filter: {self.experiment_name}")
                query = apply_regex_filter(query, col(Experiment.name), self.experiment_name)

            if self.recipe_name:
                if isinstance(self.recipe_name, (Regex, iRegex)):
                    query = apply_regex_filter(query, col(Recipe.name), self.recipe_name)
                else:
                    if self.experiment_name and not isinstance(self.experiment_name, (Regex, iRegex)):
                        # if we have exact experiment name, we can construct the full recipe name
                        query = query.where(
                            Recipe.name == f"{self.experiment_name}_{self.recipe_name}"
                        )
                    else:
                        # if no experiment name or experiment name is regex, match any experiment prefix
                        query = query.where(
                            col(Recipe.name).regexp_match(f".*_{self.recipe_name}$")
                        )

            if self.recipe_short_name:
                logger.debug(f"Applying recipe_short_name filter: {self.recipe_short_name}")
                query = apply_regex_filter(query, col(Recipe.short_name), self.recipe_short_name)

            logger.debug(f"Final network query: {query}")

            # Execute query
            try:
                networks = session.exec(query).all()
            except SQLAlchemyError as e:
                logger.error(f"Database error while executing network query: {str(e)}")
                raise ValueError(f"Failed to execute network query: {str(e)}") from e

            if not networks:
                self._log_helpful_error_message(session, query)
                return []

            # filter by output name if specified
            if self.output_name is not None:
                original_count = len(networks)
                networks = [
                    network
                    for network in networks
                    if network.network_info['dependent_outputs'][0].upper()
                    == self.output_name.upper()
                ]

            network_and_data = []

            for network in networks:
                logger.debug(f"Processing network: {network.name}")

                try:
                    if self.calibration_name:
                        logger.debug(f"Applying calibration filter: {self.calibration_name}")
                        datafile_query = (
                            select(DataFile)
                            .options(selectinload(DataFile.calibration))
                            .where(DataFile.recipe_name == network.recipe_name)
                        )

                        # add calibration name filter based on type
                        datafile_query = apply_regex_filter(
                            datafile_query, col(DataFile.calibration_name), self.calibration_name
                        )

                        datafile_query = datafile_query.order_by(col(DataFile.priority).desc())
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
                            ndp = NetworkDataPair(
                                network_name=network.name, datafile_path=datafile.file
                            )
                            logger.debug(f"NetworkDataId created: {ndp}")
                            if ndp.network is None:
                                ndp.network = network
                            if ndp.datafile is None:
                                ndp.datafile = datafile
                            network_and_data.append(ndp)
                            logger.debug(f"Matching data: {datafile.file}")
                        except Exception as e:
                            logger.error(f"Error creating NetworkDataId for {network.name}")
                            logger.exception(e)
                            raise

                except Exception as e:
                    logger.error(f"Error processing network {network.name}")
                    logger.exception(e)
                    raise

            logger.debug(
                f"Network data retrieval complete. Found {len(network_and_data)} network-data pairs for selector {self}."
            )
            return network_and_data

        except Exception as e:
            logger.exception(e)
            raise

    def _log_helpful_error_message(self, session, original_query):
        """Provide helpful error messages with available alternatives when no networks are found."""
        error_parts = []
        suggestions = []
        
        # Basic error message
        filter_parts = []
        if self.experiment_name:
            filter_parts.append(f"experiment '{self.experiment_name}'")
        if self.recipe_name:
            filter_parts.append(f"recipe '{self.recipe_name}'")
        if self.recipe_short_name:
            filter_parts.append(f"recipe_short_name '{self.recipe_short_name}'")
        
        if filter_parts:
            error_parts.append(f"No networks found for {', '.join(filter_parts)}")
        else:
            error_parts.append("No networks found")
        
        try:
            # If we have an experiment name, try to find available recipes for that experiment
            if self.experiment_name:
                exp_query = (
                    select(Recipe)
                    .join(Experiment)
                    .join(Network)
                    .options(
                        selectinload(Recipe.experiment),
                        selectinload(Recipe.networks)
                    )
                )
                exp_query = apply_regex_filter(exp_query, col(Experiment.name), self.experiment_name)
                
                available_recipes = session.exec(exp_query).all()
                
                if available_recipes:
                    recipe_info = [(r.name, r.short_name) for r in available_recipes]
                    suggestions.append(f"Available recipes for experiment '{self.experiment_name}': {recipe_info[:10]}{'...' if len(recipe_info) > 10 else ''}")
                else:
                    # If no recipes found for experiment, check if experiment exists at all
                    exp_only_query = select(Experiment)
                    exp_only_query = apply_regex_filter(exp_only_query, col(Experiment.name), self.experiment_name)
                    available_experiments = session.exec(exp_only_query).all()
                    
                    if available_experiments:
                        suggestions.append(f"Experiment '{self.experiment_name}' exists but has no recipes with networks")
                    else:
                        # Show similar experiment names
                        all_exp_query = select(Experiment.name).distinct()
                        all_experiments = [row for row in session.exec(all_exp_query).all()]
                        suggestions.append(f"Experiment '{self.experiment_name}' not found. Available experiments: {all_experiments[:10]}{'...' if len(all_experiments) > 10 else ''}")
            
            # If we have a recipe name or recipe_short_name but no experiment, try to find matching recipes
            elif self.recipe_name or self.recipe_short_name:
                recipe_query = (
                    select(Recipe)
                    .join(Experiment)
                    .join(Network)
                    .options(
                        selectinload(Recipe.experiment),
                        selectinload(Recipe.networks)
                    )
                )
                
                # Apply recipe_name filter if specified
                if self.recipe_name:
                    if isinstance(self.recipe_name, (Regex, iRegex)):
                        recipe_query = apply_regex_filter(recipe_query, col(Recipe.name), self.recipe_name)
                    else:
                        # For exact recipe name, try partial match
                        recipe_query = recipe_query.where(col(Recipe.name).regexp_match(f".*_{self.recipe_name}$"))
                
                # Apply recipe_short_name filter if specified
                if self.recipe_short_name:
                    recipe_query = apply_regex_filter(recipe_query, col(Recipe.short_name), self.recipe_short_name)
                
                available_recipes = session.exec(recipe_query).all()
                
                if available_recipes:
                    recipe_info = [(r.name, r.short_name, r.experiment.name) for r in available_recipes]
                    filter_desc = []
                    if self.recipe_name:
                        filter_desc.append(f"recipe_name '{self.recipe_name}'")
                    if self.recipe_short_name:
                        filter_desc.append(f"recipe_short_name '{self.recipe_short_name}'")
                    filter_text = " and ".join(filter_desc)
                    suggestions.append(f"Available recipes matching {filter_text}: {recipe_info[:10]}{'...' if len(recipe_info) > 10 else ''}")
                else:
                    filter_desc = []
                    if self.recipe_name:
                        filter_desc.append(f"recipe_name '{self.recipe_name}'")
                    if self.recipe_short_name:
                        filter_desc.append(f"recipe_short_name '{self.recipe_short_name}'")
                    filter_text = " and ".join(filter_desc)
                    suggestions.append(f"No recipes found matching {filter_text}")
            
            # Add calibration info if specified
            if self.calibration_name:
                error_parts.append(f"calibration '{self.calibration_name}'")
                
                
            # Add output name info if specified  
            if self.output_name:
                error_parts.append(f"output '{self.output_name}'")
                
        except Exception as e:
            logger.debug(f"Error while generating helpful error message: {e}")
            suggestions.append("Unable to query for suggestions due to database error")
        
        # Construct final error message
        main_error = ", ".join(error_parts)
        if suggestions:
            suggestion_text = "\n  ".join(suggestions)
            full_message = f"{main_error}.\n  Suggestions:\n  {suggestion_text}"
        else:
            full_message = f"{main_error}. No suggestions available."
            
        logger.error(full_message)


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                       --     network sets     --


class NetworkSet(BaseModel):
    content: List[Union[NetworkDataPair, NetworkSelector, "NetworkSet"]] = []
    name: Optional[str] = None

    @property
    def _engine(self):
        """Lazy-load the database engine when needed (otherwise unpicklable)."""
        from biocomptools.toollib.models import get_biocompdb_sqlite_engine
        from biocomptools.toollib.common import config

        _db_engine = get_biocompdb_sqlite_engine(config.db.sqlite.path)
        return _db_engine

    @property
    def db_session(self):
        return Session(self._engine)

    @model_validator(mode='before')
    def content_field_was_skipped(cls, values):
        # accept shorthand notation without content=...
        if isinstance(values, list):
            return {'content': values}
        return values

    @field_validator('content', mode='before')
    @classmethod
    def route_content(cls, v: Any, info):
        logger.debug(
            f"From class '{cls.__name__}' (field '{info.field_name}') routing content: {v}"
        )

        if isinstance(v, (NetworkDataPair, NetworkSelector, NetworkSet)):
            v = [v]
        elif not list_like(v):
            raise TypeError(
                f"Field '{info.field_name}' for {cls.__name__} must be a list or a single "
                f"NetworkDataPair/NetworkSelector/NetworkSet instance. Got input of type: {type(v)}"
            )

        def route_item(obj_in_list: Any) -> Union[NetworkDataPair, NetworkSelector, "NetworkSet"]:
            if isinstance(obj_in_list, (NetworkDataPair, NetworkSelector, NetworkSet)):
                return obj_in_list

            if hasattr(obj_in_list, 'network_name') and hasattr(obj_in_list, 'datafile_path'):
                return obj_in_list

            if isinstance(obj_in_list, Mapping):
                dict_obj = dict(obj_in_list)
                if 'datafile_path' in dict_obj:
                    return NetworkDataPair(**dict_obj)
                elif 'content' in dict_obj:  # NetworkSet-like structure
                    return NetworkSet(**dict_obj)
                else:
                    return NetworkSelector(**dict_obj)
            else:
                raise TypeError(
                    f"Invalid item in '{info.field_name}' list for {cls.__name__}. "
                    f"Expected a dict/mapping or a pre-parsed model instance, got {type(obj_in_list)}"
                )

        processed_list = [route_item(item) for item in v]
        logger.debug(
            f"Routed '{info.field_name}' to types {[type(r).__name__ for r in processed_list]}"
        )
        return processed_list

    def run_selectors(self, session=None):
        sess = session or self.db_session
        logger.debug(f"Running selectors on {len(self.content)} items with session {sess}")
        new_content = []
        for n in self.content:
            if isinstance(n, NetworkSelector):
                logger.debug(f"Running selector: {n}")
                ncontent = n.get_networkdata_ids(sess)
                new_content.extend(ncontent)
                logger.debug(f"Found {len(ncontent)} matching networks")

            elif isinstance(n, NetworkSet):
                # Recursively run selectors on nested NetworkSets
                logger.debug(f"Running nested NetworkSet: {n}")
                n.run_selectors(sess)
                new_content.extend(n.content)
            else:
                assert isinstance(n, NetworkDataPair), f"Expected NetworkDataPair but got {type(n)}"
                new_content.append(n)
        self.content = new_content
        logger.debug(f"Finished running selectors. Found {len(self.content)} items")
        if session is None:
            sess.close()

    def get_networks_and_data(self, session=None) -> List[Tuple[Network, DataFile]]:
        sess = session or self.db_session
        close_session_locally = session is None

        res = []
        logger.debug(f"Getting networks and data for {len(self.content)} items")
        for n_pair_data in self.content:
            assert isinstance(n_pair_data, NetworkDataPair)
            logger.debug(
                f"Processing {type(n_pair_data)} {n_pair_data.model_dump(exclude={'network', 'datafile'})}"
            )

            try:
                statement = (
                    select(Network)
                    .where(Network.name == n_pair_data.network_name)
                    .options(selectinload(Network.recipe))
                )
                db_network = sess.exec(statement).one_or_none()

                db_datafile = sess.get(DataFile, n_pair_data.datafile_path)

                if not db_network:
                    raise ValueError(
                        f"Network '{n_pair_data.network_name}' not found in the database."
                    )
                if not db_datafile:
                    raise ValueError(
                        f"Datafile '{n_pair_data.datafile_path}' not found in the database."
                    )

                r = (db_network, db_datafile)
            except Exception as e:
                logger.error(
                    f"Error processing or fetching entities for pair {n_pair_data.model_dump(exclude={'network', 'datafile'})}: {e}. Skipping."
                )
                logger.exception(e)
                continue
            res.append(r)

        if close_session_locally:
            sess.close()

        return res

    def model_dump(self, **kwargs):
        return super().model_dump(exclude={'recipe.networks'}, **kwargs)

    def __len__(self):
        return len(self.content)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}[{len(self.content)} items]"


class NetworkSetOperation(NetworkSet):
    """Base class for set operations."""

    sets: List[NetworkSet] = []
    # exclude from export:
    model_config = ConfigDict(exclude={'sets'})

    def _update_name_with_operator(self, operator: str):
        """Update name with given operator between set names."""
        if self.name is not None:
            return
        if any(s.name is None for s in self.sets):
            return
        name = ''
        for i, s in enumerate(self.sets):
            name += f"({s.name})"
            if i < len(self.sets) - 1:
                name += operator
        self.name = name


class NetworkSetUnion(NetworkSetOperation):
    """
    A union of multiple NetworkSets. (and itself)
    """

    allow_duplicates: bool = False

    @model_validator(mode='before')
    @classmethod
    def normalize_list_input_to_sets(cls, data: Any) -> Any:
        logger.debug(f"{cls.__name__}.normalize_list_input_to_sets received: {data}")
        if isinstance(data, list):
            # assume the list items are the sets
            return {"sets": data}
        return data

    def update_name(self):
        self._update_name_with_operator(' U ')

    def run_selectors(self, session=None):
        try:
            logger.debug(f"Running union on {len(self.sets)} sets")
            logger.debug(f"{self.sets=}")
            for s in self.sets:
                s.run_selectors(session)

            new_content = []
            for s in self.sets:
                new_content.extend(s.content)
            new_content.extend(self.content)
            if not self.allow_duplicates:
                new_content = list(set(new_content))
            self.content = new_content
            self.update_name()
        except Exception as e:
            logger.error(f"Error running union on {len(self.sets)} sets.")
            logger.exception(e)
            raise e


class NetworkSetIntersection(NetworkSetOperation):
    """
    An intersection of multiple NetworkSets. (and itself)
    """
    pass

    @model_validator(mode='before')
    @classmethod
    def normalize_list_input_to_sets(cls, data: Any) -> Any:
        logger.debug(f"{cls.__name__}.normalize_list_input_to_sets received: {data}")
        if isinstance(data, list):
            # assume the list items are the sets
            return {"sets": data}
        return data

    def update_name(self):
        self._update_name_with_operator(' ∩ ')

    def run_selectors(self, session=None):
        try:
            logger.debug(f"Running intersection on {len(self.sets)} sets")
            for s in self.sets:
                s.run_selectors(session)

            new_content = self.sets[0].content
            for s in self.sets[1:]:
                new_content = list(set(new_content) & set(s.content))
            new_content = list(set(new_content) & set(self.content))
            self.content = new_content
            self.update_name()
        except Exception as e:
            logger.error(f"Error running intersection on {len(self.sets)} sets.")
            logger.exception(e)
            raise e


class NetworkSetDifference(NetworkSet):
    """
    A difference of two NetworkSets.
    """

    set1: NetworkSet
    set2: NetworkSet

    def update_name(self):
        if self.name is not None:
            return
        if self.set1.name is None or self.set2.name is None:
            return
        self.name = f"({self.set1.name}) - ({self.set2.name})"

    def run_selectors(self, session=None):
        try:
            logger.debug(
                f"Running difference on {len(self.set1.content)} - {len(self.set2.content)} items"
            )
            self.set1.run_selectors(session)
            self.set2.run_selectors(session)

            new_content = list(set(self.set1.content) - set(self.set2.content))
            self.content = new_content
            self.update_name()
        except Exception as e:
            logger.error(
                f"Error running difference on {len(self.set1.content)} - {len(self.set2.content)}."
            )
            logger.exception(e)
            raise e


class NetworkFilter(NetworkSet):
    """Base class for filtering NetworkSets"""

    source_set: NetworkSet

    _size_diff: int = 0

    def update_name(self):
        if self.name is not None:
            return
        if self.source_set.name is None:
            return
        self.name = self.source_set.name
        if self._size_diff != 0:
            self.name += f"(- {self._size_diff})"

    def run_selectors(self, session=None):
        try:
            self.source_set.run_selectors(session)
            self.content = [
                net_id for net_id in self.source_set.content if self.should_keep(net_id, session)
            ]
            self._size_diff = len(self.source_set.content) - len(self.content)
            self.update_name()
        except Exception as e:
            logger.error(f"Error running filter on {len(self.source_set.content)} items.")
            logger.exception(e)
            raise e

    def should_keep(self, netdata: NetworkDataPair, session) -> bool:
        """
        Determine if a network should be kept in the filtered set.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement should_keep()")


class CustomFilter(NetworkFilter):
    filter_func: Callable[[Network, DataFile], bool]

    def should_keep(self, netdata: NetworkDataPair, session) -> bool:
        if not netdata.network:
            raise ValueError(f"Network data pair {netdata} has no network")
        if not netdata.datafile:
            raise ValueError(f"Network data pair {netdata} has no datafile")
        return self.filter_func(netdata.network, netdata.datafile)


class CleanupFilter(NetworkFilter):
    """General cleanup. Remove networs with:
    - multiple ERN rec
    - missing complementary ERN parts
    - recombinases
    """

    def _find_twice_same_rec_with_different_rna(self, net_info, lib):
        appears_twice = []
        all_parts = net_info['all_parts'].values()
        for i, p1 in enumerate(all_parts):
            for pname, pcat in p1.items():
                if pcat == 'ERN_recog_site_5p':
                    # make sure this part doesn't appear in any other TU
                    for j, p2 in enumerate(all_parts):
                        if i == j:
                            continue
                        if pname in p2:
                            # reject unless it's the same content
                            # TODO: technically we should allow different untranscribed regions
                            # but in the current experiments they are always the same
                            # so if there's a difference in the parts
                            p1set = set(p1.keys())
                            p2set = set(p2.keys())
                            if p1set != p2set:
                                if pname not in appears_twice:
                                    appears_twice.append(pname)
                            break
        return appears_twice

    def _find_missing_complementary_parts(self, net_info, lib):
        missing = {}
        all_parts = net_info['all_parts'].values()

        def find_missing_part(part_name, part_col, complementary_col, valid_types=None):
            if valid_types is None:
                valid_types = ['ERN']

            rows = lib.sequestrons[part_col].eq(part_name) & lib.sequestrons.type.isin(valid_types)
            if rows.any():
                complementaries = lib.sequestrons[rows][complementary_col].values
                found = False
                for j, tp2 in enumerate(all_parts):
                    if any([p in tp2 for p in complementaries]):
                        found = True
                        break
                if not found:
                    missing[part_name] = complementaries.tolist()

        for tp in all_parts:
            parts = set(tp.keys())
            for p in parts:
                find_missing_part(p, 'positive_part', 'negative_part')
                find_missing_part(p, 'negative_part', 'positive_part')

        return missing

    def _find_invalid_sequestron_types(self, net_info, lib, valid_types=None):
        invalid_pairs = []
        if valid_types is None:
            valid_types = ['ERN']
        invalid_types = set(lib.sequestrons.type.unique()) - set(valid_types)
        invalid_rows = lib.sequestrons.type.isin(invalid_types)
        invalid_parts = []
        for _, row in lib.sequestrons[invalid_rows].iterrows():
            involved_parts = set([row.positive_part, row.negative_part])
            invalid_parts.append(involved_parts)
        all_parts = net_info['all_parts'].values()
        all_parts = set([p for tp in all_parts for p in tp.keys()])
        for ip in invalid_parts:
            if ip.issubset(all_parts):
                invalid_pairs.append(ip)
        return invalid_pairs

    def _find_invalid_part_categories(self, net_info, lib, invalid_categories=None):
        invalid_parts = []
        if invalid_categories is None:
            invalid_categories = [
                'inverted_seq',
                'rcb_rec_5p',
                'rcb_rec_3p',
                'recombinase_bwd',
                'recombinase_fwd',
                'ERN_recog_site_3p',
            ]
        all_parts = net_info['all_parts'].values()
        for tp in all_parts:
            for p, c in tp.items():
                if c in invalid_categories:
                    invalid_parts.append(p)
        return invalid_parts

    def should_keep(self, netdata: NetworkDataPair, session) -> bool:
        if not netdata.network:
            raise ValueError(f"NetworkDataPair object '{netdata}' has no network")
        netname = netdata.network_name
        net_info = netdata.network.network_info
        lib = load_lib()

        missing_parts = self._find_missing_complementary_parts(net_info, lib)
        if missing_parts:
            logger.info(
                f"Network {netname} has missing complementary parts: {missing_parts}. Skipping."
            )
            return False

        invalid_types = self._find_invalid_sequestron_types(net_info, lib)
        if invalid_types:
            logger.info(
                f"Network {netname} has invalid sequestron types: {invalid_types}. Skipping."
            )
            return False

        invalid_categories = self._find_invalid_part_categories(net_info, lib)
        if invalid_categories:
            logger.info(
                f"Network {netname} has parts from invalid categories: {invalid_categories}. Skipping."
            )
            return False

        return True


class UorfFilter(NetworkFilter):
    """Filter NetworkSets based on specific uORF value pairs"""

    uorf_values: List[Tuple[int, int]] = []

    def should_keep(self, netdata: NetworkDataPair, session) -> bool:
        if not netdata.network:
            raise ValueError(f"Network data pair {netdata} has no network")

        # get uORF values from network info
        network_info = netdata.network.network_info

        if not network_info or 'uorf_values' not in network_info:
            return False

        # network_info['uorf_values'] should be list[tuple[int, int]]
        # we want single ERN networks
        if len(network_info['uorf_values']) != 1:
            logger.debug(
                f"Unexpected uORF values for {netdata.network}: {network_info['uorf_values']}. Is it a single ERN?"
            )
            return False

        uorf_values = tuple(network_info['uorf_values'][0])

        return uorf_values in self.uorf_values


##────────────────────────────────────────────────────────────────────────────}}}


def build_data_manager(
    lib: PartsLibrary,
    db_session,
    path_prefix,
    dataset: NetworkSet,
    data_conf: DataConfig = DEFAULT_DATA_CONFIG,
    data_cache=config.paths.cache.data,
    jax_sampling: bool = True,
) -> DataManager:
    assert all([isinstance(n, NetworkDataPair) for n in dataset.content]), (
        "By now, dataset should only contain NetworkDataPair objects"
    )

    net_data = dataset.get_networks_and_data(db_session)
    if not net_data:
        raise ValueError("No networks and data found in the dataset. Please check your selectors.")

    networks, datafiles = zip(*net_data)
    data = []
    actual_networks = []

    for n, f in tqdm(list(zip(networks, datafiles)), desc='Building networks & loading data'):
        n.build(lib)
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

    return DataManager(
        X,
        Y,
        actual_networks,
        data_cfg=data_conf,
        cache_location=data_cache,
        jax_sampling=jax_sampling,
    )
