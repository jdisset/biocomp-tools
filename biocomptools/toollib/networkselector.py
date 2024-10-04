from biocomp.library import PartsLibrary
from biocomp.recipe import get_network_XY
from tqdm import tqdm
from biocomp.datautils import DataConfig, DEFAULT_DATA_CONFIG, DataManager
from pydantic import BaseModel, Field, BeforeValidator, model_validator
from sqlmodel import select, Session, col
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Annotated
import biocomptools.toollib.models as md

## {{{                       --     Network Sets     --


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
        network = session.exec(
            select(md.Network).where(md.Network.name == self.network_name)
        ).first()
        datafile = session.exec(
            select(md.DataFile).where(md.DataFile.file == self.file_path)
        ).first()

        assert network, f"No network found for {self.network_name}"
        assert datafile, f"No datafile found for {self.file_path}"

        return network, datafile


class NetworkSelector(BaseModel):
    """
    Manually writing a NetworkAndData can be very annoying and verbose and error-prone.
    This class allows to batch select networks based on their names, recipes, experiments, etc.
    """

    experiment_name: Optional[str]
    recipe_name: Optional[str] = None
    calibration_name: Optional[str] = None
    output_name: Optional[str] = None

    def get_networkdata_ids(self, session) -> List[NetworkDataId]:
        query = select(md.Network).join(md.Recipe).join(md.Experiment)

        if self.experiment_name:
            query = query.where(col(md.Experiment.name).regexp_match(self.experiment_name))
        if self.recipe_name:
            query = query.where(col(md.Recipe.name).regexp_match(self.recipe_name))

        networks = session.exec(query).all()

        if self.output_name is not None:
            networks = [
                network
                for network in networks
                if network.network_info['dependent_outputs'][0].upper() == self.output_name.upper()
            ]

        network_and_data = []
        for network in networks:
            if self.calibration_name:
                datafile_query = (
                    select(md.DataFile)
                    .where(
                        md.DataFile.recipe_name == network.recipe_name,
                        col(md.DataFile.calibration_name).regexp_match(self.calibration_name),
                    )
                    .order_by(col(md.DataFile.priority).desc())
                )
                datafile = session.exec(datafile_query).first()
            else:
                datafile = network.recipe.get_best_datafile()

            assert datafile, f"No datafile found for {network.recipe_name}"
            network_and_data.append(
                NetworkDataId(network_name=network.name, file_path=datafile.file)
            )

        return network_and_data


class NetworkSet(BaseModel):
    content: List[NetworkDataId | NetworkSelector] = []

    def run_selectors(self, session):
        # we want to run this as a post-init so that we store the content in a more
        # repeatable / serializable way than selectors when dumping the whole config
        new_content = []
        for n in self.content:
            if isinstance(n, NetworkSelector):
                new_content.extend(n.get_networkdata_ids(session))
            else:
                assert isinstance(n, NetworkDataId)
                new_content.append(n)
        self.content = new_content

    def get_networks_and_data(self, session) -> List[Tuple[md.Network, md.DataFile]]:
        res = []
        for n in self.content:
            assert isinstance(n, NetworkDataId)
            res.append(n.fetch_network_and_datafile(session))
        return res


class NetworkSetUnion(NetworkSet):
    """
    A union of multiple NetworkSets. (and itself)
    """

    sets: List[NetworkSet] = []

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

    # make sure content is empty
    @model_validator(mode='before')
    def check_content(self):
        assert not self.content, "content of a SetDifference should be empty"

    def run_selectors(self, session):
        self.set1.run_selectors(session)
        self.set2.run_selectors(session)

        new_content = list(set(self.set1.content) - set(self.set2.content))
        self.content = new_content


def build_data_manager(
    lib: PartsLibrary,
    db_session,
    path_prefix,
    data_conf: DataConfig,
    dataset: NetworkSet,
    use_cache=None,
) -> DataManager:
    networks, datafiles = zip(*dataset.get_networks_and_data(db_session))
    data = []
    for n, f in tqdm(list(zip(networks, datafiles)), desc='Building networks & loading data'):
        n.build(lib, use_cache=use_cache)
        data.append(get_network_XY(n._network, path_prefix / f.file))

    X, Y = zip(*data)

    return DataManager(X, Y, networks, data_cfg=data_conf)


##────────────────────────────────────────────────────────────────────────────}}}##
