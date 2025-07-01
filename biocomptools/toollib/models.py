from sqlmodel import Field, SQLModel, create_engine, Relationship
from typing import List, Optional, Annotated, Any, TypeVar
import sqlalchemy as sa
from sqlalchemy import Column, JSON
from pydantic import BeforeValidator, BaseModel
from sqlalchemy import and_
from pathlib import Path
from biocomptools.toollib.common import config
import biocomp as bc
from biocomp.utils import load_lib

from sqlmodel import select, Session, col
from biocomptools.logging_config import get_logger
from biocomptools.toollib.hashutils import pronounceable_hash32

logger = get_logger(__name__)


def to_str(data: Any) -> Any:
    if not isinstance(data, str) and data is not None:
        return str(data)
    return data


ForcedStr = Annotated[str, BeforeValidator(to_str)]
ForcedOptionalStr = Annotated[Optional[str], BeforeValidator(to_str)]

T = TypeVar("T")
ListOr = List[T] | T


class BiocompDB(SQLModel, registry=sa.orm.registry()):
    pass


class Experiment(BiocompDB, table=True):
    name: str = Field(primary_key=True)
    path: ForcedStr
    comments: Optional[str] = None
    content: dict = Field(default_factory=dict, sa_column=Column(JSON))
    errors: dict = Field(default_factory=dict, sa_column=Column(JSON))

    recipes: List["Recipe"] = Relationship(back_populates="experiment")

    @staticmethod
    def sample_is_control(sample: dict) -> bool:
        if 'control' in sample:
            return sample['control']
        return False

    def safe_copy(self):
        """
        Copy the experiment object without any SQLAlchemy references.
        """
        new_obj = Experiment(
            name=self.name,
            path=self.path,
            comments=self.comments,
            content=self.content,
            errors=self.errors,
        )

        return new_obj

    def find_recipes(
        self,
        path_prefix: Optional[str] = None,
        recipe_subpath: Optional[str] = None,
        recipe_ext='.recipe.json5',
        **kwargs,
    ) -> List["Recipe"]:
        """
        returns a dict of sample_name -> Recipe
        """
        recipes = []
        if not self.content:
            raise ValueError("Experiment content is empty")
        for s in self.content['samples']:
            if self.sample_is_control(s):
                continue
            basepath = Path(self.path)
            basepath = basepath if recipe_subpath is None else basepath / recipe_subpath
            filepath = basepath / f"{s['recipe']}{recipe_ext}"
            recipe = Recipe.from_file(
                filepath, xp_name=self.name, path_prefix=path_prefix, **kwargs
            )
            assert recipe.content.get('name') == s['recipe'], (
                f"Recipe name mismatch {recipe.content.get('name')} != {s['recipe']}"
            )
            assert s['recipe'] not in recipes, f"Duplicate recipe name {s['recipe']}"
            recipes.append(recipe)
        return recipes


class Calibration(BiocompDB, table=True):
    fullname: str = Field(primary_key=True)
    pipeline: dict = Field(default_factory=dict, sa_column=Column(JSON))
    data_files: List["DataFile"] = Relationship(back_populates="calibration")
    name: str
    quality: Optional[float] = 0.0


class DataFile(BiocompDB, table=True):
    file: str = Field(primary_key=True)
    attrs: dict = Field(default_factory=dict, sa_column=Column(JSON))
    calibration_name: str = Field(foreign_key="calibration.fullname")
    recipe_name: Optional[str] = Field(foreign_key="recipe.name", default=None)

    priority: int = 0  # used to select the best data file for a given recipe

    calibration: Optional[Calibration] = Relationship(back_populates="data_files")
    recipe: Optional["Recipe"] = Relationship(back_populates="data_files")

    plotted_in: List["Plot"] = Relationship(back_populates="datafile")

    @property
    def url_encoded_filepath(self):
        """
        Returns the file path as a URL encoded string.
        """
        import urllib.parse

        return urllib.parse.quote(self.file)

    def safe_copy(self):
        """
        Copy the datafile object without any SQLAlchemy references.
        """
        new_obj = DataFile(
            file=self.file,
            attrs=self.attrs,
            calibration_name=self.calibration_name,
            recipe_name=self.recipe_name,
            priority=self.priority,
        )

        return new_obj

    def load_data(self, path_prefix=None):
        import pandas as pd

        filepath = Path(self.file) if path_prefix is None else Path(path_prefix) / self.file
        filepath = filepath.expanduser().resolve()

        assert filepath.exists(), f"File {filepath} does not exist"

        ext = filepath.suffix
        if ext == '.csv':
            return pd.read_csv(filepath)
        elif ext == '.parquet':
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file extension {ext}")


class Network(BiocompDB, table=True):
    name: str = Field(primary_key=True)
    recipe_name: str = Field(foreign_key="recipe.name")
    network_info: dict = Field(default_factory=dict, sa_column=Column(JSON))

    recipe: Optional["Recipe"] = Relationship(back_populates="networks")

    _network: Optional[bc.network.Network] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._network = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._network = None

    def safe_copy(self):
        """
        Copy the network object without any SQLAlchemy references.
        """
        new_obj = Network(
            name=self.name,
            recipe_name=self.recipe_name,
            network_info=self.network_info,
        )

        # can't copy recipe by accessing the recipe arg
        # because it would trigger the sqlalchemy BS to be added again
        # one way would be to use __dict__ directly but I'll just leave it for now
        # if self.recipe is not None:
        #     new_obj.recipe = self.recipe.safe_copy()

        if self._network is not None:
            new_obj._network = self._network.model_copy()

        return new_obj

    @classmethod
    def from_network(cls, network: bc.network.Network, recipe_name=None, **kwargs):
        network_info = bc.network.generate_network_info(network)
        print(f"Network markers: {network_info['markers']}")
        if recipe_name is None:
            recipe_name = 'unknown'
        obj = cls(
            name=f"{recipe_name}_{'-'.join(network_info['markers'])}",
            recipe_name=recipe_name,
            network_info=network_info,
            **kwargs,
        )
        obj._network = network
        return obj

    @property
    def network(self):
        if not self.built:
            self.build(lib=load_lib())
        assert self._network is not None
        return self._network

    @property
    def built(self):
        if self.__pydantic_private__ is None or self.__pydantic_private__.get('_network') is None:
            return False
        return self._network is not None

    def build(self, lib, use_cache=config.paths.cache.networks, force=False):
        if self.built and not force:
            return self._network
        # recipe = self.recipe  # should lazy load
        recipe = self.__dict__.get('recipe')
        if recipe is None:
            logger.error(f"Recipe for network {self.name} not found. Skipping build.")
            return None

        recipe_networks = recipe.build_networks(
            lib=lib,
            use_cache=use_cache,
            inverse='all',
            add_to_self=False,
        )

        logger.debug(
            f"Recipe {recipe.name} yielded {len(recipe_networks)} networks: {[n.name for n in recipe_networks]}"
        )

        for net in recipe_networks:
            if net.name == self.name:
                logger.debug(f"Network {net.name} found after building recipe {recipe.name}")
                self._network = net._network
                return self._network

        all_net_names = '\n   - '.join([net.name for net in recipe_networks])
        msg = f"""Network "{self.name}" not found after building recipe "{self.recipe.name}".
            Recipe yielded the following networks: {all_net_names}"""
        logger.error(msg)
        raise ValueError(msg)

    def title(self):
        if self._network is None:
            return f"{self.name}"
        fresh_info = bc.network.generate_network_info(self._network)
        # uorfstr = '\n'.join(fresh_info['uorf_names'])
        titlestr = f"{fresh_info['architecture']}"
        # make sure first letter of each word is capitalized
        titlestr = " ".join([x.capitalize() for x in titlestr.split()])
        if len(fresh_info['ern_names']) > 0:
            titlestr = f"{titlestr} ({', '.join(fresh_info['ern_names'])})"

        for val, name in zip(fresh_info['uorf_values'], fresh_info['uorf_names']):
            if val[0] > 0 or val[1] > 0:
                titlestr = f"{titlestr}\n{name}: {val}"

        return titlestr


class Recipe(BiocompDB, table=True):
    name: str = Field(primary_key=True)
    content: dict = Field(sa_column=Column(JSON))
    hash: str = Field(default=None)

    xp: Optional[str] = Field(foreign_key="experiment.name", default=None)
    file: ForcedOptionalStr = None

    errors: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # Relationships
    experiment: Optional[Experiment] = Relationship(back_populates="recipes")
    networks: List["Network"] = Relationship(back_populates="recipe")
    data_files: List["DataFile"] = Relationship(back_populates="recipe")

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.hash = self.generate_hash()

    def safe_copy(self):
        """
        Copy the recipe object without any SQLAlchemy references.
        """
        new_obj = Recipe(
            name=self.name,
            content=self.content,
            xp=self.xp,
            file=self.file,
        )

        if self.experiment is not None:
            new_obj.experiment = self.experiment.safe_copy()

        return new_obj

    def generate_hash(self):
        import xxhash
        import json
        import base64

        self_json = json.dumps(self.content, sort_keys=True)
        xxhash_obj = xxhash.xxh128()
        xxhash_obj.update(str(self_json).encode())
        return base64.b32encode(xxhash_obj.digest()).decode().rstrip('=')

    @staticmethod
    def from_file(file_path, xp_name=None, path_prefix=None, **kwargs):
        import json5

        filepath = Path(file_path) if path_prefix is None else Path(path_prefix) / file_path
        filepath = Path(filepath).expanduser().resolve()
        with open(filepath, 'r') as f:
            content = json5.load(f)

        name = content.get('name', filepath.stem)
        if xp_name is not None:
            name = f"{xp_name}_{name}"

        return Recipe(
            name=name,
            content=content,
            xp=xp_name,
            file=str(file_path),
            **kwargs,
        )

    def get_best_datafile(self):
        if not self.data_files:
            return None
        return sorted(self.data_files, key=lambda x: x.priority)[0]

    def build_networks(
        self,
        lib=None,
        use_cache=None,
        inverse='all',
        add_to_self=False,
    ) -> list["Network"]:
        """
        Build network models from the recipe content. A single recipe can generate multiple networks
        (e.g. using different "inversions" i.e. input vs output markers).

        we build directly from the recipe content, without parsing the recipe file.
        """

        logger.debug(f"Building recipe {self.name}")

        lib = lib or load_lib()
        assert lib is not None

        errors = []

        def error_handler(msg):
            nonlocal errors
            import traceback

            msg = f"{msg}\n{traceback.format_exc()}"
            errors.append(msg)

        networks: Optional[ListOr[bc.network.Network]] = bc.recipe.network_from_recipe(
            None,
            lib,
            inverse=inverse,
            use_cache=use_cache,
            recipe_object=self.content,
            error_handler=error_handler,
        )

        if networks is None:
            logger.error(f"Recipe {self.name} did not yield any networks.")
            return []

        networks = networks if isinstance(networks, list) else [networks]
        for net in networks:
            net.metadata['recipe_name'] = self.name

        if errors:
            self.errors = "\n".join(errors)
            logger.error(f"Recipe {self.name} has {len(errors)} errors: {self.errors}")
            return []

        network_models = []

        for net in networks:
            network_info = bc.network.generate_network_info(net)
            network_info['recipe_name'] = self.name
            unique_name = f"{self.name}_{'-'.join(network_info['markers'])}"
            network = Network(
                name=unique_name,
                recipe_name=self.name,
                network_info=network_info,
            )
            net.name = unique_name
            network._network = net
            network_models.append(network)

        if add_to_self:
            self.networks.extend(network_models)

        return network_models


class TrainingSetLink(BiocompDB, table=True):
    trained_model_name: str = Field(foreign_key="trainedmodel.name", primary_key=True)
    dataset_name: Optional[str] = Field(foreign_key="dataset.name", primary_key=True)
    dataset_hash: str = Field(foreign_key="dataset.hash", primary_key=True)


class DataSetNetworkDataPair(BiocompDB, table=True):
    dataset_name: Optional[str] = Field(foreign_key="dataset.name", primary_key=True)
    dataset_hash: str = Field(foreign_key="dataset.hash", primary_key=True)
    network_name: str = Field(foreign_key="networkdatapair.network_name", primary_key=True)
    datafile_path: str = Field(foreign_key="networkdatapair.datafile_path", primary_key=True)


class DataSet(BiocompDB, table=True):
    name: Optional[str] = Field(primary_key=True, default=None)
    hash: str = Field(primary_key=True)

    # relationships
    trained_models: List["TrainedModel"] = Relationship(
        back_populates="training_dataset",
        link_model=TrainingSetLink,
        sa_relationship_kwargs={
            "primaryjoin": "and_(DataSet.name == TrainingSetLink.dataset_name, DataSet.hash == TrainingSetLink.dataset_hash)",
            "secondaryjoin": "TrainedModel.name == TrainingSetLink.trained_model_name",
        },
    )

    metrics: List["Metric"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Metric.on_dataset_name, Metric.on_dataset_hash]",
            "primaryjoin": "and_(DataSet.name == Metric.on_dataset_name, DataSet.hash == Metric.on_dataset_hash)",
        }
    )

    # auto relationship to get all NetworkDataPair objects in this dataset
    network_data_pairs: List["NetworkDataPair"] = Relationship(
        link_model=DataSetNetworkDataPair,
        sa_relationship_kwargs={
            "primaryjoin": "and_(DataSet.name == DataSetNetworkDataPair.dataset_name, DataSet.hash == DataSetNetworkDataPair.dataset_hash)",
            "secondaryjoin": "and_(NetworkDataPair.network_name == DataSetNetworkDataPair.network_name, NetworkDataPair.datafile_path == DataSetNetworkDataPair.datafile_path)",
        },
    )

    @classmethod
    def from_network_data_pairs(
        cls, network_data_pairs: List["NetworkDataPair"], name: Optional[str] = None
    ):
        """Create a DataSet from a list of NetworkDataPair objects"""
        # Create ordered string dump for hash computation
        pair_strings = []
        for pair in sorted(network_data_pairs, key=lambda x: (x.network_name, x.datafile_path)):
            pair_strings.append(f"{pair.network_name}:{pair.datafile_path}")

        content = "\n".join(pair_strings)
        hash_val = pronounceable_hash32(content.encode('utf-8'))

        return cls(name=name, hash=hash_val)


class NetworkDataPair(BiocompDB, table=True):
    network_name: str = Field(foreign_key="network.name", primary_key=True)
    datafile_path: str = Field(foreign_key="datafile.file", primary_key=True)

    network: Optional["Network"] = Relationship()
    datafile: Optional["DataFile"] = Relationship()

    # Remove old many-to-many relationship with TrainedModel
    # trained_models: List["TrainedModel"] = Relationship(...)

    # metrics computed on this specific network-data pair
    # metrics: List["Metric"] = Relationship()

    def __hash__(self):
        return hash((self.network_name, self.datafile_path))

    def __repr__(self):
        return (
            f"NetworkDataPair(network_name={self.network_name}, datafile_path={self.datafile_path})"
        )

    def __str__(self):
        return self.__repr__()


class TrainedModel(BiocompDB, table=True):
    name: str = Field(primary_key=True)
    path_to_model: ForcedStr
    run_name: Optional[str] = Field(default=None)
    experiment_name: Optional[str] = Field(default=None)
    end_loss: Optional[float] = Field(default=None)

    training_config: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # Direct relationship to training dataset
    training_dataset_name: Optional[str] = Field(foreign_key="dataset.name", default=None)
    training_dataset_hash: Optional[str] = Field(foreign_key="dataset.hash", default=None)

    training_dataset: Optional["DataSet"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[TrainedModel.training_dataset_name, TrainedModel.training_dataset_hash]",
            "primaryjoin": "and_(TrainedModel.training_dataset_name == DataSet.name, TrainedModel.training_dataset_hash == DataSet.hash)",
        }
    )

    metrics: List["Metric"] = Relationship(back_populates="trained_model")


class Metric(BiocompDB, table=True):
    # Auto-incrementing primary key
    id: Optional[int] = Field(default=None, primary_key=True)

    # Basic metric info
    name: str  # e.g. "RMSE", "grid_RMSE", etc.
    value: float
    n_points: Optional[int] = None  # number of points used to compute metric

    # Foreign key relationships (all optional)
    trained_model_name: Optional[str] = Field(foreign_key="trainedmodel.name", default=None)

    # Dataset this metric was computed on
    on_dataset_name: Optional[str] = Field(foreign_key="dataset.name", default=None)
    on_dataset_hash: Optional[str] = Field(foreign_key="dataset.hash", default=None)

    # Individual NetworkDataPair this metric was computed on
    on_network_name: Optional[str] = Field(foreign_key="networkdatapair.network_name", default=None)
    on_datafile_path: Optional[str] = Field(
        foreign_key="networkdatapair.datafile_path", default=None
    )

    # Plot that generated this metric (composite foreign key)
    source_plot_figure: Optional[str] = Field(default=None)
    source_plot_position: Optional[int] = Field(default=None)

    # metadata
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # relationships
    trained_model: Optional[TrainedModel] = Relationship(back_populates="metrics")
    dataset: Optional["DataSet"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Metric.on_dataset_name, Metric.on_dataset_hash]",
            "primaryjoin": "and_(DataSet.name == Metric.on_dataset_name, DataSet.hash == Metric.on_dataset_hash)",
            "overlaps": "metrics",
        }
    )
    network_data_pair: Optional["NetworkDataPair"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Metric.on_network_name, Metric.on_datafile_path]",
            "primaryjoin": "and_(NetworkDataPair.network_name == Metric.on_network_name, NetworkDataPair.datafile_path == Metric.on_datafile_path)",
        }
    )

    # We'll define this relationship with custom join conditions
    # source_plot: Optional["Plot"] = Relationship()


class AxPosition(BaseModel):
    row: int
    col: int


class Plot(BiocompDB, table=True):
    in_figure: str = Field(foreign_key="figure.file", primary_key=True)
    position: int = Field(default=0, primary_key=True)

    from_datafile: Optional[str] = Field(foreign_key="datafile.file", default=None)
    at_location: Optional[AxPosition] = Field(sa_column=Column(JSON))

    network_name: Optional[str] = None
    plot_method: Optional[str] = None
    input_names: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    output_name: Optional[str] = None
    datasource_type: Optional[str] = None
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # relationships
    datafile: Optional[DataFile] = Relationship(back_populates="plotted_in")
    # metrics: List["Metric"] = Relationship(back_populates="source_plot")


class Figure(BiocompDB, table=True):
    file: str = Field(primary_key=True)
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))


def get_biocompdb_sqlite_engine(db_path, echo=False):
    logger.debug(f"get_biocompdb_sqlite_engine({db_path}) was called")
    db_path = Path(db_path).expanduser().resolve()
    return create_engine(f"sqlite:///{db_path}", echo=echo)


def create_biocompdb_sqlite(db_path, echo=False):
    logger.debug(f"create_biocompdb_sqlite({db_path}) was called")
    engine = get_biocompdb_sqlite_engine(db_path, echo=echo)
    BiocompDB.metadata.create_all(engine)
