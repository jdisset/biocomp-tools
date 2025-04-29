from sqlmodel import Field, SQLModel, create_engine, Relationship
from typing import List, Optional, Annotated, Any, TypeVar
import sqlalchemy as sa
from sqlalchemy import Column, JSON
from pydantic import BeforeValidator
from pathlib import Path
from biocomptools.toollib.common import config
import biocomp as bc
from biocomp.utils import load_lib

from biocomptools.logging_config import get_logger

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

        # if self.calibration is not None:
        #     new_obj.calibration = self.calibration.safe_copy()

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

        networks: ListOr[bc.network.Network] = bc.recipe.network_from_recipe(
            None,
            lib,
            inverse=inverse,
            use_cache=use_cache,
            recipe_object=self.content,
            error_handler=error_handler,
        )
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


def get_biocompdb_sqlite_engine(db_path, echo=False):
    logger.debug(f"Sqlite engine from {db_path}")
    db_path = Path(db_path).expanduser().resolve()
    return create_engine(f"sqlite:///{db_path}", echo=echo)


def create_biocompdb_sqlite(db_path, echo=False):
    engine = get_biocompdb_sqlite_engine(db_path, echo=echo)
    BiocompDB.metadata.create_all(engine)
