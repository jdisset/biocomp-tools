from sqlmodel import Field, SQLModel, create_engine, Relationship, Session
from typing import List, Optional, Annotated, Any
import sqlalchemy as sa
from sqlalchemy import Column, JSON
import datetime
from pydantic import BaseModel, BeforeValidator

def to_str(data: Any) -> Any:
    if not isinstance(data, str) and data is not None:
        return str(data)
    return data

ForcedStr = Annotated[str, BeforeValidator(to_str)]
ForcedOptionalStr = Annotated[Optional[str], BeforeValidator(to_str)]


class BiocompDB(SQLModel, registry=sa.orm.registry()):
    pass


class Collection(BiocompDB, table=True):
    name: str = Field(primary_key=True)
    description: Optional[str] = None

    networks: List["CollectionNetwork"] = Relationship(back_populates="collection")


class Experiment(BiocompDB, table=True):
    name: str = Field(primary_key=True)
    path: ForcedStr
    transfection_date: Optional[str] = None
    recipe_errors: Optional[str] = None
    network_building_errors: Optional[str] = None
    data_loading_errors: Optional[str] = None
    calibration_version: Optional[str] = None
    has_calibration_diagnostics: Optional[bool] = None
    comments: Optional[str] = None

    networks: List["Network"] = Relationship(back_populates="experiment")


class TrainingRun(BiocompDB, table=True):
    name: str = Field(default=None, primary_key=True)
    date_started: Optional[datetime.date] = Field(default_factory=datetime.date.today)
    duration: Optional[float] = None
    training_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    wb_project: str
    wb_run_name: str
    artifact_path: ForcedOptionalStr
    end_loss: Optional[float] = None
    base_compute_config_name: Optional[str] = None
    biocomp_git_hash: str
    biocomp_version: str
    compute_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    data_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    description: Optional[str] = None
    wb_run_id: Optional[str] = None
    best_replicate: Optional[int] = None
    export_dir: Optional[str] = None

    predictions: List["Prediction"] = Relationship(back_populates="training_run")

class Network(BiocompDB, table=True):
    name: str = Field(primary_key=True)
    xp: str = Field(foreign_key="experiment.name")
    sample_name: str
    data_file: ForcedOptionalStr = None
    recipe_name: str
    recipe_file: ForcedOptionalStr = None
    network_info: dict = Field(default_factory=dict, sa_column=Column(JSON))

    data_plot: ForcedOptionalStr = None
    comments: Optional[str] = None
    data_quality: Optional[int] = Field(default=-1)
    plot_error: Optional[str] = None

    experiment: Optional[Experiment] = Relationship(back_populates="networks")
    collections: List["CollectionNetwork"] = Relationship(back_populates="network")
    predictions: List["Prediction"] = Relationship(back_populates="network")

    def generate_unique_name(self):
        # n = f'{row["recipe_name"]}_{row["xp"]}_{"-".join(row["network_info"]["markers"].split(", "))}'
        return f"{self.recipe_name}_{self.xp}_{'-'.join(self.network_info['markers'])}"


class Prediction(BiocompDB, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    plot_path: ForcedStr
    pred_error: Optional[str] = None
    training_run_name: str = Field(foreign_key="trainingrun.name")
    network_name: str = Field(foreign_key="network.name")

    training_run: Optional[TrainingRun] = Relationship(back_populates="predictions")
    network: Optional[Network] = Relationship(back_populates="predictions")


class CollectionNetwork(BiocompDB, table=True):
    collection_name: str = Field(foreign_key="collection.name", primary_key=True)
    network_name: str = Field(foreign_key="network.name", primary_key=True)

    collection: Optional[Collection] = Relationship(back_populates="networks")
    network: Optional[Network] = Relationship(back_populates="collections")

def get_biocompdb_sqlite_engine(db_path, echo=False):
    return create_engine(f"sqlite:///{db_path}", echo=echo)

def create_biocompdb_sqlite(db_path, echo=False):
    engine = get_biocompdb_sqlite_engine(db_path, echo=echo)
    BiocompDB.metadata.create_all(engine)

