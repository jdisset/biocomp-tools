from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Date,
    create_engine,
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import JSON
import datetime

Base = declarative_base()


class Collection(Base):
    __tablename__ = 'collections'

    name = Column(String, primary_key=True)
    description = Column(Text, nullable=True)

    networks = relationship("CollectionNetwork", back_populates="collection")


class Configuration(Base):
    __tablename__ = 'configurations'

    name = Column(String, primary_key=True)
    description = Column(Text, nullable=True)
    config_type = Column(String, nullable=False)
    config = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=True)


class Experiment(Base):
    __tablename__ = 'experiment'

    name = Column(String, primary_key=True)
    path = Column(String, nullable=False)
    transfection_date = Column(Text, nullable=True)
    recipe_errors = Column(Text, nullable=True)
    network_building_errors = Column(Text, nullable=True)
    data_loading_errors = Column(Text, nullable=True)
    calibration_version = Column(Text, nullable=True)
    has_calibration_diagnostics = Column(Boolean, nullable=True)
    comments = Column(Text, nullable=True)

    networks = relationship("Network", back_populates="experiment")


class TrainingRun(Base):
    __tablename__ = 'training_run'

    name = Column(String, primary_key=True)
    date_started = Column(Date, default=datetime.date.today, nullable=True)
    duration = Column(Float, nullable=True)
    training_config = Column(JSON, nullable=False)
    wb_project = Column(String, nullable=False)
    wb_run_name = Column(String, nullable=False)
    model_path = Column(String, nullable=True)
    end_loss = Column(Float, nullable=True)
    base_compute_config_name = Column(String, nullable=True)
    biocomp_git_hash = Column(String, nullable=False)
    biocomp_version = Column(String, nullable=False)
    compute_config = Column(JSON, nullable=False)
    data_config = Column(JSON, nullable=False)
    description = Column(String, nullable=True)
    wb_run_id = Column(String, nullable=True)
    best_replicate = Column(Integer, nullable=True)
    export_dir = Column(String, nullable=True)

    predictions = relationship("Prediction", back_populates="training_run")


class Network(Base):
    __tablename__ = 'network'

    name = Column(String, primary_key=True)
    xp = Column(String, ForeignKey('experiment.name'), nullable=False)
    sample_name = Column(String, nullable=False)
    data_file = Column(String, nullable=True)
    recipe_name = Column(String, nullable=False)
    recipe_file = Column(String, nullable=True)
    sequestron_type = Column(String, nullable=False)
    architecture = Column(String, nullable=False)
    ern_names = Column(Text, nullable=True)
    uorf_values = Column(Text, nullable=True)
    uorf_names = Column(Text, nullable=True)
    genes = Column(Text, nullable=True)
    markers = Column(Text, nullable=True)
    output_proteins = Column(Text, nullable=True)
    data_plot = Column(String, nullable=True)
    comments = Column(Text, nullable=True)
    data_quality = Column(Integer, default=-1, nullable=True)
    plot_error = Column(String, nullable=True)

    experiment = relationship("Experiment", back_populates="networks")
    predictions = relationship("Prediction", back_populates="network")
    collections = relationship("CollectionNetwork", back_populates="network")


class Prediction(Base):
    __tablename__ = 'prediction'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plot_path = Column(String, nullable=True)
    pred_error = Column(String, nullable=True)
    training_run_name = Column(String, ForeignKey('training_run.name'), nullable=False)
    network_name = Column(String, ForeignKey('network.name'), nullable=False)

    training_run = relationship("TrainingRun", back_populates="predictions")
    network = relationship("Network", back_populates="predictions")


class CollectionNetwork(Base):
    __tablename__ = 'collection_network'

    collection_name = Column(String, ForeignKey('collections.name'), primary_key=True)
    network_name = Column(String, ForeignKey('network.name'), primary_key=True)

    collection = relationship("Collection", back_populates="networks")
    network = relationship("Network", back_populates="collections")
