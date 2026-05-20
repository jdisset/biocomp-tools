# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator
from sqlalchemy import and_
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import with_loader_criteria
from sqlmodel import select, Session, col
from typing import Literal
from enum import Enum
from itertools import groupby
from biocomptools.toollib.models import (
    TrainedModel,
    NetworkDataPair,
    DataSet,
    DataSetNetworkDataPair,
    Metric,
)
from biocomptools.logging_config import get_logger
from biocomptools.toollib.networkselector import NetworkSet, Regex, iRegex, apply_regex_filter


logger = get_logger(__name__)


class LossOperator(str, Enum):
    LESS_THAN = "lt"
    GREATER_THAN = "gt"
    LESS_EQUAL = "le"
    GREATER_EQUAL = "ge"
    EQUAL = "eq"
    NOT_EQUAL = "ne"


class LossCriteria(BaseModel):
    """Criteria for filtering models based on loss value."""

    operator: LossOperator
    value: float

    def matches(self, loss: float | None) -> bool:
        """Check if a loss value matches this criteria."""
        if loss is None:
            return False

        if self.operator == LossOperator.LESS_THAN:
            return loss < self.value
        elif self.operator == LossOperator.GREATER_THAN:
            return loss > self.value
        elif self.operator == LossOperator.LESS_EQUAL:
            return loss <= self.value
        elif self.operator == LossOperator.GREATER_EQUAL:
            return loss >= self.value
        elif self.operator == LossOperator.EQUAL:
            return abs(loss - self.value) < 1e-9
        elif self.operator == LossOperator.NOT_EQUAL:
            return abs(loss - self.value) >= 1e-9
        return False


class TrainingSetCriteria(BaseModel):
    """Criteria for filtering models based on training set."""

    network_set: NetworkSet
    mode: Literal["exact", "includes"] = "includes"

    @property
    def _resolved_pairs(self) -> list[NetworkDataPair]:
        """Get resolved NetworkDataPair objects from the NetworkSet."""
        if not hasattr(self, '_cached_pairs'):
            # run selectors if not already done
            self.network_set.run_selectors()
            self._cached_pairs = self.network_set.content
        return self._cached_pairs

    def _get_dataset_pairs(
        self, dataset: DataSet | None, session: Session
    ) -> list[NetworkDataPair]:
        """Get NetworkDataPair objects associated with a DataSet."""
        if dataset is None:
            return []

        # Query DataSetNetworkDataPair junction table to get network data pairs
        query = (
            select(NetworkDataPair)
            .join(
                DataSetNetworkDataPair,
                (DataSetNetworkDataPair.network_name == NetworkDataPair.network_name)
                & (DataSetNetworkDataPair.datafile_path == NetworkDataPair.datafile_path),
            )
            .where(
                (DataSetNetworkDataPair.dataset_name == dataset.name)
                & (DataSetNetworkDataPair.dataset_hash == dataset.hash)
            )
        )

        return session.exec(query).all()

    def matches(self, dataset: DataSet | None, session: Session) -> bool:
        """Check if a model's training dataset matches this criteria."""
        model_pairs = self._get_dataset_pairs(dataset, session)
        required_pairs = set(self._resolved_pairs)
        model_pairs_set = set(model_pairs)

        if self.mode == "exact":
            return required_pairs == model_pairs_set
        elif self.mode == "includes":
            # model was trained on at least these pairs
            return required_pairs.issubset(model_pairs_set)
        return False


class ModelSelector(BaseModel):
    """
    Select trained models based on various criteria.

    Examples:
        # Select by exact name
        ModelSelector(name="model_v1_final")

        # Select by regex pattern
        ModelSelector(name=Regex("model_v[0-9]+.*"))

        # Select by case-insensitive regex pattern
        ModelSelector(name=iRegex("MODEL_v[0-9]+.*"))

        # Select by loss criteria
        ModelSelector(loss=LossCriteria(operator=LossOperator.LESS_THAN, value=0.01))

        # Select best model among matches
        ModelSelector(experiment_name="exp_2024", pick_best_loss=True)

        # Select models trained on specific network sets
        ModelSelector(
            training_set=TrainingSetCriteria(
                network_set=NetworkSet(...),
                mode="includes"
            )
        )

        # Group by field and pick best from each group
        ModelSelector(
            experiment_name="exp_2024",
            group_by="training_dataset_hash",
            pick_best_per_group=True
        )

        # Return all models matching criteria
        ModelSelector(
            run_name=Regex("run_.*"),
            pick_best_loss=False  # Returns all matches
        )
    """

    name: str | Regex | iRegex | None = None
    run_name: str | Regex | iRegex | None = None
    experiment_name: str | Regex | iRegex | None = None

    # loss criteria - multiple ways to specify
    loss: LossCriteria | None = None
    loss_less_than: float | None = None
    loss_greater_than: float | None = None
    pick_best_loss: bool = False

    # training set criteria
    training_set: TrainingSetCriteria | None = None
    trained_on_exact: NetworkSet | None = None
    trained_on_includes: NetworkSet | None = None

    # grouping and selection
    group_by: str | list[str] | None = None
    pick_best_per_group: bool = False

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

    @model_validator(mode='after')
    def validate_loss_criteria(self):
        """Ensure loss criteria are specified correctly."""
        # convert convenience fields to LossCriteria
        if self.loss_less_than is not None:
            if self.loss is not None:
                raise ValueError("Cannot specify both 'loss' and 'loss_less_than'")
            self.loss = LossCriteria(operator=LossOperator.LESS_THAN, value=self.loss_less_than)

        if self.loss_greater_than is not None:
            if self.loss is not None:
                raise ValueError("Cannot specify both 'loss' and 'loss_greater_than'")
            self.loss = LossCriteria(
                operator=LossOperator.GREATER_THAN, value=self.loss_greater_than
            )

        # validate training set criteria
        training_set_count = sum(
            [
                self.training_set is not None,
                self.trained_on_exact is not None,
                self.trained_on_includes is not None,
            ]
        )
        if training_set_count > 1:
            raise ValueError("Can only specify one training set criteria")

        # convert convenience fields to TrainingSetCriteria
        if self.trained_on_exact is not None:
            self.training_set = TrainingSetCriteria(network_set=self.trained_on_exact, mode="exact")
        elif self.trained_on_includes is not None:
            self.training_set = TrainingSetCriteria(
                network_set=self.trained_on_includes, mode="includes"
            )

        # validate grouping options
        if self.pick_best_per_group and not self.group_by:
            raise ValueError("pick_best_per_group requires group_by to be specified")

        # ensure group_by is a list
        if self.group_by and isinstance(self.group_by, str):
            self.group_by = [self.group_by]

        return self

    def get_models(
        self, session: Session | None = None, load_metrics_criteria: dict[str, object] | None = None
    ) -> list[TrainedModel]:
        sess = session or self.db_session
        close_session_locally = session is None

        try:
            # Start with the base query
            query = select(TrainedModel)

            # Always eagerly load the training dataset relationship
            query = query.options(
                selectinload(TrainedModel.training_dataset).selectinload(DataSet.network_data_pairs)
            )

            if load_metrics_criteria:
                metric_filters = []
                for key, value in load_metrics_criteria.items():
                    if key.endswith(".is_not"):
                        field_name = key.replace(".is_not", "")
                        metric_filters.append(getattr(Metric, field_name).is_not(value))
                    else:
                        metric_filters.append(getattr(Metric, key) == value)

                query = query.options(
                    selectinload(TrainedModel.metrics),
                    with_loader_criteria(TrainedModel.metrics, lambda _: and_(*metric_filters)),
                )

            if self.name:
                query = apply_regex_filter(query, col(TrainedModel.name), self.name)
            if self.run_name:
                query = apply_regex_filter(query, col(TrainedModel.run_name), self.run_name)
            if self.experiment_name:
                query = apply_regex_filter(
                    query, col(TrainedModel.experiment_name), self.experiment_name
                )

            if self.loss:
                op_map = {
                    LossOperator.LESS_THAN: "__lt__",
                    LossOperator.GREATER_THAN: "__gt__",
                    LossOperator.LESS_EQUAL: "__le__",
                    LossOperator.GREATER_EQUAL: "__ge__",
                    LossOperator.EQUAL: "__eq__",
                    LossOperator.NOT_EQUAL: "__ne__",
                }
                operator_method = getattr(col(TrainedModel.end_loss), op_map[self.loss.operator])
                query = query.where(operator_method(self.loss.value))

            models = sess.exec(query).all()

            # Post-query filtering for complex criteria that are hard to express in SQL
            if self.training_set:
                filtered_models = []
                for model in models:
                    if self.training_set.matches(model.training_dataset, sess):
                        filtered_models.append(model)
                models = filtered_models

            if self.group_by and models:
                models = self._apply_grouping(models)
            elif self.pick_best_loss and models:
                models_with_loss = [m for m in models if m.end_loss is not None]
                if models_with_loss:
                    best_model = min(models_with_loss, key=lambda m: m.end_loss)
                    models = [best_model]
                else:
                    models = []

            return models

        except Exception as e:
            logger.exception(e)
            raise
        finally:
            if close_session_locally:
                sess.close()

    def _apply_grouping(self, models: list[TrainedModel]) -> list[TrainedModel]:
        """Apply grouping logic to models."""
        if not self.group_by:
            return models

        logger.debug(f"Grouping models by: {self.group_by}")

        # Optimize for the common case of grouping by training_dataset_hash
        if self.group_by == ["training_dataset_hash"]:
            # Use defaultdict for O(1) grouping instead of sorting + groupby
            from collections import defaultdict

            groups = defaultdict(list)

            for model in models:
                # Extract the hash once per model
                if model.training_dataset:
                    key = model.training_dataset.hash
                else:
                    key = None
                groups[key].append(model)

            # Process groups
            result_models = []
            for group_key, group_list in groups.items():
                logger.debug(f"Group {group_key}: {len(group_list)} models")

                if self.pick_best_per_group:
                    # Pick best model from this group
                    models_with_loss = [m for m in group_list if m.end_loss is not None]
                    if models_with_loss:
                        best_model = min(models_with_loss, key=lambda m: m.end_loss)
                        result_models.append(best_model)
                        logger.debug(
                            f"Selected best model from group: {best_model.name} (loss: {best_model.end_loss})"
                        )
                    else:
                        # If no models have loss, just take the first one
                        result_models.append(group_list[0])
                        logger.warning(
                            f"No models with loss in group {group_key}, taking first model"
                        )
                else:
                    # Include all models from this group
                    result_models.extend(group_list)

            logger.debug(f"After grouping: {len(result_models)} models selected")
            return result_models

        # General case for other grouping fields
        # Create a key function that extracts values for all group_by fields
        def make_group_key(model):
            values = []
            for field in self.group_by:
                if hasattr(model, field):
                    value = getattr(model, field)
                    # Handle special cases for related fields
                    if field == 'training_dataset' and value:
                        # Use both name and hash for dataset grouping
                        values.append((value.name, value.hash))
                    else:
                        values.append(value)
                else:
                    logger.warning(f"Model does not have field '{field}' for grouping")
                    values.append(None)
            return tuple(values)

        # Sort models by group key to enable groupby
        sorted_models = sorted(models, key=make_group_key)

        # Group models and optionally pick best from each group
        result_models = []
        for group_key, group_models in groupby(sorted_models, key=make_group_key):
            group_list = list(group_models)
            logger.debug(f"Group {group_key}: {len(group_list)} models")

            if self.pick_best_per_group:
                # Pick best model from this group
                models_with_loss = [m for m in group_list if m.end_loss is not None]
                if models_with_loss:
                    best_model = min(models_with_loss, key=lambda m: m.end_loss)
                    result_models.append(best_model)
                    logger.debug(
                        f"Selected best model from group: {best_model.name} (loss: {best_model.end_loss})"
                    )
                else:
                    # If no models have loss, just take the first one
                    result_models.append(group_list[0])
                    logger.warning(f"No models with loss in group {group_key}, taking first model")
            else:
                # Include all models from this group
                result_models.extend(group_list)

        logger.debug(f"After grouping: {len(result_models)} models selected")
        return result_models

    def get_model(self, session: Session | None = None, raise_if_multiple: bool = True) -> TrainedModel:
        from pathlib import Path
        from biocomptools.toollib.models import create_trained_model_from_file

        if self.name and isinstance(self.name, str):
            if '/' in self.name or '\\' in self.name or self.name.endswith('.pickle'):
                file_path = Path(self.name)
                if file_path.exists():
                    trained_model, _, _ = create_trained_model_from_file(file_path)
                    return trained_model

        sess = session or self.db_session
        close_session_locally = session is None

        try:
            models = self.get_models(sess)
            if raise_if_multiple and len(models) > 1:
                raise ValueError(
                    f"Multiple models found for selector: {self}. Use get_models() instead."
                )
            if not models:
                raise ValueError(f"No models found for selector: {self}")
            return models[0]
        finally:
            if close_session_locally:
                sess.close()

    @classmethod
    def best_per_training_set(
        cls,
        name: str | Regex | iRegex | None = None,
        run_name: str | Regex | iRegex | None = None,
        experiment_name: str | Regex | iRegex | None = None,
        **kwargs: object,
    ) -> ModelSelector:
        """
        Create a selector that returns the best model for each unique training set.

        Example:
            selector = ModelSelector.best_per_training_set(experiment_name="exp_2024")
            models = selector.get_models(session)  # Returns best model for each training dataset
        """
        return cls(
            name=name,
            run_name=run_name,
            experiment_name=experiment_name,
            group_by="training_dataset_hash",
            pick_best_per_group=True,
            **kwargs,
        )

    @classmethod
    def best_per_field(
        cls,
        field: str | list[str],
        name: str | Regex | iRegex | None = None,
        run_name: str | Regex | iRegex | None = None,
        experiment_name: str | Regex | iRegex | None = None,
        **kwargs: object,
    ) -> ModelSelector:
        """
        Create a selector that returns the best model for each unique value of the specified field(s).

        Args:
            field: Field name(s) to group by (e.g., "run_name", ["experiment_name", "run_name"])

        Example:
            # Get best model for each run
            selector = ModelSelector.best_per_field("run_name", experiment_name="exp_2024")
            models = selector.get_models(session)
        """
        return cls(
            name=name,
            run_name=run_name,
            experiment_name=experiment_name,
            group_by=field,
            pick_best_per_group=True,
            **kwargs,
        )

    @classmethod
    def all_matching(
        cls,
        name: str | Regex | iRegex | None = None,
        run_name: str | Regex | iRegex | None = None,
        experiment_name: str | Regex | iRegex | None = None,
        **kwargs: object,
    ) -> ModelSelector:
        """
        Create a selector that returns all models matching the criteria.

        Example:
            # Get all models from an experiment
            selector = ModelSelector.all_matching(experiment_name="exp_2024")
            models = selector.get_models(session)
        """
        return cls(
            name=name,
            run_name=run_name,
            experiment_name=experiment_name,
            pick_best_loss=False,
            **kwargs,
        )


class ModelSet(BaseModel):
    """
    A set of trained models, similar to NetworkSet.
    Can contain ModelSelectors or individual model references.
    """

    content: list[TrainedModel | ModelSelector | ModelSet] = []

    @property
    def _engine(self):
        """Lazy-load the database engine when needed."""
        from biocomptools.toollib.models import get_biocompdb_sqlite_engine
        from biocomptools.toollib.common import config

        return get_biocompdb_sqlite_engine(config.db.sqlite.path)

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
    def route_content(cls, v: object, info: object) -> list[TrainedModel | ModelSelector | ModelSet]:
        """Route content items to appropriate types."""
        logger.debug(f"Routing ModelSet content: {v}")

        if isinstance(v, (TrainedModel, ModelSelector, cls)):
            v = [v]
        elif not isinstance(v, list):
            raise TypeError(f"ModelSet content must be a list, got {type(v)}")

        def route_item(item: object) -> TrainedModel | ModelSelector | ModelSet:
            if isinstance(item, (TrainedModel, ModelSelector, cls)):
                return item
            elif isinstance(item, dict):
                if 'path_to_model' in item:
                    return TrainedModel(**item)
                elif 'content' in item:
                    return cls(**item)
                else:
                    return ModelSelector(**item)
            else:
                raise TypeError(f"Invalid item in ModelSet: {type(item)}")

        return [route_item(item) for item in v]

    def run_selectors(self, session: Session | None = None) -> None:
        """Resolve all ModelSelectors to TrainedModel instances."""
        sess = session or self.db_session
        close_session = session is None

        logger.debug(f"Running selectors on {len(self.content)} items")
        new_content = []

        # Track unique models to avoid duplicates
        seen_models = set()

        for item in self.content:
            if isinstance(item, ModelSelector):
                logger.debug(f"Running selector: {item}")
                models = item.get_models(sess)
                for model in models:
                    if model.name not in seen_models:
                        new_content.append(model)
                        seen_models.add(model.name)
                logger.debug(f"Found {len(models)} matching models")
            elif isinstance(item, ModelSet):
                # recursively run selectors on nested ModelSets
                logger.debug("Running nested ModelSet")
                item.run_selectors(sess)
                for model in item.content:
                    if model.name not in seen_models:
                        new_content.append(model)
                        seen_models.add(model.name)
            else:
                assert isinstance(item, TrainedModel)
                if item.name not in seen_models:
                    new_content.append(item)
                    seen_models.add(item.name)

        self.content = new_content
        logger.debug(f"Finished running selectors. Found {len(self.content)} models")

        if close_session:
            sess.close()

    def get_models(self, session: Session | None = None) -> list[TrainedModel]:
        """Get all TrainedModel instances in this set."""
        # ensure selectors are run
        if any(isinstance(item, (ModelSelector, ModelSet)) for item in self.content):
            self.run_selectors(session)

        return self.content

    def __len__(self):
        return len(self.content)

    def __repr__(self):
        return f"{self.__class__.__name__}[{len(self.content)} items]"
