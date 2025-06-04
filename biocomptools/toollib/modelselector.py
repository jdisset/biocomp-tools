from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import selectinload
from sqlalchemy import func
from sqlmodel import select, Session, col
from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum
from biocomptools.toollib.models import TrainedModel, NetworkDataPair, TrainingSetLink
from biocomptools.logging_config import get_logger
from biocomptools.toollib.networkselector import NetworkSet, Regex
from sqlalchemy.exc import SQLAlchemyError

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

    def matches(self, loss: Optional[float]) -> bool:
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
    def _resolved_pairs(self) -> List[NetworkDataPair]:
        """Get resolved NetworkDataPair objects from the NetworkSet."""
        if not hasattr(self, '_cached_pairs'):
            # run selectors if not already done
            self.network_set.run_selectors()
            self._cached_pairs = self.network_set.content
        return self._cached_pairs

    def matches(self, model_training_set: List[NetworkDataPair]) -> bool:
        """Check if a model's training set matches this criteria."""
        required_pairs = set(self._resolved_pairs)
        model_pairs = set(model_training_set)

        if self.mode == "exact":
            return required_pairs == model_pairs
        elif self.mode == "includes":
            # model was trained on at least these pairs
            return required_pairs.issubset(model_pairs)
        return False


class ModelSelector(BaseModel):
    """
    Select trained models based on various criteria.

    Examples:
        # Select by exact name
        ModelSelector(name="model_v1_final")

        # Select by regex pattern
        ModelSelector(name=Regex("model_v[0-9]+.*"))

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
    """

    name: Optional[Union[str, Regex]] = None
    run_name: Optional[Union[str, Regex]] = None
    experiment_name: Optional[Union[str, Regex]] = None

    # loss criteria - multiple ways to specify
    loss: Optional[LossCriteria] = None
    loss_less_than: Optional[float] = None
    loss_greater_than: Optional[float] = None
    pick_best_loss: bool = False

    # training set criteria
    training_set: Optional[TrainingSetCriteria] = None
    trained_on_exact: Optional[NetworkSet] = None
    trained_on_includes: Optional[NetworkSet] = None

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

        return self

    def get_models(self, session: Session) -> List[TrainedModel]:
        """
        Retrieve trained models based on specified filters.

        Args:
            session: The database session

        Returns:
            List[TrainedModel]: List of trained models matching criteria

        Raises:
            ValueError: If no models are found or query execution fails
            SQLAlchemyError: For database-related errors
        """
        try:
            # build base query with eager loading of relationships
            query = select(TrainedModel).options(selectinload(TrainedModel.training_set))

            # apply name filters
            if self.name:
                logger.debug(f"Applying name filter: {self.name}")
                if isinstance(self.name, Regex):
                    query = query.where(col(TrainedModel.name).regexp_match(self.name))
                else:
                    query = query.where(TrainedModel.name == self.name)

            if self.run_name:
                logger.debug(f"Applying run_name filter: {self.run_name}")
                if isinstance(self.run_name, Regex):
                    query = query.where(col(TrainedModel.run_name).regexp_match(self.run_name))
                else:
                    query = query.where(TrainedModel.run_name == self.run_name)

            if self.experiment_name:
                logger.debug(f"Applying experiment_name filter: {self.experiment_name}")
                if isinstance(self.experiment_name, Regex):
                    query = query.where(
                        col(TrainedModel.experiment_name).regexp_match(self.experiment_name)
                    )
                else:
                    query = query.where(TrainedModel.experiment_name == self.experiment_name)

            # execute query
            logger.debug(f"Executing model query: {query}")
            try:
                models = session.exec(query).all()
            except SQLAlchemyError as e:
                logger.error(f"Database error while executing model query: {str(e)}")
                raise ValueError(f"Failed to execute model query: {str(e)}") from e

            if not models:
                logger.warning(f"No models found for selector: {self}")

            # apply loss criteria (post-query filtering)
            if self.loss:
                logger.debug(f"Applying loss criteria: {self.loss}")
                models = [m for m in models if self.loss.matches(m.end_loss)]

            # apply training set criteria (post-query filtering)
            if self.training_set:
                logger.debug(f"Applying training set criteria: {self.training_set}")
                filtered_models = []
                for model in models:
                    # need to ensure training_set is loaded
                    if self.training_set.matches(model.training_set):
                        filtered_models.append(model)
                models = filtered_models

            # pick best loss if requested
            if self.pick_best_loss and models:
                logger.debug("Picking model with best (lowest) loss")
                # filter out models without loss values
                models_with_loss = [m for m in models if m.end_loss is not None]
                if models_with_loss:
                    best_model = min(models_with_loss, key=lambda m: m.end_loss)
                    models = [best_model]
                else:
                    logger.warning("No models with loss values found, cannot pick best")
                    models = []

            logger.debug(f"Model selection complete. Found {len(models)} models.")
            return models

        except Exception as e:
            logger.exception(e)
            raise

    def get_model(self, session: Session, raise_if_multiple: bool = True) -> Optional[TrainedModel]:
        models = self.get_models(session)
        if raise_if_multiple and len(models) > 1:
            raise ValueError(
                f"Multiple models found for selector: {self}. Use get_models() instead."
            )
        return models[0] if models else None


class ModelSet(BaseModel):
    """
    A set of trained models, similar to NetworkSet.
    Can contain ModelSelectors or individual model references.
    """

    content: List[Union[TrainedModel, ModelSelector, "ModelSet"]] = []

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
    def route_content(cls, v: Any, info):
        """Route content items to appropriate types."""
        logger.debug(f"Routing ModelSet content: {v}")

        if isinstance(v, (TrainedModel, ModelSelector, cls)):
            v = [v]
        elif not isinstance(v, list):
            raise TypeError(f"ModelSet content must be a list, got {type(v)}")

        def route_item(item: Any) -> Union[TrainedModel, ModelSelector, "ModelSet"]:
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

    def run_selectors(self, session=None):
        """Resolve all ModelSelectors to TrainedModel instances."""
        sess = session or self.db_session
        close_session = session is None

        logger.debug(f"Running selectors on {len(self.content)} items")
        new_content = []

        for item in self.content:
            if isinstance(item, ModelSelector):
                logger.debug(f"Running selector: {item}")
                models = item.get_models(sess)
                new_content.extend(models)
                logger.debug(f"Found {len(models)} matching models")
            elif isinstance(item, ModelSet):
                # recursively run selectors on nested ModelSets
                logger.debug(f"Running nested ModelSet")
                item.run_selectors(sess)
                new_content.extend(item.content)
            else:
                assert isinstance(item, TrainedModel)
                new_content.append(item)

        self.content = new_content
        logger.debug(f"Finished running selectors. Found {len(self.content)} models")

        if close_session:
            sess.close()

    def get_models(self, session=None) -> List[TrainedModel]:
        """Get all TrainedModel instances in this set."""
        # ensure selectors are run
        if any(isinstance(item, (ModelSelector, ModelSet)) for item in self.content):
            self.run_selectors(session)

        return self.content

    def __len__(self):
        return len(self.content)

    def __repr__(self):
        return f"{self.__class__.__name__}[{len(self.content)} items]"
