"""Recipe prediction utilities for uniform vs experimental X sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, PrivateAttr, model_validator

if TYPE_CHECKING:
    from biocomp.plotutils import PlotData


class RecipePredictionData(BaseModel):
    """Data holder for recipe predictions with two sampling strategies.

    Used by Dracon YAML: !biocomptools.toollib.figuremakers.recipepredictutils.RecipePredictionData
    """

    recipe_path: str
    model_path: str
    resolution: int = 50
    n_samples: int = 5000
    seed: int = 42

    _uniform_data: PlotData = PrivateAttr()
    _experimental_data: PlotData = PrivateAttr()
    _network_name: str = PrivateAttr(default="")

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode='after')
    def _compute(self):
        import dracon as dr
        from biocomp.recipe import Recipe
        from biocomp.network import recipe_to_networks
        from biocomptools.modelmodel import BiocompModel, NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction, make_hypercube
        from biocomptools.toollib.typical_experimental_distribution import sample_latent

        model = BiocompModel.load(self.model_path)

        with open(self.recipe_path, 'r') as f:
            content = f.read()
        if '\n!biocomp.recipe.Recipe' in content:
            idx = content.index('\n!biocomp.recipe.Recipe')
            recipe_part = content[idx + 1:]
        elif content.startswith('!biocomp.recipe.Recipe'):
            recipe_part = content
        else:
            recipe_part = content
        recipe_context = {'Recipe': Recipe, 'biocomp.recipe.Recipe': Recipe}
        recipe = dr.loads(recipe_part, context=recipe_context)
        if hasattr(recipe, 'get'):
            recipe = recipe.get('recipe', recipe)

        networks = recipe_to_networks(recipe, invert=True, inversion_mode='main')
        network = networks[0]
        self._network_name = network.name
        ndim = network.nb_inputs

        if ndim > 2:
            raise ValueError(
                f"Recipe '{network.name}' has {ndim} inputs, "
                f"but experimental sampling only supports 1D/2D"
            )

        nm = NetworkModel(model=model, network=[network])

        X_uniform = make_hypercube(ndim, res=self.resolution)
        X_exp = sample_latent(self.n_samples, ndim, seed=self.seed)

        pred_uniform = NetworkPrediction(
            predict_at=[X_uniform],
            network_model=nm,
            already_latent=True,
            enable_gridstats=False,
            skip_input_reorder=True,
        )
        self._uniform_data = pred_uniform.get_data(rescale_latent=True)[0]
        self._uniform_data.metadata.update(
            {
                'network_name': network.name,
                'built_network': network,
                'sampling': 'uniform',
            }
        )

        pred_exp = NetworkPrediction(
            predict_at=[X_exp],
            network_model=nm,
            already_latent=True,
            enable_gridstats=False,
            skip_input_reorder=True,
        )
        self._experimental_data = pred_exp.get_data(rescale_latent=True)[0]
        self._experimental_data.metadata.update(
            {
                'network_name': network.name,
                'built_network': network,
                'sampling': 'experimental',
            }
        )

        return self

    @property
    def uniform_data(self) -> PlotData:
        return self._uniform_data

    @property
    def experimental_data(self) -> PlotData:
        return self._experimental_data

    @property
    def network_name(self) -> str:
        return self._network_name
