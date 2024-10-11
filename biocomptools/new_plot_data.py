# {{{                        --     imports     -
import ray
import biocomp.utils as ut
import biocomptools.toollib.plot as pl
from pathlib import Path
import hydra
import rich
import logging
import argparse
from omegaconf import OmegaConf
import dracon as dr
def setup_logging(loglevel=logging.WARNING):
    import warnings

    warnings.filterwarnings("ignore", message=".*Defaults list is missing")
    warnings.filterwarnings("ignore", message=".*fork() was called")
    import logging

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=loglevel,
    )
    for name in [
        'biocomp',
        'matplotlib',
        'PIL',
        'biocomptools',
        'hydra',
        'omegaconf',
        'jax',
        'ray',
    ]:
        ut.set_loglevel(name, loglevel)


setup_logging()


##────────────────────────────────────────────────────────────────────────────}}}
"""
- [ ] Replace custom Resolvable class with Dracon's Resolvable.
>  Update imports and remove the custom class definition.

- [ ] Update Pydantic models to use Resolvable[T] from Dracon.
>  Adjust type hints and default values.

- [ ] Remove custom merging functions and use Dracon's merged function.
>  Replace calls to merged, merged_into, etc., with Dracon's merged.

- [ ] Adjust inheritance logic to use Dracon's merging in YAML.
>  Simplify or remove InheritableAttrsModel. Handle inheritance directly in YAML configurations.

- [ ] Replace context management code with Dracon's context passing in .resolve().
>  Update functions that pass context to use Dracon's resolve(context=...).

- [ ] Remove OmegaConf resolver registrations and related code.
>  Eliminate functions and classes related to OmegaConf resolvers.

- [ ] Replace OmegaConf functions with Dracon equivalents.
>  Update functions that create or manipulate configurations.

- [ ] Update type hints and remove OmegaConf imports.
>  Replace DictConfig and ListConfig with Mapping and Sequence.

- [ ] Update YAML configuration files to use Dracon's merge operators and interpolation syntax.
>  Adjust merging and variable interpolation to Dracon's syntax.

- [ ] Run and update unit tests to ensure functionality.
>  Ensure all tests pass with the updated code.

- [ ] Verify that the plotting library functions correctly with Dracon.
>  Perform integration tests and validate outputs.

"""




