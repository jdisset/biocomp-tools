## {{{                         --     docstring     --
"""
    Classes can declare some attributes as being inheritable by some designated
    children objects. The idea is that you want things like metadata, context,
    or plot_config to be resolved at the last minute, and to able to be
    modified at any level of the hierarchy (for example PlotJob -> DataSource
    -> FigureTask -> PlotTask).

    So let's say we create a PlotJob from a config file, and in it we have a
    little bit of metadata information (like the name of the job, the author,
    etc.). We also have a PlotConfig object that contains the default rc_params
    for matplotlib, and a figure_maker that uses the default FigureMaker. We
    then have data_sources that are created and that can override all these
    attributes.

    First we initialize PlotJob with all the attributes as Resolvable objects
    (they keep their DictConfig representation until resolved).

    When accessing an attribute, let's say job.data_source, we are going to
    trigger its resolution. Before calling resolve() and therefore losing the
    DictConfig representation of the data_source attribute, we want to *check if
    it's been flagged as child that should inherit some attributes from the
    parent*.

    If yes, we want - prior to resolution - to merge the parent's attribute
    (metadata, context, etc.) with the config of the child attribute. Then we
    can resolve it.
"""


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                          --     imports     --
from contextlib import contextmanager
from biocomp import utils as ut
from copy import deepcopy
from functools import partial
from typing import (
    Dict,
    Any,
    Callable,
    TypeVar,
)
import logging
from rich.logging import RichHandler
from omegaconf import DictConfig, OmegaConf, open_dict
from biocomptools.toollib.common import DictLike, ListOrSingle
from biocomptools.toollib.common import dict_like, open_dictlike
from biocomptools.toollib import common as cm

from biocomptools.toollib.resolvable import open_dictlike
from biocomptools.toollib.resolvable import Resolvable

##────────────────────────────────────────────────────────────────────────────}}}
## {{{                           --     types     --
T = TypeVar('T')
U = TypeVar('U')
##────────────────────────────────────────────────────────────────────────────}}}
## {{{                       --     logger utils --
LOGFORMAT = "in %(funcName)s: %(message)s"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level="NOTSET", format=LOGFORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger('biocomptools.biocomplot.utils.inheritable')
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     inherit utils     --

D = TypeVar('D', Dict, DictConfig)

# a set of utils to transparently treat dict-like objects and resolvables as mergeable objects


def config_holder(obj) -> bool:
    return hasattr(obj, 'config') and (dict_like(obj.config) or obj.config is None)


def as_dict(obj) -> Dict:
    if obj is None:
        return {}
    if isinstance(obj, DictConfig):
        d = OmegaConf.to_container(obj, resolve=False)
        assert isinstance(d, dict), f'Invalid dict-like object {d}'
        return d
    if config_holder(obj):
        return as_dict(obj.config)
    return obj


def merged(parent: DictLike, child: D, **kw) -> D:
    """Merge parent DictLike structure into child,
    return the result without modifying child,
    and in the same type as child.
    """

    # TODO: handle lists?

    merged = ut.updated_dict(as_dict(parent), as_dict(child), **kw)
    assert isinstance(merged, dict), f'Invalid merged object {merged}'

    if isinstance(child, DictConfig):
        return OmegaConf.create(merged)

    return merged


def get_dict_attr(obj: DictLike, attr_name: str) -> Any:
    try:
        return obj[attr_name]
    except TypeError:
        return getattr(obj, attr_name)


def set_dict_attr(obj: DictLike, attr_name: str, value: Any):
    try:
        obj[attr_name] = value
    except TypeError:
        setattr(obj, attr_name, value)


@contextmanager
def open_dictlike(obj: Any):
    if isinstance(obj, DictConfig):
        with open_dict(obj):
            yield
    else:
        yield


def inplace_merge_into(target: Any, parent: Any, attr_names: ListOrSingle[str], **kw):
    """merge parent[attr_name] into target[attr_name]"""

    if isinstance(attr_names, str):
        attr_names = [attr_names]

    for attr_name in attr_names:
        try:
            parent_attr = get_dict_attr(parent, attr_name)
        except KeyError:
            log.debug('No parent attribute %s found in %s', attr_name, parent)
            continue

        try:
            target_attr = get_dict_attr(target, attr_name)
        except KeyError:
            target_attr = {}

        log.debug('Merging %s from %s into %s of %s', attr_name, parent_attr, target_attr, target)

        merged_attr = merged(parent_attr, target_attr, **kw)
        log.debug('Merged %s -> %s', attr_name, merged_attr)
        with open_dictlike(target):
            set_dict_attr(target, attr_name, merged_attr)


def merged_into(target: T, parent: Any, attr_names: ListOrSingle[str], **kw) -> T:
    """merge parent[attr_name] into target[attr_name] and return a new target object"""
    target_copy = deepcopy(target)
    inplace_merge_into(target_copy, parent, attr_names, **kw)
    return target_copy


def merged_into_container(target: T, parent: Any, attr_names: ListOrSingle[str], **kw) -> T:
    # target is a container of objects to merge
    if isinstance(target, list):
        return [merged_into(obj, parent, attr_names, **kw) for obj in target]  # type: ignore
    if isinstance(target, dict):
        return {k: merged_into(v, parent, attr_names, **kw) for k, v in target.items()}  # type: ignore
    raise ValueError(f'Invalid target type {type(target)}')


def wrap_resolvable_constructor(obj: Resolvable[T], fn: Callable) -> Resolvable[T]:

    assert isinstance(obj, Resolvable)

    def wrapped_constructor(config: DictConfig, fn=fn, base_constructor=obj.constructor):
        return base_constructor(fn(config))

    return Resolvable(
        constructor=wrapped_constructor,
        config=obj.config,
        name=obj.name,
    )


def make_inheritable_on_resolve(
    target: Resolvable[T], parent: Any, attr_names: ListOrSingle[str]
) -> Resolvable[T]:
    # we wrap the target's constructor so that, just before resolving it,
    # we merge the inherited parent attributes into it
    return wrap_resolvable_constructor(
        target,
        partial(
            merged_into,
            parent=parent,
            attr_names=attr_names,
        ),
    )


class InheritableAttrsModel(cm.ArbitraryModel):

    _inherit: dict[str, list[str] | str] = {}

    def model_post_init(self, *_):
        for k, v in self._inherit.items():
            if isinstance(v, str):
                v = [v]
            setattr(self, k, merged_into(getattr(self, k), self, v))


##────────────────────────────────────────────────────────────────────────────}}}
