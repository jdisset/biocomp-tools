## {{{                         --     docstring     --
"""
A resolvable attribute keeps its DictConfig representation until it is
resolved, at which point it calls the constructor with the resolved config.
Therefore a resolved attribute needs to store 2 things:
- the unresolved config (i.e. with ${OmegaConf:interpolation variable} type
  strings in it)
- the constructor to build the object from the config

Why resolvable attributes?
--------------------------

The declarative task system relies on modular, nested configurations.
In order to customize figures and plots for example, users will often need to access
the context of the plot task, the figure task, the data, etc. (e.g. to set
the title of a plot, ...). The desired piece of context is oftentime not
available until instanciation of many things down the pipeline.

For example, if you need to know the index of the figure task in the list
of all figure tasks to set the title, you can use the ${figure_task:index}
interpolation variable. Or if you want to access some metadata information
that will only be added after the data is loaded, you can use ${this:
metadata.some_key}. But you can't resolve any of that at the time the
configuration is parsed, because the figure task doesn't exist yet.

"""
##────────────────────────────────────────────────────────────────────────────}}}
## {{{                          --     imports     --
from dataclasses import is_dataclass, fields, MISSING
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
from functools import partial
from pydantic.functional_validators import AfterValidator, BeforeValidator
from typing import (
    Annotated,
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    Type,
    Union,
    Any,
    Callable,
    Generic,
    TypeVar,
    get_args,
)
import logging
from rich.logging import RichHandler
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pydantic import BaseModel, ValidationError, Field, field_validator, model_validator
from biocomptools.toollib.common import DictLike, DictOrList, AnyConfig, Pair, ListOrSingle
from biocomptools.toollib.common import dict_like
from biocomptools.toollib import common as cm

##────────────────────────────────────────────────────────────────────────────}}}
## {{{                           --     types     --
T = TypeVar('T')
U = TypeVar('U')


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                      --     omegaconf utils     --
@contextmanager
def open_dictlike(obj: Any):
    if isinstance(obj, DictConfig):
        with open_dict(obj):
            yield
    else:
        yield


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                       --     logger utils --
LOGFORMAT = "in %(funcName)s: %(message)s"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level="NOTSET", format=LOGFORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger('biocomptools.biocomplot.utils.resolvable')
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


def short_conf(conf: DictLike) -> str:
    if conf is None:
        return 'None'
    try:
        long = OmegaConf.to_yaml(conf, resolve=False)
        return long[:100] + ' [...] \n' if len(long) > 100 else long
    except Exception as e:
        return str(conf)


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                     --     instantiate utils     --


def with_str_keys(d: DictLike) -> Dict[str, Any]:
    return {str(k): v for k, v in d.items()}


def get_public_attrs(obj):
    if is_dataclass(obj):
        return [f.name for f in fields(obj)]
    # if it's a type, try to instantiate it
    if isinstance(obj, type):
        obj = obj()
    return [a for a in dir(obj) if not a.startswith('__')]


def has_target(obj):
    return '_target_' in obj


def get_target_class(obj: DictLike, default_module='biocomptools.toollib.plot') -> Type:
    """get the class of the target object"""

    assert '_target_' in obj, f'Invalid data source object: {obj=}'

    # target should be a string module.path.ClassName
    # let's check that the class exists and it is a subclass of DataSource
    target = obj['_target_']
    target_parts = target.split('.')
    target_module = '.'.join(target_parts[:-1])
    target_class = target_parts[-1]

    if target_module == '':
        target_module = import_module(default_module)
    else:
        target_module = import_module(target_module)

    target_class = getattr(target_module, target_class)
    return target_class


def target_instantiate(obj: DictLike, default_module='biocomptools.toollib.plot'):
    """simply create an instance of _target_ with the rest of the dict as kwargs"""
    target_class = get_target_class(obj, default_module)
    kwargs = {k: v for k, v in obj.items() if k != '_target_'}
    return target_class(**with_str_keys(kwargs))


def generic_instanciation_ctor(cfg: DictLike) -> Any:
    assert has_target(cfg), f'instanciation ctor called on non-target object {cfg}'
    return target_instantiate(cfg)


def remove_non_ctor_args(cls, kwargs):
    """remove any keys that are not in the class' constructor signature"""
    # WARN: this is a very bad implementation, but it's a start
    ctor_args = {}
    for k, v in kwargs.items():
        if k in get_public_attrs(cls):
            ctor_args[k] = v
    return ctor_args


def build_from_config(cls: Optional[Type[T]], cfg: DictLike, filter_out_non_ctor_args=False) -> T:
    """
    Build an object of type cls from a DictLike cfg (by essentially calling cls(**cfg)).
    If the config has a _target_ key, build the target object instead.
    """

    # check if it has a _target_ key
    if has_target(cfg):
        subclass = get_target_class(cfg)
        print(f'{type(subclass)=}')
        # if (cls is not None) and (not issubclass(subclass, cls)):
        # raise ValueError(f'Invalid target class {subclass}. Expected base class {cls}')
        return target_instantiate(cfg)

    assert cls is not None, f'Invalid config {cfg=}, {cls=}'

    log.debug('Building %s from config:\n%s', cls.__name__, short_conf(cfg))

    ctor_args = with_str_keys(cfg)

    if filter_out_non_ctor_args:
        ctor_args = remove_non_ctor_args(cls, ctor_args)

    return cls(**ctor_args)


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                        --     Resolvable     --


class Resolvable(BaseModel, Generic[T]):
    """
    A Resolvable object is a wrapper around an object that can be constructed at the last minute.
    Useful for things like variable interpolation that depends on the right context
    (this, plot_task, data, figure_task, etc.)
    When not resolved, it carries its DictLike representation
    and a constructor to build the object from it.

    """

    constructor: Callable[..., T]
    config: Optional[DictLike] = None

    # debug info:
    name: Optional[str] = None
    clsname: Optional[str] = None
    typename: Optional[str] = None

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        confstr = '\n ' + short_conf(self.config) if self.config is not None else 'Empty'
        indented_conf = indentstr + confstr.replace('\n', '\n ' + indentstr)
        typename = self.typename if self.typename is not None else 'UnknownType'
        name = f' "{self.name}" ' if self.name is not None else 'Unknownname'
        return f'Resolvable{name}->[{typename}]:{indented_conf}'

    # a Resolvable is dict_like (it transparently forwards dict-like operations to its config)

    def __getitem__(self, key):
        if self.config is None:
            raise KeyError(f'No config found for {self}')
        with open_dictlike(self.config):
            return self.config[key]

    def __setitem__(self, key, value):
        if self.config is None:
            self.config = OmegaConf.create({})
        with open_dictlike(self.config):
            self.config[key] = value

    def __contains__(self, key):
        return self.config is not None and key in self.config

    def __iter__(self):
        if self.config is None:
            return iter([])
        return iter(self.config)

    def keys(self):
        if self.config is None:
            return []
        return self.config.keys()

    def values(self):
        if self.config is None:
            return []
        return self.config.values()

    def items(self):
        if self.config is None:
            return []
        return self.config.items()

    def get(self, key, default=None):
        if self.config is None:
            return default
        return self.config.get(key, default)

    def resolve(self):
        if self.config is None:
            return self.constructor()
        return self.constructor(self.config)

    class Config:
        arbitrary_types_allowed = True


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                     --     Resolvable utils     --


def make_resolvable(
    target_type: Type[T],
    value: Optional[Union[Resolvable, DictLike]] = None,
    name: Optional[str] = None,
    clsname: Optional[str] = None,
) -> Resolvable[T]:
    """
    Return a Resolvable object from a value and a target type.
    If the value is a Resolvable, check that it has the right target type and return it.
    If the value is a DictLike, wrap it in a Resolvable object with the target type.
    """

    try:
        typename = target_type.__name__  # type: ignore
    except:
        typename = str(target_type)

    if value is None:
        value = {}

    if isinstance(value, Resolvable):
        value = deepcopy(value.config)

    constructor = partial(build_from_config, target_type)

    return Resolvable[T](
        constructor=constructor, config=value, typename=typename, name=name, clsname=clsname
    )


def make_resolvable_validator(target_type) -> Callable:

    def validator(v: Any, info) -> Any:
        clsname = info.config['title']
        name = info.field_name
        if isinstance(v, target_type):
            return v
        return make_resolvable(target_type, value=v, name=name, clsname=clsname)

    return validator


ResolvableOr = Union[Resolvable[T], T, AnyConfig]


def wrapped_resolvable(t):
    return Annotated[
        ResolvableOr[t],
        BeforeValidator(make_resolvable_validator(t)),
        Field(validate_default=True),
    ]


def resolve(resolvable: Any, resolvers=None):
    """
    Resolve a Resolvable object with the given resolvers in context.
    i.e. returns the object constructed by calling the constructor with the config
    using the provided resolvers.
    """

    if not isinstance(resolvable, Resolvable):
        return resolvable  # not a resolvable, return as is

    resolvers = resolvers or {}

    log.debug('Resolving %s', resolvable)

    # with omegaconf_resolvers(resolvers):
    return resolvable.resolve()


##────────────────────────────────────────────────────────────────────────────}}}#
