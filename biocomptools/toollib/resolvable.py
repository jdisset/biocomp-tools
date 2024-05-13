## {{{                         --     docstring     --
"""
A resolvable attribute keeps its DictLike representation until it is
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
from dataclasses import is_dataclass, fields
from copy import deepcopy, copy
from importlib import import_module
from functools import partial
from pydantic.functional_validators import BeforeValidator
from typing import (
    Annotated,
    Optional,
    Union,
    Dict,
    Type,
    Union,
    Any,
    Callable,
    Generic,
    TypeVar,
)
import logging
from rich.logging import RichHandler
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import BaseModel, Field, model_serializer

from biocomptools.toollib import common as cm
from biocomptools.toollib.common import DictLike, open_dictlike, ArbitraryModel

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
log = logging.getLogger('biocomptools.biocomplot.utils.resolvable')
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


def short_conf(conf: DictLike) -> str:
    if conf is None:
        return 'None'
    try:

        long = OmegaConf.to_yaml(conf, resolve=False)
        if isinstance(conf, DictConfig):
            long = '(DictConfig):' + long
        return long[:300] + ' [...] \n' if len(long) > 300 else long
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


def get_type(target_type: Union[str, Type[T]], default_target_module='') -> Type[T]:
    if isinstance(target_type, str):
        target_parts = target_type.split('.')
        target_module = '.'.join(target_parts[:-1])
        try:
            target_module = import_module(target_module)
            target_class = getattr(target_module, target_parts[-1])
        except:
            target_module = import_module(default_target_module)
            target_class = getattr(target_module, target_parts[-1])
        return target_class
    else:
        return target_type


def get_explicit_target_type(obj, default_module='biocomptools.toollib.plot'):
    # a _target_ key in the dict indicates the target type

    def isnull(x):
        return x is None or x == '' or x == 'None'

    target = None

    if '_target_' in obj and obj['_target_'] != '' and obj['_target_'] != 'None':
        target = obj['_target_']

    if isnull(target):
        return None

    # check if it's a type directly
    if isinstance(target, type):
        return target

    assert isinstance(target, str), f'Invalid target type {target}'

    return get_type(target, default_module)


def remove_target_key(obj: DictLike):
    return {k: v for k, v in obj.items() if k != '_target_'}


def target_instantiate(obj: DictLike, default_module='biocomptools.toollib.plot'):
    """simply create an instance of _target_ with the rest of the dict as kwargs"""
    target_class = get_explicit_target_type(obj, default_module)
    assert target_class is not None, f'No target class found in {obj}'
    kwargs = remove_target_key(obj)
    return target_class(**with_str_keys(kwargs))


def generic_instanciation_ctor(cfg: DictLike) -> Any:
    assert (
        get_explicit_target_type(cfg) is not None
    ), f'instanciation ctor called on non-target object {cfg}'
    return target_instantiate(cfg)


def remove_non_ctor_args(cls, kwargs):
    """remove any keys that are not in the class' constructor signature"""
    # WARN: this is a very bad implementation, but it's a start
    ctor_args = {}
    for k, v in kwargs.items():
        if k in get_public_attrs(cls):
            ctor_args[k] = v
    return ctor_args


def build_from_config(
    cls: Optional[Type[T]], cfg: DictLike, filter_out_non_ctor_args=False, ignore_target=False
) -> T:
    """
    Build an object of type cls from a DictLike cfg (by essentially calling cls(**cfg)).
    If the config has a _target_ key, build the target object instead.
    """

    if hasattr(cls, 'from_config'):
        return cls.from_config(cfg)  # type: ignore

    if get_explicit_target_type(cfg) is not None and not ignore_target:
        return target_instantiate(cfg)

    assert cls is not None, f'Invalid config {cfg=}, {cls=}'

    log.debug('Building %s from config:\n%s', cls.__name__, short_conf(cfg))

    ctor_args = with_str_keys(remove_target_key(cfg))

    if filter_out_non_ctor_args:
        ctor_args = remove_non_ctor_args(cls, ctor_args)

    return cls(**ctor_args)


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                        --     Resolvable     --


class Resolvable(ArbitraryModel, Generic[T]):
    """
    A Resolvable is a wrapper around an object that can be constructed at the last minute.
    Useful for things like variable interpolation that depends on the right context
    (this, plot_task, data, figure_task, etc.)
    When not resolved, it carries its DictLike representation from which it can be constrArbitrar
    """


    target_type: Type[T]
    config: DictLike = {}

    # debug info:
    name: Optional[str] = None
    clsname: Optional[str] = None

    dump_target: bool = True # add a _target_ key to the dumped config

    def model_post_init(self, *_):
        log.debug('Resolvable post-init: %s', self)

    def resolve(self):
        return build_from_config(self.target_type, self.config)

    def __repr__(self, indent=0):
        try:
            typename = self.target_type.__name__  # type: ignore
        except:
            typename = str(self.target_type)
        indentstr = ' ' * indent
        confstr = '\n ' + short_conf(self.config) if self.config is not None else 'Empty'
        indented_conf = indentstr + confstr.replace('\n', '\n ' + indentstr)
        name = f' "{self.name}" ' if self.name is not None else ''
        return indentstr+f'Resolvable{name}<{typename}>:{indented_conf}'

    # a Resolvable is dict-like (it transparently forwards dict-like operations to its config)

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

    # eq test:
    def __eq__(self, other):
        if not isinstance(other, Resolvable):
            return False
        return self.config == other.config and self.target_type == other.target_type

    @model_serializer()
    def model_dump(self):
        cfg = self.config
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=False)
        assert isinstance(cfg, dict), f'Invalid config {cfg=}'
        if self.dump_target:
            cfg = {'_target_': dump_type(self.target_type), **cfg}
        return cfg


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                     --     Resolvable utils     --

ResolvableOr = Union[Resolvable[T], T, DictLike]


def dump_type(t: Type[T]) -> str:
    return t.__module__ + '.' + t.__name__

def is_dictlike(obj):
    return isinstance(obj, DictLike) or is_dataclass(obj)


def make_resolvable(
    target_type: Optional[Type[T]] = None,
    value: Optional[Union[T, Resolvable, DictLike]] = None,
    name: Optional[str] = None,
    clsname: Optional[str] = None,
    force_resolvable=True,  # if True, will attempt to wrap any value in a Resolvable
    force_omegaconf=False,  # if True, will wrap any value in an OmegaConf DictConfig
) -> Resolvable[T]:
    """
    Return a Resolvable object from a value and a target type.
    """

    log.debug('Making resolvable %s with target_type=%s, value=%s', name, target_type, value)
    log.debug(f'{type(value)=}')

    original_target_type = target_type

    if value is None:
        value = {}


    if isinstance(value, Resolvable):
        target_type = value.target_type
        value = copy(value.config)
        log.debug('value is already a Resolvable, using target_type=%s', target_type)
    else:
        if not is_dictlike(value):
            log.debug('Value is not dict-like, trying to dump and wrap in Resolvable')
            if force_resolvable:
                try:
                    target_type = type(value)  # type: ignore
                    value = value.model_dump(by_alias=True)  # type: ignore
                    log.debug('Dumped value=%s, target_type=%s', value, target_type)
                except AttributeError:
                    raise ValueError(f'Can\'t dump {value=} with {target_type=}')
            else:
                raise ValueError(f'Invalid {value=} for {target_type=}')

    assert isinstance(value, DictLike), f'Invalid {value=} for {target_type}'

    if force_omegaconf:
        value = DictConfig(value)

    if not isinstance(target_type, type):
        raise ValueError(f'Invalid target type {target_type}')

    if original_target_type is None:
        original_target_type = target_type

    if not issubclass(target_type, original_target_type):
        raise ValueError(f'Invalid {target_type=} is unrelated to {original_target_type=}')

    assert is_dictlike(value), f'Invalid {value=} for {target_type}'

    return Resolvable[T](
        target_type=target_type,
        config=value,  # type: ignore
        name=name,
        clsname=clsname,
    )


def make_resolvable_validator(target_type, **kw) -> Callable:

    def validator(v: Any, info) -> Any:
        clsname = info.config['title']
        name = info.field_name
        return make_resolvable(target_type, value=v, name=name, clsname=clsname, **kw)

    return validator


from typing import TypeAlias

WRes: TypeAlias = Annotated[
    Union[
        Annotated[
            T,
            BeforeValidator(make_resolvable_validator(T)),
            Field(validate_default=True),
        ],
        T,
        Resolvable[T],
    ],
    Field(union_mode='left_to_right'),
]


class WrappedResolvable(Generic[T]):
    def __class_getitem__(cls, target_type):
        return Annotated[
            Resolvable[T],
            BeforeValidator(make_resolvable_validator(target_type)),
            Field(validate_default=True),
        ]


def resolvable(t, **kw):
    return (
        BeforeValidator(make_resolvable_validator(t, **kw)),
        Field(validate_default=True),
    )


def resolved(resolvable: Any, resolvers=None):
    """
    Resolve a Resolvable object with the given resolvers in context.
    i.e. returns A COPY of the object constructed by calling the constructor with the config
    using the provided resolvers.
    """

    if not isinstance(resolvable, Resolvable):
        return copy(resolvable)

    if resolvers is None:
        resolvers = {}

    log.debug('Resolving %s', resolvable)

    # with omegaconf_resolvers(resolvers):
    return resolvable.resolve()


##────────────────────────────────────────────────────────────────────────────}}}#

