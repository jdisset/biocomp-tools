from pydantic import BaseModel, ValidationError, validator, Field, field_validator
from typing import (
    Annotated,
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    Sequence,
    Any,
    Callable,
    TypeVar,
    Generic,
    Iterable,
    TypeAlias,
    get_args,
)
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic.functional_validators import AfterValidator, BeforeValidator

# from biocomptools.toollib.plot import open_dictlike, short_conf
from biocomptools.toollib.resolvable import open_dictlike, short_conf
from biocomptools.toollib.resolvable import (
    wrapped_resolvable,
    build_from_config,
    resolve,
    Resolvable,
    make_resolvable,
)
from biocomptools.toollib import resolvable as br
from dataclasses import MISSING
from functools import partial

AnyConfig = Union[DictConfig, ListConfig]
DictLike = Union[Dict[str, Any], DictConfig]


class A(BaseModel):
    a: int

    class Config:
        arbitrary_types_allowed = True


class B(BaseModel):
    a: wrapped_resolvable(A) # type: ignore
    b: int

    class Config:
        arbitrary_types_allowed = True


ra = br.make_resolvable(A, value={'a': 3})
assert isinstance(ra, Resolvable)
a = resolve(ra)
assert type(a) == A

rb = br.make_resolvable(B, {'b': 1, 'a': {'a': 42}})
b = resolve(rb)
assert isinstance(rb, Resolvable)
assert type(b) == B
assert isinstance(b.a, Resolvable)
ba = resolve(b.a)
assert isinstance(ba, A)

rb2 = make_resolvable(B, {'b': 12, 'a': ra})
assert isinstance(rb2, Resolvable)
assert isinstance(resolve(rb2).a, Resolvable)
assert resolve(resolve(rb2).a) == a


b2 = B(a=make_resolvable(A, {'a': 4}), b=3)
assert isinstance(b2, B)
assert isinstance(b2.a, Resolvable)
ba2 = resolve(b2.a)
assert isinstance(ba2, A)
assert ba2.a == 4

b3 = B(a=a, b=3)
assert isinstance(b3.a, A)
assert b3.a == a
