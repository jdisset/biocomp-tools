## {{{                    --     import and types     --
from pydantic import BaseModel, ValidationError, validator, Field, field_validator
from rich import print
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
    build_from_config,
    resolved,
    Resolvable,
    make_resolvable,
    ResolvableOr,
    make_resolvable_validator,
    resolvable,
)
from biocomptools.toollib import resolvable as br
from dataclasses import MISSING
from functools import partial
from dataclasses import dataclass

AnyConfig = Union[DictConfig, ListConfig]
DictLike = Union[Dict[str, Any], DictConfig]

T = TypeVar("T")
U = TypeVar("U")


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     classes     --
class Araw:
    def __init__(self, a: int):
        self.a = a

    def __eq__(self, other):
        return self.a == other.a


class A(BaseModel):
    a: int

    class Config:
        arbitrary_types_allowed = True

class Asub(A):
    asub: int


ResolvableA = Annotated[ResolvableOr[A], *resolvable(A)]


class B(BaseModel):
    a: ResolvableA
    b: int

    class Config:
        arbitrary_types_allowed = True

ResolvableB = Annotated[ResolvableOr[B], *resolvable(B)]


##────────────────────────────────────────────────────────────────────────────}}}

ra = br.make_resolvable(A, value={'a': 3})

assert isinstance(ra, Resolvable)
a = resolved(ra)
assert type(a) == A

rsa = br.make_resolvable(Asub, value={'a': 3, 'asub': 4})
assert isinstance(rsa, Resolvable)
sa = resolved(rsa)
assert type(sa) == Asub

rb = br.make_resolvable(B, {'b': 1, 'a': {'a': 42}})
b = resolved(rb)
assert isinstance(rb, Resolvable)
assert type(b) == B
assert isinstance(b.a, Resolvable)
ba = resolved(b.a)
assert isinstance(ba, A)

rb2 = make_resolvable(B, {'b': 12, 'a': ra})
assert isinstance(rb2, Resolvable)
assert isinstance(resolved(rb2).a, Resolvable)
rb2a = resolved(resolved(rb2).a)
assert resolved(resolved(rb2).a) == a

b2 = B(a=make_resolvable(A, {'a': 4}), b=3)
assert isinstance(b2, B)
assert isinstance(b2.a, Resolvable)
ba2 = resolved(b2.a)
assert isinstance(ba2, A)
assert ba2.a == 4

b3 = B(a=a, b=3)
assert isinstance(b3.a, Resolvable)  # type: ignore
assert resolved(b3) == b3
assert resolved(b3.a) == a

b4 = B(a=sa, b=3)
assert isinstance(b4.a, Resolvable)  # type: ignore
assert resolved(b4) == b4
assert resolved(b4.a) == sa

a = A(a=3)
ra = make_resolvable(value=a)
assert isinstance(ra, Resolvable)
rra = ra.resolve() # fails
issubclass(A, BaseModel)
assert ra.resolve() == a # fails


class C(BaseModel):
    b: ResolvableB
    class Config:
        arbitrary_types_allowed = True

c = C(b=rb)
assert isinstance(c.b, Resolvable)
assert resolved(c) == c
assert resolved(c.b) == b


# when dumping, since b.a is a resolvable, it dumps a dict that will then be interpreted as a config
# we need to see that it is the dump of a resolvable (thanks to the __resolvable__ key)!!

c2 = C(b=b)
assert isinstance(c2.b, Resolvable)
rc2 = resolved(c2)
rc2.b
assert rc2 == c2
rc2b = resolved(rc2.b)

assert rc2b == b
b = B(a=make_resolvable(A, {'a': 4}), b=3)
rb = make_resolvable(value=b)
assert resolved(rb) == b


from biocomp.utils import PartialFunction
import biocomptools.toollib.plot as pl


pcfg = pl.PlotConfig(rc_context={'hey': 120000})

pf = PartialFunction(func=lambda: print('hello world'))
pfr = make_resolvable(value=pf)
rpf = resolved(pfr)  # -> ok
rpf()


pt = pl.PlotTask(plot_method=pf) # pt is just a PlotTask. its plot_method is supposed to be a resolvable[PartialFunction]
assert isinstance(pt, pl.PlotTask)
assert isinstance(pt.plot_method, Resolvable)
print(pt.plot_method)

# when we resolve its plot_method, we should get a PartialFunction
rpm = resolved(pt.plot_method)
assert isinstance(rpm, PartialFunction) # indeed, we do
rpm()  # -> ok

# now we rewrap the PlotTask in a resolvable, which should dump it into a dict first and then wrap that dict in a config field of a dictionnary
dmp = pt.model_dump()
ptr = make_resolvable(value=pt)  # this adds some bullshit...
assert isinstance(ptr, Resolvable)
rptr = resolved(ptr)
assert isinstance(rptr, pl.PlotTask)
assert isinstance(rptr.plot_method, Resolvable)
assert rptr.plot_method == pt.plot_method
assert rptr.plot_config == pt.plot_config

resolved(rptr.plot_method)()


