## {{{                     --     import and types     --
from pydantic import BaseModel, ValidationError, Field, field_validator
from pydantic.functional_validators import AfterValidator, BeforeValidator

from omegaconf import DictConfig, ListConfig, OmegaConf

from typing import (
    Annotated,
    Union,
    Dict,
    Any,
    TypeVar,
)

import biocomp.utils as ut
from biocomp.utils import ArbitraryModel
from biocomptools.toollib.resolvable import (
    resolved,
    Resolvable,
    ResolvableOr,
    resolvable,
    make_resolvable,
    get_explicit_target_type,
    get_type,
)
from biocomptools.toollib import resolvable as br
from biocomptools.toollib import inheritable as bi
from biocomptools.toollib.inheritable import merged_into, InheritableAttrsModel, as_dict
from biocomptools.toollib.common import dict_like

AnyConfig = Union[DictConfig, ListConfig]
DictLike = Union[Dict[str, Any], DictConfig]

T = TypeVar("T")
U = TypeVar("U")

##────────────────────────────────────────────────────────────────────────────}}}


"""
The inheritable module makes possible for an attribute in a member objeect to inherit from an attribute in the parent object.
For example: B.d should be merged into B.a.d when B.a is constructed.
"""


class A(ArbitraryModel):
    d: dict[str, Any]


class B(InheritableAttrsModel):

    a: Annotated[ResolvableOr[A], *resolvable(A)]
    d: dict[str, Any]

    _inherit = {'a': 'd'}


b = B(a=A.model_validate({'d': {'x': 1}}), d={'y': 2})

assert isinstance(b.a, Resolvable)
assert b.d == {'y': 2}
assert resolved(b.a).d == {'x': 1, 'y': 2}

b2 = B(a={'d': {'x': 1}}, d={'y': 2})
assert isinstance(b2.a, br.Resolvable), type(b2.a)
assert b2.a['d'] == {'x': 1, 'y': 2}
assert resolved(b2.a).d == {'x': 1, 'y': 2}

# now we have an issue when merging the SingleFigure declared in job into the more generic FigureMaker
# in datasource. Because it will merge the content, but not the type, so we should make it possible
# to either have a FigureMaker that's None and gets completely overwritten by anything
# or to specify that a more specialized type can overwrite a more general one

# what we have:
##


class A(ArbitraryModel):
    x: int = 0

    def hello(self):
        print(f'hello base, {self.x=}')


class Asub(A):

    def hello(self):
        print(f'hello sub, {self.x=}')


class B(ArbitraryModel):
    a: Annotated[ResolvableOr[A], *resolvable(A)] = {}


class C(ArbitraryModel):
    a: Annotated[ResolvableOr[A], *resolvable(A)]
    b: Annotated[ResolvableOr[B], *resolvable(B)]

    def model_post_init(self, *_):
        self.b = merged_into(self.b, self, 'a')



rasub = make_resolvable(value=Asub(x=10))
rb = make_resolvable(value=B())
rb_from_type = make_resolvable(B)

b = B()
c = C(a=Asub(x=20), b=b)

rb['a'] # maybe we could only add _target_ if
rb_from_type['a'] = rasub


##

c2 = C(a=Asub(x=20), b=rb)

rba = resolved(b.a)
rba.hello()

rcb = resolved(c.b)
rcba = resolved(rcb.a)
assert type(rcba) == Asub

c2 = C(a=Asub(x=20), b=rb)
assert type(resolved(resolved(c2.b).a)) == Asub

c3 = C(a=Asub(x=20), b=rb_from_type)
assert type(resolved(resolved(c3.b).a)) == Asub

rcba.hello()
