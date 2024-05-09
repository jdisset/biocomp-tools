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

from biocomptools.toollib.common import ArbitraryModel
from biocomptools.toollib.resolvable import (
    resolve,
    ResolvableOr,
    resolvable,
)
from biocomptools.toollib import resolvable as br
from biocomptools.toollib import inheritable as bi
from biocomptools.toollib.inheritable import merged_into, InheritableAttrsModel

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

assert isinstance(b.a, A)
assert b.d == {'y': 2}
assert b.a.d == {'x': 1, 'y': 2}

b2 = B(a={'d': {'x': 1}}, d={'y': 2})
assert isinstance(b2.a, br.Resolvable), type(b2.a)
assert b2.a['d'] == {'x': 1, 'y': 2}
assert resolve(b2.a).d == {'x': 1, 'y': 2}
