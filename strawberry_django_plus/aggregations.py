import enum
from typing import Callable, Optional, Sequence, Type, TypeVar, cast

import strawberry
from django.db.models.base import Model
from strawberry import UNSET
from strawberry.field import StrawberryField
from typing_extensions import dataclass_transform
from strawberry_django.fields.field import field as _field

from strawberry_django_plus.utils.typing import is_auto

from . import field
from .strawberry_django_field_aggregations import Aggregations

_T = TypeVar("_T")


# @__dataclass_transform__(
#     field_descriptors=(
#         StrawberryField,
#         _field,
#         node,
#         connection,
#         field.field,
#         field.node,
#         field.connection,
#     ),
# )
@dataclass_transform(kw_only_default=True)
def aggregations(
    model: Type[Model],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    directives: Optional[Sequence[object]] = (),
) -> Callable[[_T], _T]:
    def wrapper(cls):
        for fname, type_ in cls.__annotations__.items():
            if is_auto(type_):
                type_ = Aggregations  # noqa: PLW2901

            cls.__annotations__[fname] = Optional[type_]
            setattr(cls, fname, UNSET)

        return strawberry.input(
            cls,
            name=cast(str, name),
            description=cast(str, description),
            directives=directives,
        )

    return wrapper
