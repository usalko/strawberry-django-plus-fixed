from enum import Enum
from typing import Any, Callable, Optional, Sequence, Type, TypeVar, cast

from django.db.models.base import Model
from django.db.models.sql.query import get_field_names_from_opts  # type:ignore
from strawberry import UNSET
from strawberry.field import StrawberryField
from strawberry.utils.typing import __dataclass_transform__
from strawberry_django import filters as _filters
from strawberry_django import utils
from strawberry_django.fields.field import field as _field

from django.db.models import QuerySet
from django.db.models import Q


from . import field
from .relay import GlobalID, connection, node
from .type import input

_T = TypeVar("_T")

_LOGICAL_EXPRESSIONS = {
    '_or': 'or',
    '_and': 'and',
    '_not': 'not',
}


def _normalize_value(value: Any):
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    elif isinstance(value, GlobalID):
        return value.node_id

    return value


def _build_filter_kwargs(filters):
    filter_kwargs = {}
    filter_methods = []
    django_model = cast(Type[Model], utils.get_django_model(filters))

    for f in utils.fields(filters):
        field_name = f.name
        field_value = _normalize_value(getattr(filters, field_name))

        # Unset means we are not filtering this. None is still acceptable
        if field_value is UNSET:
            continue

        # Logical expressions
        if field_name in _LOGICAL_EXPRESSIONS:
            print(field_name)
            continue

        if isinstance(field_value, Enum):
            field_value = field_value.value

        field_name = _filters.lookup_name_conversion_map.get(field_name, field_name)
        filter_method = getattr(filters, f"filter_{field_name}", None)
        if filter_method:
            filter_methods.append(filter_method)
            continue

        if django_model and field_name not in get_field_names_from_opts(django_model._meta):
            continue

        if utils.is_strawberry_type(field_value):
            subfield_filter_kwargs, subfield_filter_methods = _build_filter_kwargs(field_value)
            for subfield_name, subfield_value in subfield_filter_kwargs.items():
                if isinstance(subfield_value, Enum):
                    subfield_value = subfield_value.value
                filter_kwargs[f"{field_name}__{subfield_name}"] = subfield_value

            filter_methods.extend(subfield_filter_methods)
        else:
            filter_kwargs[field_name] = field_value

    return filter_kwargs, filter_methods


def _apply(filters, queryset: QuerySet, info=UNSET, pk=UNSET) -> QuerySet:
    if pk is not UNSET:
        queryset = queryset.filter(pk=pk)

    if (
        filters is UNSET
        or filters is None
        or not hasattr(filters, "_django_type")
        or not filters._django_type.is_filter
    ):
        return queryset

    filter_method = getattr(filters, "filter", None)
    if filter_method:
        return filter_method(queryset)

    filter_kwargs, filter_methods = _build_filter_kwargs(filters)
    queryset = queryset.filter(**filter_kwargs)
    for filter_method in filter_methods:
        if _filters.function_allow_passing_info(filter_method):
            queryset = filter_method(queryset=queryset, info=info)

        else:
            queryset = filter_method(queryset=queryset)

    return queryset


## Replace build_filter_kwargs by our implementation that can handle GlobalID
#_filters.build_filter_kwargs = _build_filter_kwargs

# Replace apply filter for the generate logical combination of the filters
_filters.apply = _apply


@__dataclass_transform__(
    order_default=True,
    field_descriptors=(
        StrawberryField,
        _field,
        node,
        connection,
        field.field,
        field.node,
        field.connection,
    ),
)
def filter(  # noqa:A001
    model: Type[Model],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    directives: Optional[Sequence[object]] = (),
    lookups: bool = False,
) -> Callable[[_T], _T]:
    return input(
        model,
        name=name,
        description=description,
        directives=directives,
        is_filter="lookups" if lookups else True,
        partial=True,
    )
