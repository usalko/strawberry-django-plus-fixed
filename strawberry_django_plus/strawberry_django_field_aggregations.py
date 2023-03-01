from enum import Enum
from typing import List, Optional, Type

import strawberry
from django.db.models import QuerySet
from django.db.models.aggregates import Count
from django.db.models import Model
from strawberry import UNSET
from strawberry.arguments import StrawberryArgument
from strawberry.auto import StrawberryAuto
from strawberry.types import Info
from strawberry.utils.typing import __dataclass_transform__
from strawberry_django.arguments import argument
from strawberry_django.utils import fields, is_django_type, unwrap_type

from strawberry_django_plus.utils.resolvers import _django_fields_from_info

from .group_concat import GroupConcat


@strawberry.enum
class Aggregations(Enum):
    ARRAY = "array"


def generate_aggregations_args(aggregations, prefix=""):
    args = []
    for field in fields(aggregations):
        _aggregations = getattr(aggregations, field.name, UNSET)
        if _aggregations is UNSET:
            continue
        if hasattr(_aggregations, '_kwargs_order') and _aggregations._kwargs_order:
            subargs = generate_aggregations_args(_aggregations, prefix=f"{prefix}{field.name}__")
            args.extend(subargs)
        else:
            args.append(f"{prefix}{field.name}")
    return args


def aggregations(model):
    def wrapper(cls):
        for name, type_ in cls.__annotations__.items():
            if isinstance(type_, StrawberryAuto):
                type_ = Aggregations
            cls.__annotations__[name] = Optional[type_]
            setattr(cls, name, UNSET)
        return strawberry.input(cls)

    return wrapper


class StrawberryDjangoFieldAggregations:

    def __init__(self, aggregations=UNSET, **kwargs):
        self.aggregations = aggregations
        super().__init__(**kwargs)

    @property
    def arguments(self) -> List[StrawberryArgument]:
        arguments = []
        if not self.base_resolver:
            aggregations = self.get_aggregations()
            if aggregations and aggregations is not UNSET and self.is_list:
                arguments.append(argument("aggregations", aggregations))
        return super().arguments + arguments

    def get_aggregations(self) -> Optional[Type]:
        if self.aggregations is not UNSET:
            return self.aggregations
        type_ = unwrap_type(self.type or self.child.type)

        if is_django_type(type_):
            return type_._django_type.aggregations
        return None

    def apply_aggregations(self, queryset: QuerySet, info: Info, aggregations: Type) -> QuerySet:
        if aggregations is UNSET or aggregations is None:
            return queryset
        args = generate_aggregations_args(aggregations)
        if not args:
            return queryset
        '''
            LogModel.objects.values('level', 'info').annotate(
            count=Count(1), time=GroupConcat('time', ordering='time DESC', separator=' | ')
            ).order_by('-time', '-count')
        '''
        result = queryset
        # TODO: put the model into _django_fields_from_info function
        values = [field.replace(self.name + '__', '') for field in _django_fields_from_info(info)]
        # Algo:
        #  - For all selected fields (map to the values)
        #  - For all aggregated fields (aka aggregations) do GroupConcat for map to array
        # for arg in args:
        
        # Exclude group from values
        values = [x for x in values if x not in args]
        result = result.values(*values)
        groups = {a: GroupConcat(a) for a in args}
        if groups:
            result = result.annotate(count=Count(1), **groups)
        return result

    def get_queryset(
        self, queryset: QuerySet, info: Info, aggregations: Type = UNSET, **kwargs
    ):
        queryset = super().get_queryset(queryset, info, **kwargs)
        return self.apply_aggregations(queryset, info, aggregations)


