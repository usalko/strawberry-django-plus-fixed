from enum import Enum
from typing import List, Optional, Type

import strawberry
from django.db.models import Model, QuerySet
from django.db.models.aggregates import Count
from strawberry import UNSET
from strawberry.arguments import StrawberryArgument
from strawberry.auto import StrawberryAuto
from strawberry.types import Info
from strawberry_django.arguments import argument
from strawberry_django.utils import fields, is_django_type, unwrap_type
from strawberry_django_plus.utils.resolvers import _django_fields_from_info
from typing_extensions import dataclass_transform

from .group_concat import GroupConcat
from .remap import Remap


@strawberry.enum
class Aggregations(Enum):
    CONCAT = "concat"


def generate_aggregations_args(aggregations, prefix=""):
    args = []
    for field in fields(aggregations):
        _aggregations = getattr(aggregations, field.name, UNSET)
        if _aggregations is UNSET:
            continue
        if hasattr(_aggregations, '_kwargs_order') and _aggregations._kwargs_order:
            subargs = generate_aggregations_args(
                _aggregations, prefix=f"{prefix}{field.name}__")
            args.extend(subargs)
        else:
            args.append(f"{prefix}{field.name}")
    return args

@dataclass_transform(kw_only_default=True)
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
            # TODO: modify type for ability to array values (List instead Optional)
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
        values = [field.replace(self.name + '__', '')
                  for field in _django_fields_from_info(info)]
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

        def _dirty_remap(record: dict):
            # TODO: Answer for question: How to get type name for the selected field
            # Target function django_key -> object properties path
            cls1 = self.origin_django_type.origin
            cls1_instance = cls1()
            for field in cls1._type_definition.fields:
                if field.name in record:
                    setattr(cls1_instance, field.name, record[field.name])
                elif field.name == 'message':
                    cls2_td = info.schema.get_type_by_name('TgMessage')
                    cls2_instance = cls2_td.fields[0].origin()
                    setattr(cls2_instance, 'id', record[f'{field.name}__id'])
                    setattr(cls1_instance, field.name, cls2_instance)
                elif field.name == 'peer':
                    cls2_td = info.schema.get_type_by_name('TgInputPeer')
                    cls2_instance = cls2_td.fields[0].origin()

                    for sub_field in cls2_td.fields:
                        if sub_field.name == 'user':
                            cls3_td = info.schema.get_type_by_name('TgUser')
                            cls3_instance = cls3_td.fields[0].origin()
                            setattr(cls3_instance, 'first_name',
                                    record['peer__user__first_name'])

                            setattr(cls2_instance, 'user', cls3_instance)
                    setattr(cls1_instance, field.name, cls2_instance)
            return cls1_instance
        return Remap(_dirty_remap, result)

    def get_queryset(
        self, queryset: QuerySet, info: Info, aggregations: Type = UNSET, **kwargs
    ):
        queryset = super().get_queryset(queryset, info, **kwargs)
        return self.apply_aggregations(queryset, info, aggregations)
