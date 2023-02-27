from enum import Enum
from typing import List, Optional, Type

import strawberry
from django.db.models import QuerySet
from django.db.models.aggregates import Count
from strawberry import UNSET
from strawberry.arguments import StrawberryArgument
from strawberry.auto import StrawberryAuto
from strawberry.types import Info
from strawberry.utils.typing import __dataclass_transform__
from strawberry_django.arguments import argument
from strawberry_django.utils import fields, is_django_type, unwrap_type

from .group_concat import GroupConcat


@strawberry.enum
class Groups(Enum):
    ARRAY = "array"


def generate_groups_args(groups, prefix=""):
    args = []
    for field in fields(groups):
        _groups = getattr(groups, field.name, UNSET)
        if _groups is UNSET:
            continue
        if _groups:
            args.append(f"{prefix}{field.name}")
        else:
            subargs = generate_groups_args(_groups, prefix=f"{prefix}{field.name}__")
            args.extend(subargs)
    return args


def groups(model):
    def wrapper(cls):
        for name, type_ in cls.__annotations__.items():
            if isinstance(type_, StrawberryAuto):
                type_ = Groups
            cls.__annotations__[name] = Optional[type_]
            setattr(cls, name, UNSET)
        return strawberry.input(cls)

    return wrapper


def apply(groups, queryset: QuerySet) -> QuerySet:
    if groups is UNSET or groups is None:
        return queryset
    args = generate_groups_args(groups)
    if not args:
        return queryset
    '''
    LogModel.objects.values('level', 'info').annotate(
    count=Count(1), time=GroupConcat('time', ordering='time DESC', separator=' | ')
).order_by('-time', '-count')
    '''
    result = queryset
    # TODO: implement algo:
    #  - For all selected fields (map to the values)
    #  - For all aggregated fields (aka groups) do GroupConcat for map to array
    # for arg in args:
    # result = result.values('emoticon', 'message__id')
    # result = result.annotate(count=Count(1),
    #                          peer__user__first_name=GroupConcat('peer__user__first_name'))
    return result


class StrawberryDjangoFieldGroups:

    def __init__(self, groups=UNSET, **kwargs):
        self.groups = groups
        super().__init__(**kwargs)

    @property
    def arguments(self) -> List[StrawberryArgument]:
        arguments = []
        if not self.base_resolver:
            groups = self.get_groups()
            if groups and groups is not UNSET and self.is_list:
                arguments.append(argument("groups", groups))
        return super().arguments + arguments

    def get_groups(self) -> Optional[Type]:
        if self.groups is not UNSET:
            return self.groups
        type_ = unwrap_type(self.type or self.child.type)

        if is_django_type(type_):
            return type_._django_type.groups
        return None

    def apply_groups(self, queryset: QuerySet, groups) -> QuerySet:
        return apply(groups, queryset)

    def get_queryset(
        self, queryset: QuerySet, info: Info, groups: Type = UNSET, **kwargs
    ):
        queryset = super().get_queryset(queryset, info, **kwargs)
        return self.apply_groups(queryset, groups)


