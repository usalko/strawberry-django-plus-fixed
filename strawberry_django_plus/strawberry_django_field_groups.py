from typing import List, Optional, Type

import strawberry
from django.db.models import QuerySet
from strawberry import UNSET
from strawberry.arguments import StrawberryArgument
from strawberry.types import Info
from strawberry.utils.typing import __dataclass_transform__
from strawberry_django.arguments import argument
from strawberry_django.fields.field import field as _field
from strawberry_django.utils import fields

from strawberry_django_plus.utils.typing import is_auto

from . import field, utils
from .relay import connection, node


def generate_groups_args(groups, prefix=""):
    args = []
    for field in fields(groups):
        _groups = getattr(groups, field.name, UNSET)
        if _groups is UNSET:
            continue
        # if _groups == Groups.ASC:
        #     args.append(f"{prefix}{field.name}")
        # elif _groups == Groups.DESC:
        #     args.append(f"-{prefix}{field.name}")
        # else:
        #     subargs = generate_groups_args(_groups, prefix=f"{prefix}{field.name}__")
        #     args.extend(subargs)
    return args


def groups(model):
    def wrapper(cls):
        for name, type_ in cls.__annotations__.items():
            # if isinstance(type_, StrawberryAuto):
            #     type_ = Groups
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
    return queryset.groups_by(*args)


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
        type_ = utils.unwrap_type(self.type or self.child.type)

        if utils.is_django_type(type_):
            return type_._django_type.groups
        return None

    def apply_groups(self, queryset: QuerySet, groups) -> QuerySet:
        return apply(groups, queryset)

    def get_queryset(
        self, queryset: QuerySet, info: Info, groups: Type = UNSET, **kwargs
    ):
        queryset = super().get_queryset(queryset, info, **kwargs)
        return self.apply_groups(queryset, groups)


