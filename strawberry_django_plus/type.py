import copy
from enum import Enum
import dataclasses
import inspect
import sys
import types
from functools import cached_property
from typing import (
    Callable,
    ForwardRef,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import strawberry
from django.core.exceptions import FieldDoesNotExist
from django.db.models.base import Model
from django.db.models.fields.reverse_related import ManyToManyRel, ManyToOneRel
from strawberry import UNSET, relay
from strawberry.annotation import StrawberryAnnotation
from strawberry.exceptions import MissingFieldAnnotationError
from strawberry.field import StrawberryField
from strawberry.private import is_private
from strawberry.type import get_object_definition
from strawberry.unset import UnsetType
from strawberry_django.fields.field import field as _field
from strawberry_django.fields.types import get_model_field, resolve_model_field_name
from strawberry_django.type import StrawberryDjangoType as _StraberryDjangoType
from strawberry_django.utils import get_annotations
from typing_extensions import dataclass_transform

from strawberry_django_plus.optimizer import OptimizerStore, PrefetchType
from strawberry_django_plus.utils.typing import TypeOrSequence, is_auto

from . import field
from .descriptors import ModelProperty
from .field import StrawberryDjangoField, connection, node
from .relay import Connection, ConnectionField, Node
from .types import resolve_model_field_type
from .utils.resolvers import (
    resolve_model_id,
    resolve_model_id_attr,
    resolve_model_node,
    resolve_model_nodes,
)


try:
    from django.contrib.contenttypes.fields import GenericForeignKey, GenericRel

    GenericTypes = (GenericForeignKey, GenericRel)
except (ImportError, RuntimeError):  # pragma:nocover
    GenericTypes = ()

class JointType(Enum):

    AND = '_and'
    OR = '_or'
    NOT = '_not'

    @classmethod
    def choices(cls):
        return tuple((i, i.value) for i in cls)

    @classmethod
    def logical_expressions(cls):
        return {value: key for key, value in cls.choices()}


_LOGICAL_EXPRESSIONS = JointType.logical_expressions()

__all = [
    "StrawberryDjangoType",
    "type",
    "interface",
    "input",
    "partial",
]

_T = TypeVar("_T")
_O = TypeVar("_O", bound=type)
_M = TypeVar("_M", bound=Model)


def _from_django_type(
    django_type: "StrawberryDjangoType",
    name: str,
    *,
    type_annotation: Optional[StrawberryAnnotation] = None,
) -> StrawberryDjangoField:
    origin = django_type.origin

    attr = getattr(origin, name, dataclasses.MISSING)
    if attr is UNSET or attr is dataclasses.MISSING:
        attr = getattr(StrawberryDjangoField, "__dataclass_fields__", {}).get(name, UNSET)

    if type_annotation:
        try:
            type_origin = get_origin(type_annotation.annotation)
            is_connection = issubclass(type_origin, Connection) if type_origin else False
        except Exception:  # noqa: BLE001
            is_connection = False
        if is_private(type_annotation.annotation):
            raise PrivateStrawberryFieldError(name, django_type.origin)
    else:
        is_connection = False

    if is_connection or isinstance(attr, ConnectionField):
        field = attr
        if not isinstance(field, ConnectionField):
            field = connection()

        field = cast(StrawberryDjangoField, field)

        # FIXME: Improve this...
        if not field.base_resolver:

            def conn_resolver(root):
                return getattr(root, name).all()

            field.base_resolver = StrawberryResolver(conn_resolver)
            if type_annotation is not None:
                field.type_annotation = type_annotation
    elif isinstance(attr, StrawberryDjangoField) and not attr.origin_django_type:
        field = attr
    elif isinstance(attr, dataclasses.Field):
        default = getattr(attr, "default", dataclasses.MISSING)
        default_factory = getattr(attr, "default_factory", dataclasses.MISSING)

        if type_annotation is None:
            type_annotation = getattr(attr, "type_annotation", None)
        if type_annotation is None:
            type_annotation = StrawberryAnnotation(attr.type)

        store = getattr(attr, "store", None)
        field = StrawberryDjangoField(
            django_name=getattr(attr, "django_name", None) or attr.name,
            graphql_name=getattr(attr, "graphql_name", None),
            origin=getattr(attr, "origin", None),
            is_subscription=getattr(attr, "is_subscription", False),
            description=getattr(attr, "description", None),
            base_resolver=getattr(attr, "base_resolver", None),
            permission_classes=getattr(attr, "permission_classes", ()),
            default=default,
            default_factory=default_factory,
            deprecation_reason=getattr(attr, "deprecation_reason", None),
            directives=getattr(attr, "directives", ()),
            type_annotation=type_annotation,
            filters=getattr(attr, "filters", UNSET),
            order=getattr(attr, "order", UNSET),
            aggregations=getattr(attr, "aggregations", UNSET),
            only=store and store.only,
            select_related=store and store.select_related,
            prefetch_related=store and store.prefetch_related,
            disable_optimization=getattr(attr, "disable_optimization", False),
            extensions=getattr(attr, "extensions", ()),
        )
    elif isinstance(attr, StrawberryResolver):
        field = StrawberryDjangoField(base_resolver=attr)
    elif callable(attr):
        field = cast(StrawberryDjangoField, StrawberryDjangoField()(attr))
    else:
        field = StrawberryDjangoField(default=attr)

    field.python_name = name
    # store origin django type for further usage
    field.origin_django_type = django_type

    # annotation of field is used as a class type
    if type_annotation is not None:
        field.type_annotation = type_annotation
        field.is_auto = is_auto(field.type_annotation)

    # resolve the django_name and check if it is relation field. django_name
    # is used to access the field data in resolvers
    try:
        model_field = get_model_field(
            django_type.model,
            getattr(field, "django_name", None) or name,
        )
    except FieldDoesNotExist:
        model_attr = getattr(django_type.model, name, None)
        if model_attr is not None and isinstance(model_attr, ModelProperty):
            if field.is_auto:
                annotation = model_attr.type_annotation
                if get_origin(annotation) is Annotated:
                    annotation = get_args(annotation)[0]
                field.type_annotation = StrawberryAnnotation(annotation)
                field.is_auto = is_auto(field.type_annotation)

            if field.description is None:
                field.description = model_attr.description
        elif field.django_name or field.is_auto:
            raise  # field should exist, reraise caught exception
    else:
        field.is_relation = model_field.is_relation
        if not field.django_name:
            field.django_name = resolve_model_field_name(
                model_field,
                is_input=django_type.is_input,
                is_filter=bool(django_type.is_filter),
            )

        # change relation field type to auto if field is inherited from another
        # type. for example if field is inherited from output type but we are
        # configuring field for input type
        if field.is_relation and not is_similar_django_type(django_type, field.origin_django_type):
            field.is_auto = True

        # resolve type of auto field
        if field.is_auto:
            field.type_annotation = StrawberryAnnotation(
                resolve_model_field_type(model_field, django_type),
            )

        if field.description is None:
            if isinstance(model_field, (GenericRel, GenericForeignKey)):
                description = None
            elif isinstance(model_field, (ManyToOneRel, ManyToManyRel)):
                description = model_field.field.help_text
            else:
                description = getattr(model_field, "help_text")  # noqa: B009

            if description:
                field.description = str(description)

    return field


def _get_fields(django_type: "StrawberryDjangoType"):
    origin = django_type.origin
    fields = {}
    seen_fields = set()

    # collect all annotated fields
    for name, annotation in get_annotations(origin).items():
        with suppress(PrivateStrawberryFieldError):
            fields[name] = _from_django_type(
                django_type,
                name,
                type_annotation=annotation,
            )
        seen_fields.add(name)

    # collect non-annotated strawberry fields
    for name in dir(origin):
        if name in seen_fields:
            continue

        attr = getattr(origin, name, None)
        if not isinstance(attr, StrawberryField):
            continue

        fields[name] = _from_django_type(django_type, name)

    return fields


def _has_own_node_resolver(cls, name: str) -> bool:
    resolver = getattr(cls, name, None)
    if resolver is None:
        return False

    if id(resolver.__func__) == id(getattr(Node, name).__func__):
        return False

    return True


def _process_type(
    cls: _O,
    model: Type[Model],
    *,
    field_cls: Type[StrawberryDjangoField] = StrawberryDjangoField,
    filters: Optional[type] = UNSET,
    order: Optional[type] = UNSET,
    aggregations: Optional[type] = UNSET,
    pagination: Optional[bool] = UNSET,
    only: Optional[TypeOrSequence[str]] = None,
    select_related: Optional[TypeOrSequence[str]] = None,
    prefetch_related: Optional[TypeOrSequence[PrefetchType]] = None,
    disable_optimization: bool = False,
    partial: bool = False,
    is_filter: Union[Literal["lookups"], bool] = False,
    **kwargs,
) -> _O:
    is_input = kwargs.get("is_input", False)
    original_annotations = cls.__dict__.get("__annotations__", {})

    is_filter = kwargs.pop("is_filter", False)
    if is_filter:
        cls.__annotations__ = {**cls.__annotations__, ** {
            JointType.AND.value: Optional[ForwardRef(cls.__name__)],
            JointType.OR.value: Optional[ForwardRef(cls.__name__)],
            JointType.NOT.value: Optional[ForwardRef(cls.__name__)],
        }}

    django_type = StrawberryDjangoType(
        origin=cls,
        model=model,
        field_cls=field_cls,
        is_partial=partial,
        is_input=is_input,
        is_filter=is_filter,
        filters=filters,
        order=order,
        aggregations=aggregations,
        pagination=pagination,
        disable_optimization=disable_optimization,
        store=OptimizerStore.with_hints(
            only=only,
            select_related=select_related,
            prefetch_related=prefetch_related,
        ),
    )

    auto_fields: set[str] = set()
    for field_name, field_annotation in get_annotations(cls).items():
        annotation = field_annotation.annotation
        if is_private(annotation):
            continue

        if is_auto(annotation):
            auto_fields.add(field_name)

        # FIXME: For input types it is imported to set the default value to UNSET
        # Is there a better way of doing this?
        if is_input:
            # First check if the field is defined in the class. If it is,
            # then we just need to set its default value to UNSET in case
            # it is MISSING
            if field_name in cls.__dict__:
                field = cls.__dict__[field_name]
                if isinstance(field, dataclasses.Field) and field.default is dataclasses.MISSING:
                    field.default = UNSET
                    if isinstance(field, StrawberryField):
                        field.default_value = UNSET

                continue

            if not hasattr(cls, field_name):
                base_field = getattr(cls, "__dataclass_fields__", {}).get(field_name)
                if base_field is not None and isinstance(base_field, StrawberryField):
                    new_field = copy.copy(base_field)
                    for attr in [
                        "_arguments",
                        "permission_classes",
                        "directives",
                        "extensions",
                    ]:
                        old_attr = getattr(base_field, attr)
                        if old_attr is not None:
                            setattr(new_field, attr, old_attr[:])
                else:
                    new_field = _field(default=UNSET)

                new_field.type_annotation = field_annotation
                new_field.default = UNSET
                if isinstance(base_field, StrawberryField):
                    new_field.default_value = UNSET
                setattr(cls, field_name, new_field)

    # Make sure model is also considered a "virtual subclass" of cls
    if "is_type_of" not in cls.__dict__:
        cls.is_type_of = lambda obj, info: isinstance(obj, (cls, model))  # type: ignore

    # Default querying methods for relay
    if issubclass(cls, relay.Node):
        for attr, func in [
            ("resolve_id", resolve_model_id),
            ("resolve_id_attr", resolve_model_id_attr),
            ("resolve_node", resolve_model_node),
            ("resolve_nodes", resolve_model_nodes),
        ]:
            existing_resolver = getattr(cls, attr, None)
            if (
                existing_resolver is None
                or existing_resolver.__func__ is getattr(relay.Node, attr).__func__
            ):
                setattr(cls, attr, types.MethodType(func, cls))

            # Adjust types that inherit from other types/interfaces that implement Node
            # to make sure they pass themselves as the node type
            meth = getattr(cls, attr)
            if isinstance(meth, types.MethodType) and meth.__self__ is not cls:
                setattr(cls, attr, types.MethodType(cast(classmethod, meth).__func__, cls))

    strawberry.type(cls, **kwargs)

    # update annotations and fields
    type_def = get_object_definition(cls, strict=True)
    new_fields: List[StrawberryField] = []
    for f in type_def.fields:
        django_name: Optional[str] = getattr(f, "django_name", None) or f.python_name or f.name
        description: Optional[str] = getattr(f, "description", None)
        type_annotation: Optional[StrawberryAnnotation] = getattr(
            f,
            "type_annotation",
            None,
        )

        if f.name in auto_fields:
            f_is_auto = True
            # Force the field to be auto again for it to be re-evaluated
            if type_annotation:
                type_annotation.annotation = strawberry.auto
        else:
            f_is_auto = type_annotation is not None and is_auto(
                type_annotation.annotation,
            )

        try:
            if django_name is None:
                raise FieldDoesNotExist  # noqa: TRY301
            model_attr = get_model_field(django_type.model, django_name)
        except FieldDoesNotExist as e:
            model_attr = getattr(django_type.model, django_name, None)
            is_relation = False

            if model_attr is not None and isinstance(model_attr, ModelProperty):
                if type_annotation is None or f_is_auto:
                    type_annotation = StrawberryAnnotation(
                        model_attr.type_annotation,
                        namespace=sys.modules[model_attr.func.__module__].__dict__,
                    )

                if description is None:
                    description = model_attr.description
            elif model_attr is not None and isinstance(model_attr, (property, cached_property)):
                func = model_attr.fget if isinstance(model_attr, property) else model_attr.func

                if type_annotation is None or f_is_auto:
                    if (return_type := func.__annotations__.get("return")) is None:
                        raise MissingFieldAnnotationError(django_name, type_def.origin) from e

                    type_annotation = StrawberryAnnotation(
                        return_type,
                        namespace=sys.modules[func.__module__].__dict__,
                    )

                if description is None and func.__doc__:
                    description = inspect.cleandoc(func.__doc__)
        else:
            is_relation = model_attr.is_relation
            if not django_name:
                django_name = resolve_model_field_name(
                    model_attr,
                    is_input=django_type.is_input,
                    is_filter=bool(django_type.is_filter),
                )

            if description is None:
                if isinstance(model_attr, GenericTypes):
                    f_description = None
                elif isinstance(model_attr, (ManyToOneRel, ManyToManyRel)):
                    f_description = model_attr.field.help_text
                else:
                    f_description = getattr(model_attr, "help_text", None)

                if f_description:
                    description = str(f_description)

        if isinstance(f, StrawberryDjangoField) and not f.origin_django_type:
            # If the field is a StrawberryDjangoField, just update its annotations/description/etc
            f.type_annotation = type_annotation
            f.description = description
        elif (
            not isinstance(f, StrawberryDjangoField)
            and getattr(f, "base_resolver", None) is not None
        ):
            # If this is not a StrawberryDjangoField, but has a base_resolver, no need
            # avoid forcing it to be a StrawberryDjangoField
            new_fields.append(f)
            continue
        else:
            store = getattr(f, "store", None)
            f = StrawberryDjangoField(  # noqa: PLW2901
                django_name=django_name,
                description=description,
                type_annotation=type_annotation,
                python_name=f.python_name,
                graphql_name=getattr(f, "graphql_name", None),
                origin=getattr(f, "origin", None),
                is_subscription=getattr(f, "is_subscription", False),
                base_resolver=getattr(f, "base_resolver", None),
                permission_classes=getattr(f, "permission_classes", ()),
                default=getattr(f, "default", dataclasses.MISSING),
                default_factory=getattr(f, "default_factory", dataclasses.MISSING),
                deprecation_reason=getattr(f, "deprecation_reason", None),
                directives=getattr(f, "directives", ()),
                filters=getattr(f, "filters", UNSET),
                order=getattr(f, "order", UNSET),
                only=store and store.only,
                select_related=store and store.select_related,
                prefetch_related=store and store.prefetch_related,
                disable_optimization=getattr(f, "disable_optimization", False),
                extensions=getattr(f, "extensions", ()),
            )

        f.django_name = django_name
        f.is_relation = is_relation
        f.origin_django_type = django_type  # type: ignore

        new_fields.append(f)
        if f.base_resolver and f.python_name:
            setattr(cls, f.python_name, f)

    type_def = get_object_definition(cls, strict=True)
    type_def._fields = new_fields
    cls._django_type = django_type  # type: ignore

    return cls


@dataclasses.dataclass
class StrawberryDjangoType(_StraberryDjangoType[_O, _M]):
    """Strawberry django type metadata."""

    is_filter: Union[Literal["lookups"], bool]
    order: Optional[Union[type, UnsetType]]
    filters: Optional[Union[type, UnsetType]]
    aggregations: Optional[Union[type, UnsetType]]
    pagination: Optional[Union[bool, UnsetType]]
    disable_optimization: bool
    store: OptimizerStore


@dataclass_transform(
    order_default=True,
    field_specifiers=(
        StrawberryField,
        _field,
        node,
        connection,
        field.field,
        field.node,
        field.connection,
    ),
)
def type(  # noqa: A001
    model: Type[Model],
    *,
    name: Optional[str] = None,
    field_cls: Type[StrawberryDjangoField] = StrawberryDjangoField,
    is_input: bool = False,
    is_interface: bool = False,
    is_filter: Union[Literal["lookups"], bool] = False,
    description: Optional[str] = None,
    directives: Optional[Sequence[object]] = (),
    extend: bool = False,
    filters: Optional[type] = UNSET,
    pagination: Optional[bool] = UNSET,
    order: Optional[type] = UNSET,
    aggregations: Optional[type] = UNSET,
    only: Optional[TypeOrSequence[str]] = None,
    select_related: Optional[TypeOrSequence[str]] = None,
    prefetch_related: Optional[TypeOrSequence[PrefetchType]] = None,
    disable_optimization: bool = False,
) -> Callable[[_T], _T]:
    """Annotates a class as a Django GraphQL type.

    Examples:
        It can be used like this:

        >>> @gql.django.type(SomeModel)
        ... class X:
        ...     some_field: gql.auto
        ...     otherfield: str = gql.django.field()

    """

    def wrapper(cls):
        return _process_type(
            cls,
            model,
            name=name,
            field_cls=field_cls,
            is_input=is_input,
            is_filter=is_filter,
            is_interface=is_interface,
            description=description,
            directives=directives,
            extend=extend,
            filters=filters,
            pagination=pagination,
            order=order,
            aggregations=aggregations,
            only=only,
            select_related=select_related,
            prefetch_related=prefetch_related,
            disable_optimization=disable_optimization,
        )

    return wrapper


@dataclass_transform(
    order_default=True,
    field_specifiers=(
        StrawberryField,
        _field,
        node,
        connection,
        field.field,
        field.node,
        field.connection,
    ),
)
def interface(
    model: Type[Model],
    *,
    name: Optional[str] = None,
    field_cls: Type[StrawberryDjangoField] = StrawberryDjangoField,
    description: Optional[str] = None,
    directives: Optional[Sequence[object]] = (),
) -> Callable[[_T], _T]:
    """Annotates a class as a Django GraphQL interface.

    Examples:
        It can be used like this:

        >>> @gql.django.interface(SomeModel)
        ... class X:
        ...     some_field: gql.auto
        ...     otherfield: str = gql.django.field()

    """

    def wrapper(cls):
        return _process_type(
            cls,
            model,
            name=name,
            field_cls=field_cls,
            is_interface=True,
            description=description,
            directives=directives,
        )

    return wrapper


@dataclass_transform(
    order_default=True,
    field_specifiers=(
        StrawberryField,
        _field,
        node,
        connection,
        field.field,
        field.node,
        field.connection,
    ),
)
def input(  # noqa: A001
    model: Type[Model],
    *,
    name: Optional[str] = None,
    field_cls: Type[StrawberryDjangoField] = StrawberryDjangoField,
    description: Optional[str] = None,
    directives: Optional[Sequence[object]] = (),
    is_filter: Union[Literal["lookups"], bool] = False,
    partial: bool = False,
) -> Callable[[_T], _T]:
    """Annotates a class as a Django GraphQL input.

    Examples:
        It can be used like this:

        >>> @gql.django.input(SomeModel)
        ... class X:
        ...     some_field: gql.auto
        ...     otherfield: str = gql.django.field()

    """

    def wrapper(cls):
        return _process_type(
            cls,
            model,
            name=name,
            field_cls=field_cls,
            is_input=True,
            is_filter=is_filter,
            description=description,
            directives=directives,
            partial=partial,
        )

    return wrapper


@dataclass_transform(
    order_default=True,
    field_specifiers=(
        StrawberryField,
        _field,
        node,
        connection,
        field.field,
        field.node,
        field.connection,
    ),
)
def partial(
    model: Type[Model],
    *,
    name: Optional[str] = None,
    field_cls: Type[StrawberryDjangoField] = StrawberryDjangoField,
    description: Optional[str] = None,
    directives: Optional[Sequence[object]] = (),
) -> Callable[[_T], _T]:
    """Annotates a class as a Django GraphQL partial.

    Examples:
        It can be used like this:

        >>> @gql.django.partial(SomeModel)
        ... class X:
        ...     some_field: gql.auto
        ...     otherfield: str = gql.django.field()

    """

    def wrapper(cls):
        return _process_type(
            cls,
            model,
            name=name,
            field_cls=field_cls,
            is_input=True,
            description=description,
            directives=directives,
            partial=True,
        )

    return wrapper
