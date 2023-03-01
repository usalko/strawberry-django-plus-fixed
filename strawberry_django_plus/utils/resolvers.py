import functools
import inspect
import warnings
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from asgiref.sync import async_to_sync, sync_to_async
from django.db.models import Model, QuerySet, Max, Q
from django.db.models.manager import BaseManager
from strawberry.types.fields.resolver import StrawberryResolver
from strawberry.types.info import Info
from strawberry.utils.await_maybe import AwaitableOrValue
from strawberry_django.utils import is_async
from typing_extensions import ParamSpec
from strawberry.types.nodes import SelectedField
from strawberry.schema.schema import Schema
from re import sub

from strawberry_django_plus import relay
from strawberry_django_plus.relay import GlobalID, Node

from .aio import is_awaitable, resolve_async
from .inspect import get_django_type

_T = TypeVar("_T")
_M = TypeVar("_M", bound=Model)
_R = TypeVar("_R")
_P = ParamSpec("_P")
_sentinel = object()
_async_to_sync = cast(
    Callable[[Callable[[Awaitable[_T]], Coroutine[Any, Any, _T]]], Callable[[Awaitable[_T]], _T]],
    async_to_sync,
)


@overload
def async_safe(
    func: Callable[_P, _R],
    /,
    *,
    thread_sensitive: bool = True,
) -> Callable[_P, AwaitableOrValue[_R]]:
    ...


@overload
def async_safe(
    func: None,
    /,
    *,
    thread_sensitive: bool = True,
) -> Callable[[Callable[_P, _R]], Callable[_P, AwaitableOrValue[_R]]]:
    ...


def async_safe(func=None, /, *, thread_sensitive=True):
    """Decorate a function to be async safe, ensuring it is called in a sync context always.

    - If `f` is a coroutine function, this is a noop.
    - When running, if an asyncio loop is running, the function will
      be called wrapped in an asgi.sync_to_async_ context.

    Args:
        func:
            The function to call
        thread_sensitive:
            If the sync function should run in the same thread as all other
            thread_sensitive functions

    Returns
    -------
        The wrapped function

    .. _asgi.sync_to_async:
        https://docs.djangoproject.com/en/dev/topics/async/#asgiref.sync.sync_to_async

    """

    def make_resolver(f):
        if (
            inspect.iscoroutinefunction(f)
            or inspect.isasyncgenfunction(f)
            or (isinstance(f, StrawberryResolver) and f.is_async)
        ):
            return f

        async_resolver = sync_to_async(f, thread_sensitive=thread_sensitive)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            resolver = async_resolver if is_async() else f

            return resolver(*args, **kwargs)

        return wrapper

    if func is not None:
        return make_resolver(func)

    return make_resolver


def async_unsafe(*args, **kwargs):
    warnings.warn("use `async_safe` instead", DeprecationWarning)
    return async_safe(*args, **kwargs)


@_async_to_sync
async def resolve_sync(value: Awaitable[_T]) -> _T:
    """Resolve the given value, resolving any returned awaitable.

    Args:
        value:
            The awaitable to be resolved

    Returns
    -------
        The resolved value.

    """
    return await value


getattr_async_safe = async_safe(lambda obj, key, *args: getattr(obj, key, *args))
getattr_str_async_safe = async_safe(lambda obj, key, *args: str(getattr(obj, key, *args)))


@overload
def resolve_qs(
    qs: AwaitableOrValue[Union[BaseManager[_M], QuerySet[_M]]],
    *,
    resolver: None = ...,
    info: Optional[Info] = ...,
) -> AwaitableOrValue[QuerySet[_M]]:
    ...


@overload
def resolve_qs(
    qs: AwaitableOrValue[Union[BaseManager[_M], QuerySet[_M]]],
    *,
    resolver: Optional[Callable[[QuerySet[_M]], AwaitableOrValue[_R]]],
    info: Optional[Info] = ...,
) -> AwaitableOrValue[_R]:
    ...


def resolve_qs(qs, *, resolver=None, info=None) -> Any:  # type: ignore
    """Resolve the queryset, ensuring its db operations are executed in a sync context.

    Args:
        qs:
            The function to call
        resolver:
            An optional function that, when present, will be called in a thread
            sensitive context to resolve the queryset.
            thread_sensitive functions
        info:
            Optional gql execution info. If present, will use its implementation
            of `is_awaitable`, which might have some optimizations. Otherwise
            will fallback to `inspect.is_awaitable`

    Returns
    -------
        The wrapped function

    .. _asgi.sync_to_async:
        https://docs.djangoproject.com/en/dev/topics/async/#asgiref.sync.sync_to_async

    """
    if isinstance(qs, BaseManager):
        qs = qs.all()

    if is_awaitable(qs, info=info):
        return resolve_async(
            qs,
            functools.partial(resolve_qs, resolver=resolver, info=info),
            info=info,
        )

    if resolver is None:
        # This is what QuerySet does internally to fetch results.
        # After this, iterating over the queryset should be async safe
        def resolver(r):
            return r._fetch_all() or r

    if is_async() and not (
        inspect.iscoroutinefunction(resolver) or inspect.isasyncgenfunction(resolver)
    ):
        resolver = sync_to_async(resolver, thread_sensitive=True)

    return resolver(qs)


resolve_qs_get_list = cast(
    Callable[[AwaitableOrValue[QuerySet[_M]]], AwaitableOrValue[List[_M]]],
    functools.partial(resolve_qs, resolver=list),
)
resolve_qs_get_first = cast(
    Callable[[AwaitableOrValue[QuerySet[_M]]], AwaitableOrValue[Optional[_M]]],
    functools.partial(resolve_qs, resolver=lambda qs: qs.first()),
)
resolve_qs_get_one = cast(
    Callable[[AwaitableOrValue[QuerySet[_M]]], AwaitableOrValue[_M]],
    functools.partial(resolve_qs, resolver=lambda qs: qs.get()),
)


@overload
def resolve_result(
    res: AwaitableOrValue[Union[BaseManager[_M], QuerySet[_M]]],
    *,
    qs_resolver: None = ...,
    info: Optional[Info] = ...,
) -> AwaitableOrValue[QuerySet[_M]]:
    ...


@overload
def resolve_result(
    res: AwaitableOrValue[Union[BaseManager[_M], QuerySet[_M]]],
    *,
    info: Optional[Info] = ...,
    qs_resolver: Callable[[Union[BaseManager[_M], QuerySet[_M]]], AwaitableOrValue[_T]],
) -> AwaitableOrValue[_T]:
    ...


@overload
def resolve_result(
    res: AwaitableOrValue[Callable[[], _T]],
    *,
    qs_resolver: None = ...,
    info: Optional[Info] = ...,
) -> AwaitableOrValue[_T]:
    ...


@overload
def resolve_result(
    res: AwaitableOrValue[_T],
    *,
    info: Optional[Info] = ...,
    qs_resolver: Callable[[Union[BaseManager[Any], QuerySet[Any]]], AwaitableOrValue[Any]],
) -> AwaitableOrValue[_T]:
    ...


def resolve_result(res, *, info=None, qs_resolver=None):
    """Resolve the result, ensuring any qs and callables are resolved in a sync context.

    Args:
        res:
            The result to resolve
        info:
            Optional gql execution info. Make sure to always provide this or
            otherwise, the queryset cannot be optimized in case DjangoOptimizerExtension
            is enabled. This will also be used for `is_awaitable` check.
        qs_resolver:
            Optional qs_resolver to use to resolve any queryset. If not provided,
            `resolve_qs` will be used, which by default returns the queryset
            already prefetched from the database.

    Returns
    -------
        The resolved result.

    """
    from strawberry_django_plus import optimizer  # avoid circular import

    if isinstance(res, (BaseManager, QuerySet)):
        if isinstance(res, BaseManager):
            res = cast(QuerySet, res.all())

        if info is not None:
            ext = optimizer.optimizer.get()
            if ext is not None:
                # If optimizer extension is enabled, optimize this queryset
                res = ext.optimize(qs=res, info=info)

        qs_resolver = qs_resolver or resolve_qs
        return qs_resolver(res)

    if callable(res):
        return resolve_result(async_safe(res)(), info=info, qs_resolver=qs_resolver)

    if is_awaitable(res, info=info):
        return resolve_async(
            res,
            functools.partial(resolve_result, info=info, qs_resolver=qs_resolver),
            info=info,
        )

    return res


def camelBack_to_underscores(text: str):
    return sub(r'[A-Z]', lambda x: '_' + x.group(0).lower(), text)


def _is_valid_django_model_attributes_path(django_model: Model, parent_attribute: str, attribute: str) -> bool:
    if parent_attribute and hasattr(django_model, parent_attribute):
        model_for_attribute = getattr(django_model, parent_attribute)
        if hasattr(model_for_attribute, 'field') and model_for_attribute.field.many_to_many:
            return hasattr(model_for_attribute.rel.identity[1], attribute)
        elif hasattr(model_for_attribute, 'field') and model_for_attribute.field.is_relation and hasattr(model_for_attribute.field, 'type'):
            return hasattr(model_for_attribute.field.type, attribute)
        elif hasattr(model_for_attribute, 'field') and model_for_attribute.field.is_relation and hasattr(model_for_attribute.field, 'related_model'):
            return hasattr(model_for_attribute.field.related_model, attribute)
        return True
    return hasattr(django_model, attribute)


def _make_filter_from_selected_fields(schema: Schema, selected_fields: List[SelectedField], model: Model, parent: str = ''):
    '''
        There is the simple 'group by' model for distinct simulation (it uses max function for id).
        The design:
            Get all selected fields (if selected fields contain id, this function never called).
            For each selected field add additional filter (id in max(id) for every field value)
    '''
    #self.model._default_manager.values('lexeme__name').annotate(Max('id')).values('id__max') = None
    results = []
    for selected_field in filter(lambda selected_field: not selected_field.name.startswith('__'), selected_fields):
        django_field_name = f'{parent}__{camelBack_to_underscores(selected_field.name)}' if parent else camelBack_to_underscores(selected_field.name)
        if selected_field.selections and hasattr(model, django_field_name) and getattr(model, django_field_name).field.is_relation:
            results.append(_make_filter_from_selected_fields(schema, selected_field.selections, model, django_field_name))
        elif _is_valid_django_model_attributes_path(model, parent, selected_field.name):
            results.append(Q(id__in=model._default_manager.values(django_field_name).annotate(Max('id')).values('id__max')))
    if len(results) == 0:
        return [] # All fields are calculated in graphql schema
    if len(results) == 1:
        return results[0]
    result = results[0]
    for i in range(1, len(results)):
        result = result | results[i]
    return result


def _simple_django_fields_mapper(selections: List[SelectedField], prefix: str, result: List[str]) -> None:
    for selected_field in filter(lambda selected_field: not selected_field.name.startswith('__'), selections):
        django_field_name = f'{prefix}__{camelBack_to_underscores(selected_field.name)}' if prefix else camelBack_to_underscores(
            selected_field.name)
        if selected_field.selections:
            _simple_django_fields_mapper(
                selected_field.selections, django_field_name, result)
        else:
            result.append(django_field_name)


def _django_fields_from_info(info: Info, prefix: str = '') -> List[str]:
    '''
        There is the simple 'group by' model for distinct simulation (it uses max function for id).
        The design:
            Get all selected fields (if selected fields contain id, this function never called).
            For each selected field add additional filter (id in max(id) for every field value)
    '''
    result = []
    _simple_django_fields_mapper(info.selected_fields, '', result)
    return result


def _has_ability_to_apply_distinct(selected_fields: List[SelectedField]):
    return selected_fields and not any(any(field.name == 'id' for field in selected_field.selections) for selected_field in selected_fields)


def _get_selections(selected_field: SelectedField, *path) -> List[SelectedField]:
    if not path:
        return selected_field.selections
    if len(path) == 1:
        for selected_child_field in selected_field.selections:
            if selected_child_field.name == path[0]:
                return selected_child_field.selections
        return []
    for selected_child_field in selected_field.selections:
        if selected_child_field.name == path[0]:
            return _get_selections(selected_child_field, *path[1:])
    return []


def resolve_model_nodes(
    source: Union[Type[Node], Type[_M]],
    *,
    info: Optional[Info] = None,
    node_ids: Optional[Iterable[Union[str, GlobalID]]] = None,
    filter_perms: bool = False,
) -> AwaitableOrValue[QuerySet[_M]]:
    """Resolve model nodes, ensuring those are prefetched in a sync context.

    Args:
        source:
            The source model or the model type that implements the `Node` interface
        info:
            Optional gql execution info. Make sure to always provide this or
            otherwise, the queryset cannot be optimized in case DjangoOptimizerExtension
            is enabled. This will also be used for `is_awaitable` check.
        node_ids:
            Optional filter by those node_ids instead of retrieving everything

    Returns
    -------
        The resolved queryset, already prefetched from the database

    """
    # avoid circular import
    from strawberry_django_plus import optimizer
    from strawberry_django_plus.permissions import filter_with_perms

    if issubclass(source, Model):
        origin = None
    else:
        origin = source
        django_type = get_django_type(source, ensure_type=True)
        source = cast(Type[_M], django_type.model)
        
    # TODO: make another way to the distinct implementation 
    # # DISTINCT by default
    # # It's implemented as additional filter induced from results
    # distinguished_ids_filter = []
    # if _has_ability_to_apply_distinct(_get_selections(info.selected_fields[0], 'edges')):
    #     distinguished_ids_filter = _make_filter_from_selected_fields(info.schema, _get_selections(info.selected_fields[0], 'edges', 'node'), source)
    # if distinguished_ids_filter:
    #     qs = source._default_manager.filter(distinguished_ids_filter)
    # else:
    #     qs = source._default_manager.all().distinct()
    qs = source._default_manager.all()

    if origin and hasattr(origin, "get_queryset"):
        qs = origin.get_queryset(qs, info)  # type: ignore

    if node_ids is not None:
        id_attr = getattr(origin, "id_attr", "pk")
        qs = qs.filter(
            **{f"{id_attr}__in": [i.node_id if isinstance(i, GlobalID) else i for i in node_ids]},
        )

    if filter_perms:
        assert info
        qs = filter_with_perms(qs, info)

    qs_resolver = resolve_qs
    if info is not None:
        ext = optimizer.optimizer.get()
        if ext is not None:
            # If optimizer extension is enabled, optimize this queryset
            qs = ext.optimize(qs, info=info)
        # Connection will filter the results when its is being resolved. We don't want to
        # fetch everything before it does that
        if isinstance(return_type := info.return_type, type) and issubclass(
            return_type,
            relay.Connection,
        ):

            def qs_resolver(qs):
                return qs

    return resolve_result(qs, info=info, qs_resolver=qs_resolver)


@overload
def resolve_model_node(
    source: Union[Type[Node], Type[_M]],
    node_id: Union[str, GlobalID],
    *,
    info: Optional[Info] = ...,
    required: Literal[False] = ...,
) -> AwaitableOrValue[Optional[_M]]:
    ...


@overload
def resolve_model_node(
    source: Union[Type[Node], Type[_M]],
    node_id: Union[str, GlobalID],
    *,
    info: Optional[Info] = ...,
    required: Literal[True],
) -> AwaitableOrValue[_M]:
    ...


def resolve_model_node(source, node_id, *, info: Optional[Info] = None, required=False):
    """Resolve model nodes, ensuring it is retrieved in a sync context.

    Args:
        source:
            The source model or the model type that implements the `Node` interface
        node_id:
            The node it to retrieve the model from
        info:
            Optional gql execution info. Make sure to always provide this or
            otherwise, the queryset cannot be optimized in case DjangoOptimizerExtension
            is enabled. This will also be used for `is_awaitable` check.
        required:
            If the return value is required to exist. If true, `qs.get()` will be
            used, which might raise `model.DoesNotExist` error if the node doesn't exist.
            Otherwise, `qs.first()` will be used, which might return None.

    Returns
    -------
        The resolved node, already prefetched from the database

    """
    if issubclass(source, Model):
        origin = None
    else:
        origin = source
        django_type = get_django_type(source, ensure_type=True)
        source = cast(Type[_M], django_type.model)

    if isinstance(node_id, GlobalID):
        node_id = node_id.node_id

    id_attr = getattr(origin, "id_attr", "pk")

    qs = source._default_manager.filter(**{id_attr: node_id})

    if origin and hasattr(origin, "get_queryset"):
        qs = origin.get_queryset(qs, info)

    if required:
        ret = resolve_result(qs, info=info, qs_resolver=resolve_qs_get_one)
    else:
        ret = resolve_result(qs, info=info, qs_resolver=resolve_qs_get_first)

    return ret  # noqa: RET504


def resolve_model_id(source: Union[Type[Node], Type[_M]], root: Model) -> AwaitableOrValue[str]:
    """Resolve the model id, ensuring it is retrieved in a sync context.

    Args:
        source:
            The source model or the model type that implements the `Node` interface
        root:
            The source model object.

    Returns
    -------
        The resolved object id

    """
    id_attr = getattr(source, "id_attr", "pk")
    assert isinstance(root, Model)
    if id_attr == "pk":
        pk = root.__class__._meta.pk
        assert pk
        id_attr = pk.attname
    assert id_attr
    try:
        # Prefer to retrieve this from the cache
        return str(root.__dict__[id_attr])
    except KeyError:
        return getattr_str_async_safe(root, id_attr)
