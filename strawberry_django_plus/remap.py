from typing import Any, Callable, Dict

from django.db.models.query import QuerySet


class Remap:

    def __init__(self, _map: Callable[[Dict], Any], query_set: QuerySet) -> None:
        super().__init__()
        self.query_set = query_set
        self._map = _map

    def _fetch_all(self):
        self.query_set._fetch_all()

    def __iter__(self):
        return map(self._map, self.query_set.__iter__())
