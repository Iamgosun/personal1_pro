from __future__ import annotations

from typing import Any, Callable, Dict, Type


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, name: str | None = None):
        def deco(obj):
            key = name or obj.__name__
            if key in self._items:
                raise KeyError(f"{self.name} registry already has key: {key}")
            self._items[key] = obj
            return obj
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"{name} is not registered in {self.name}. Available: {list(self._items)}")
        return self._items[name]

    def keys(self):
        return list(self._items.keys())


METHOD_REGISTRY = Registry("method")
EXECUTOR_REGISTRY = Registry("executor")
