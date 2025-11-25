from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from core.exceptions import CacheError

T = TypeVar("T")
AsyncFn = Callable[..., Awaitable[T]]


@dataclass
class CacheEntry:
    value: Any
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() >= self.expires_at


class InMemoryTTLCache:
    """Simple in-memory TTL cache with optional default expiration."""

    def __init__(self, default_ttl: Optional[int] = None) -> None:
        self._default_ttl = default_ttl
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _compute_expiry(ttl_seconds: Optional[int]) -> Optional[float]:
        if ttl_seconds is None:
            return None
        return time.time() + ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if entry.is_expired():
                self._store.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        expires_at = self._compute_expiry(ttl_seconds or self._default_ttl)
        with self._lock:
            self._store[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def async_ttl_cache(cache: InMemoryTTLCache, ttl_seconds: Optional[int] = None) -> Callable[[AsyncFn], AsyncFn]:
    """Decorator that caches the result of an async function using the provided cache."""

    def decorator(func: AsyncFn) -> AsyncFn:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = _build_cache_key(func.__qualname__, args, kwargs)
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            result = await func(*args, **kwargs)

            try:
                cache.set(cache_key, result, ttl_seconds=ttl_seconds)
            except Exception as cache_exc:
                raise CacheError(f"Failed to store value in cache: {cache_exc}") from cache_exc

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _build_cache_key(namespace: str, args: Any, kwargs: Any) -> str:
    """Builds a deterministic cache key based on function arguments."""

    def normalize(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, dict):
            return tuple(sorted((normalize(k), normalize(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(normalize(v) for v in value)
        return repr(value)

    normalized_args = normalize(args)
    normalized_kwargs = normalize(kwargs)
    return f"{namespace}:{normalized_args}:{normalized_kwargs}"
