"""
cache.py  --  a small thread-safe LRU + TTL cache for query results.
====================================================================

WHY CACHE
---------
Search traffic is heavily skewed: a small set of popular queries account for a
large share of requests, and the expensive stages (cross-encoder, LTR feature
build) are pure functions of (query, parameters). Caching the final ranked
result for a query means a repeated request is answered in microseconds instead
of re-running the transformer -- a large latency and CPU win.

DESIGN
------
* Keyed by a normalized tuple ``(method, query, top_k, extra...)`` so different
  parameterizations never collide.
* LRU eviction (``OrderedDict``): when full, the least-recently-used entry is
  dropped, bounding memory.
* TTL expiry: entries older than ``ttl_seconds`` are treated as misses, so a
  future re-index cannot serve stale rankings forever (set ttl=0 to disable).
* Thread-safe: a lock guards every mutation because uvicorn may serve requests
  from a threadpool.

This is intentionally lightweight (standard library only). In a multi-instance
deployment you would swap this for Redis behind the same tiny interface; the
service depends only on ``get`` / ``set`` / ``stats``.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable, Optional, Tuple


class QueryCache:
    """Bounded LRU cache with per-entry time-to-live."""

    def __init__(self, max_size: int = 512, ttl_seconds: int = 300,
                 enabled: bool = True) -> None:
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.enabled = enabled
        self._store: "OrderedDict[Hashable, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _expired(self, stored_at: float) -> bool:
        if self.ttl_seconds == 0:
            return False
        return (time.time() - stored_at) > self.ttl_seconds

    def get(self, key: Hashable) -> Optional[Any]:
        """Return the cached value for ``key`` or None (counts hit/miss)."""
        if not self.enabled:
            return None
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self._misses += 1
                return None
            stored_at, value = item
            if self._expired(stored_at):
                # Lazily evict the stale entry.
                del self._store[key]
                self._misses += 1
                return None
            # Mark as most-recently-used.
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: Hashable, value: Any) -> None:
        """Insert/replace a value, evicting the LRU entry if over capacity."""
        if not self.enabled:
            return
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.time(), value)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)  # drop least-recently-used

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Return hit/miss counters and current size (for /health & benchmarks)."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total else 0.0
            return {
                "enabled": self.enabled,
                "size": len(self._store),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            }
