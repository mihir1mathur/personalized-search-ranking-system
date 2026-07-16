"""
timing.py  --  tiny, dependency-free timing helpers.
====================================================

Latency is a first-class metric for a search API (users feel every extra
millisecond), so we make measuring it trivial and consistent everywhere.

``Timer`` is a context manager that records elapsed wall-clock time in
milliseconds using ``time.perf_counter`` (the correct clock for durations)::

    with Timer() as t:
        do_work()
    print(t.ms)   # -> 12.34

We deliberately use perf_counter, not time.time, because perf_counter is
monotonic and high-resolution, so it never goes backwards if the system clock
is adjusted mid-request.
"""

from __future__ import annotations

import time


def now_ms() -> float:
    """High-resolution monotonic timestamp in milliseconds (for durations)."""
    return time.perf_counter() * 1000.0


class Timer:
    """Context manager that measures elapsed time in milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0
        self.ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._end = time.perf_counter()
        self.ms = (self._end - self._start) * 1000.0
        return False  # never suppress exceptions

    @property
    def elapsed_ms(self) -> float:
        """Elapsed ms so far (works before the context has exited too)."""
        end = self._end if self._end else time.perf_counter()
        return (end - self._start) * 1000.0
