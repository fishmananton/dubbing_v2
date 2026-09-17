"""Bounded thread-pool fan-out for independent, I/O- or subprocess-bound work.

Used where the pipeline runs many independent external calls (ffmpeg per segment,
pyloudnorm per line) that today execute serially. Threads are the right tool: the work
is dominated by subprocess/native calls that release the GIL, and results must line up
with inputs, so this returns an order-preserving list.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def default_workers() -> int:
    return min(32, (os.cpu_count() or 4))


def parallel_map(fn: Callable[[T], R], items: Iterable[T],
                 max_workers: int | None = None) -> list[R]:
    """Apply `fn` to each item concurrently, returning results in INPUT order.

    An exception in any call propagates (first one raised wins). `max_workers=1` runs
    serially. Bounded by `default_workers()` when unset."""
    items = list(items)
    if not items:
        return []
    workers = max(1, min(max_workers or default_workers(), len(items)))
    if workers == 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))
