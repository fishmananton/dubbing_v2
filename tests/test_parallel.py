import time

import pytest

from parallel import parallel_map


def test_preserves_input_order():
    # Results must line up with inputs even though work finishes out of order.
    def work(x):
        time.sleep(0.02 if x == 0 else 0.0)  # item 0 finishes last
        return x * 10
    assert parallel_map(work, [0, 1, 2, 3]) == [0, 10, 20, 30]


def test_empty_input_returns_empty():
    assert parallel_map(lambda x: x, []) == []


def test_runs_concurrently():
    # 8 items each sleeping 50ms must finish well under the 400ms serial total.
    def slow(_):
        time.sleep(0.05)
        return 1
    t0 = time.time()
    out = parallel_map(slow, list(range(8)), max_workers=8)
    assert out == [1] * 8
    assert time.time() - t0 < 0.25  # concurrent, not 0.4s serial


def test_exception_propagates():
    def boom(x):
        if x == 2:
            raise ValueError("bad item")
        return x
    with pytest.raises(ValueError, match="bad item"):
        parallel_map(boom, [0, 1, 2, 3])


def test_max_workers_one_is_serial_order():
    assert parallel_map(lambda x: x + 1, [1, 2, 3], max_workers=1) == [2, 3, 4]
