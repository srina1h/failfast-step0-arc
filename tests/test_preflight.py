import pytest

from failfast_arc.preflight import check_headroom


GIB = 1024**3


def test_accepts_exact_headroom():
    check_headroom(16 * GIB, 15 * GIB, GIB)


def test_rejects_insufficient_headroom():
    with pytest.raises(RuntimeError, match="headroom"):
        check_headroom(16 * GIB, 15 * GIB + 1, GIB)
