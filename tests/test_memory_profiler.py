"""Spec tests for MemoryProfiler (TODO P3.1). CPU-safe: CUDA queries
return 0 without a GPU, so these only check structure, not values."""

from tinyr.profiling import MemoryProfiler


def test_snapshot_has_three_float_keys():
    snap = MemoryProfiler().snapshot()
    assert set(snap) == {"allocated", "reserved", "peak"}
    assert all(isinstance(v, float) for v in snap.values())
    assert all(v >= 0 for v in snap.values())


def test_mark_and_report():
    prof = MemoryProfiler()
    prof.mark("after load")
    prof.mark("after training")
    report = prof.report()
    assert "after load" in report
    assert "after training" in report
    assert "overall peak" in report
