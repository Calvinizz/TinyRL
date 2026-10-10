"""Spec tests for Timer (TODO P4.1). Red until implemented."""

import pytest

from tinyr.profiling import Timer


def make_timer(*ticks: float) -> Timer:
    """Timer whose clock returns the scripted ticks in order.

    Each section consumes exactly two ticks (begin + end), so a test's
    timing story reads left to right:
        make_timer(1.0, 2.5)  ->  one section lasting 1.5 s
    """
    it = iter(ticks)
    return Timer(clock=lambda: next(it))


class TestSection:
    def test_records_duration(self):
        t = make_timer(1.0, 2.5)
        with t.section("rollout"):
            pass
        assert t.last("rollout") == 1.5

    def test_exception_still_records(self):
        # __exit__ must run on the exceptional path: the wall time was
        # spent, and a section that dies mid-flight (e.g. rollout OOM)
        # still belongs in the accounting
        t = make_timer(0.0, 3.0)
        with pytest.raises(RuntimeError):
            with t.section("rollout"):
                raise RuntimeError("boom")
        assert t.last("rollout") == 3.0

    def test_exception_propagates(self):
        # __exit__ returning True would swallow the exception entirely
        t = make_timer(0.0, 1.0)
        with pytest.raises(RuntimeError):
            with t.section("train"):
                raise RuntimeError("must reach the test")


class TestQueries:
    def test_accumulates_and_means(self):
        t = make_timer(0.0, 1.0, 10.0, 12.0)
        with t.section("train"):
            pass
        with t.section("train"):
            pass
        assert t.last("train") == 2.0
        assert t.mean("train") == 1.5

    def test_unknown_section_raises(self):
        # a typo'd section name must be loud, not a silent 0.0
        t = make_timer()
        with pytest.raises(KeyError):
            t.last("nope")
        with pytest.raises(KeyError):
            t.mean("nope")

    def test_summary_covers_all_sections(self):
        t = make_timer(0.0, 1.0, 0.0, 2.0)
        with t.section("rollout"):
            pass
        with t.section("train"):
            pass
        assert t.summary() == {"rollout": 1.0, "train": 2.0}


class TestReentrancy:
    def test_nested_sections_rejected(self):
        # one section at a time: silently overwriting the outer start
        # time would corrupt both durations. Outer still needs its end
        # tick after the inner rejection.
        t = make_timer(0.0, 5.0)
        with t.section("outer"):
            with pytest.raises(AssertionError):
                with t.section("inner"):
                    pass
        # the rejection must not have corrupted the outer section
        assert t.last("outer") == 5.0
