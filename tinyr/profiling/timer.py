"""Timer: named-section wall-clock accounting for Phase 4 profiling."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter


class _Section:
    """The context manager `Timer.section()` hands back.

    Thin handle: all state lives in the Timer. Dunder protocol recap —
    `with timer.section("rollout"):` calls __enter__ on entry and
    __exit__ on exit (normal OR exceptional), which is exactly the
    guarantee timing needs: the clock stops even if the section body
    raises.
    """

    def __init__(self, timer: "Timer", name: str):
        self._timer = timer
        self._name = name

    def __enter__(self) -> "_Section":
        """TODO P4.1 (a): tell the timer this section starts NOW.

        One line: delegate to the timer's _begin(name), then return
        `self` (the `as` variable — unused by callers, but the protocol
        expects a value).
        """
        self._timer._begin(self._name)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """TODO P4.1 (b): tell the timer this section ended NOW.

        Delegate to the timer's _end(). Return False (or None — same
        thing) so exceptions from the body KEEP propagating; returning
        True would silently swallow them. __exit__ runs on the
        exceptional path too — that is why the duration still gets
        recorded when a section dies mid-flight.
        """
        self._timer._end()
        return False


class Timer:
    """Accumulates wall-clock durations under named sections.

    Usage:
        timer = Timer()
        with timer.section("rollout"):
            ...generate...
        with timer.section("train"):
            ...update...
        timer.last("rollout")   # most recent duration, for the step log
        timer.mean("train")     # average across steps, for benchmarks

    `clock` is injectable so unit tests can feed a scripted sequence of
    timestamps instead of sleeping — same dependency-injection trick a
    deterministic test of anything time-related needs.

    One section at a time: nesting is a caller bug and _begin must
    reject it with an AssertionError (silent overwrite would lose the
    outer section's start time and produce garbage durations).
    """

    def __init__(self, clock: Callable[[], float] = perf_counter):
        self._clock = clock
        # section name -> list of recorded durations (seconds)
        self._log: dict[str, list[float]] = {}
        self._start: tuple[str, float] | None = None

    # -- internal: called by _Section --------------------------------

    def section(self, name: str) -> _Section:
        """Hand back the context manager that times `name`."""
        return _Section(self, name)

    def _begin(self, name: str) -> None:
        """TODO P4.1 (c): open a section.

        Assert no section is already open (nesting), then store
        (name, clock()) in self._start. Do NOT append anything to the
        log yet — the duration only exists once the section closes.
        """
        assert self._start is None
        self._start = (name,self._clock())

    def _end(self) -> None:
        """TODO P4.1 (d): close the open section.

        Read clock(), pop self._start, and append (end - start) to that
        section's duration list. Creating the list on first sight of the
        name is fine (dict.setdefault).
        """
        assert self._start is not None
        end = self._clock()
        name, start = self._start
        self._start = None
        duration = end - start
        self._log.setdefault(name,[]).append(duration)
        

    # -- queries ------------------------------------------------------

    def last(self, name: str) -> float:
        """Most recent duration of `name` (seconds).

        Raises KeyError if the section never ran — a typo'd section name
        should be loud, not silently 0.0.
        """
        # TODO P4.1 (e)
        return self._log[name][-1]

    def mean(self, name: str) -> float:
        """Arithmetic mean of all durations of `name`.

        Same KeyError contract as last().
        """
        # TODO P4.1 (f)
        return sum(self._log[name]) / len(self._log[name])

    def summary(self) -> dict[str, float]:
        """{section_name: mean_seconds} for every section seen."""
        # TODO P4.1 (g)
        summary =  dict()
        for name in self._log: 
            summary[name] = self.mean(name)
        return summary
        
