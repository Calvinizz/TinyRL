"""ExperienceQueue: bounded blocking hand-off between rollout and trainer (TODO P6.1)."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

from tinyr.experience.experience import Experience


class ExperienceQueue:
    """A bounded, thread-safe FIFO of Experiences (producer/consumer).

    The rollout thread `put`s finished experiences; the trainer thread
    `get`s them. Bounded = BACKPRESSURE: when the trainer is slow, the
    queue fills, and `put` BLOCKS — the rollout side cannot run more
    than `maxsize` steps ahead. That is what bounds staleness in an
    async pipeline (and, combined with the policy-version stamp, what
    the max-staleness cap polices).

    Implement with a `collections.deque` + one `threading.Condition`
    (NOT queue.Queue — writing the wait/notify dance once by hand is
    the exercise). The idiom for both directions:

        with self._cond:                  # the condition's lock guards
            while <not my turn>:          # the deque AND the counters
                self._cond.wait()         # releases the lock while asleep
            ...mutate deque...
            self._cond.notify_all()       # wake the other side

    `while`, not `if`, around wait(): the woken thread must re-check —
    another thread may have consumed the slot first (spurious wakeups
    and lost races are real; `if` works in tests and deadlocks in
    production).

    `None` is the shutdown sentinel: a producer puts None to say "no
    more work"; get() passes it through untouched (the consumer loop
    terminates on it). Nothing special-cased inside the queue.
    """

    def __init__(self, maxsize: int = 2):
        self._maxsize = maxsize
        self._items: deque = deque()
        self._cond = threading.Condition()
        self._n_put = 0
        self._n_get = 0

    def put(self, item: Experience | None) -> None:
        """TODO P6.1 (a): enqueue one item; BLOCK while the queue is full.

        Count every successful put in _n_put (under the lock).
        """
        raise NotImplementedError

    def get(self) -> Experience | None:
        """TODO P6.1 (b): dequeue one item; BLOCK while the queue is empty.

        Count every successful get in _n_get (under the lock).
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Current depth (advisory under races — fine for logging)."""
        # framework: __len__ dunder -> len(queue) works (len() protocol)
        return len(self._items)

    @property
    def n_put(self) -> int:
        """Total items ever enqueued (for the benchmark's accounting)."""
        return self._n_put

    @property
    def n_get(self) -> int:
        """Total items ever dequeued."""
        return self._n_get
