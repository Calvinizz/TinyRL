"""ExperienceBuffer: a bounded FIFO between rollout and training.

In the current synchronous loop it only ever holds one experience, but
the interface is what Phase 6 (rollout/training decoupling) will build
the async experience queue on.
"""

from collections import deque

from tinyr.experience.experience import Experience


class ExperienceBuffer:
    def __init__(self, max_size: int = 8):
        self._buffer: deque[Experience] = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, experience: Experience) -> None:
        """TODO 4a: append an experience.

        deque(maxlen=...) already drops the OLDEST item when full — think
        about whether you want that behavior made explicit here (or log /
        count the evictions for Phase 6 staleness tracking).
        """
        self._buffer.append(experience)

    def get(self) -> list[Experience]:
        """TODO 4b: return buffered experiences oldest-first and CLEAR them.

        Drain semantics: after get() returns, the buffer must be empty —
        the synchronous training loop consumes everything each step, and
        retraining stale experiences by accident would silently change
        the algorithm.
        """
        items = list(self._buffer)
        self._buffer.clear()
        return items
        

    def __len__(self) -> int:
        """TODO 4c: number of buffered experiences."""
        return len(self._buffer)