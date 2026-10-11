"""Policy version stamping and staleness (TODO P6.2 / P6.3, Phase 6)."""

from __future__ import annotations

import threading


class PolicyVersion:
    """A thread-safe policy version counter.

    The version answers: "which optimizer step of the policy is this?"
    Every Experience is stamped with the version it was SAMPLED under
    (producer reads .current before generate()); the trainer bumps once
    per optimizer step. The gap between the two at consume time is the
    STALENESS of that experience — the number that async RL systems
    log, bound, and argue about.

    Why a lock, when "the GIL makes things thread-safe"?
    `self._v += 1` compiles to LOAD / ADD / STORE bytecodes — the thread
    can be preempted BETWEEN them, two bumps load the same old value,
    both store old+1, and one increment vanishes. The GIL makes each
    bytecode atomic, not the whole statement. tests/test_version.py
    races 4 threads x 250 bumps and will lose updates without the lock
    (run it enough times and a lucky kernel scheduling might hide it —
    the lock removes the possibility, not just the probability).
    """

    def __init__(self, start: int = 0):
        self._v = start
        self._lock = threading.Lock()

    @property
    def current(self) -> int:
        """The version right now. A single int read is atomic under the
        GIL, so no lock is needed HERE — only compound operations race."""
        return self._v

    def bump(self) -> int:
        """TODO P6.2: advance the version by one and RETURN the new value.

        One optimizer step = one bump. Take the lock around the
        read-modify-write (see class docstring for why `+=` alone is
        not safe). Return the value AFTER incrementing, so the caller
        can log "trained to version v".
        """
        with self._lock:
            self._v += 1
            return self._v
        


def policy_lag(current_version: int, stamped_version: int) -> int:
    """TODO P6.3: how stale is an experience?

    Args:
        current_version: the policy version at CONSUME time (what the
            trainer has already trained to)
        stamped_version: the version the experience was SAMPLED under
            (exp.policy_version, stamped by the producer)
    Returns:
        current_version - stamped_version

        0        = perfectly on-policy (sync mode, by construction)
        1, 2, .. = sampled k optimizer steps ago — off-policy drift
        negative = impossible in-flow (consumed a "future" version);
                   left unclamped so bugs show up as nonsense instead
                   of being silently folded into 0.
    """
    return current_version - stamped_version
