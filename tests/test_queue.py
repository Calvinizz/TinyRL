"""Spec tests for Phase 6 (TODO P6.1): ExperienceQueue. Red until implemented.

The queue holds arbitrary items in these tests (ints) — it never inspects
them; only the poison-pill `None` is meaningful to the consumer LOOP.
"""

import threading

import pytest

from tinyr.experience import ExperienceQueue


class TestBasics:
    def test_fifo_order(self):
        q = ExperienceQueue(maxsize=4)
        for i in (1, 2, 3):
            q.put(i)
        assert [q.get(), q.get(), q.get()] == [1, 2, 3]

    def test_len_tracks_depth(self):
        q = ExperienceQueue(maxsize=4)
        q.put(1)
        q.put(2)
        assert len(q) == 2
        q.get()
        assert len(q) == 1

    def test_put_get_counters(self):
        q = ExperienceQueue(maxsize=4)
        q.put(1)
        q.put(2)
        q.get()
        assert q.n_put == 2
        assert q.n_get == 1

    def test_none_sentinel_passes_through(self):
        # None means "no more work" — the queue must hand it over as-is,
        # not reject it and not count it differently.
        q = ExperienceQueue(maxsize=2)
        q.put(None)
        assert q.get() is None
        assert q.n_put == 1 and q.n_get == 1


class TestBlocking:
    def test_get_blocks_until_put(self):
        q = ExperienceQueue(maxsize=1)
        released = threading.Event()
        put_done = threading.Event()

        def delayed_put():
            released.wait(timeout=5.0)
            q.put(42)
            put_done.set()

        threading.Thread(target=delayed_put).start()
        got = []
        getter = threading.Thread(target=lambda: got.append(q.get()))
        getter.start()
        getter.join(timeout=0.2)
        # empty queue: the getter must still be waiting, no value yet
        assert got == []
        released.set()
        assert put_done.wait(timeout=5.0)
        getter.join(timeout=5.0)
        assert got == [42]

    def test_put_blocks_when_full_then_proceeds(self):
        q = ExperienceQueue(maxsize=1)
        q.put(1)  # full now
        put_done = threading.Event()

        def blocked_put():
            q.put(2)  # must block until the consumer frees a slot
            put_done.set()

        threading.Thread(target=blocked_put).start()
        assert not put_done.wait(timeout=0.2)  # still blocked
        assert q.get() == 1  # free the slot
        assert put_done.wait(timeout=5.0)      # put goes through
        assert q.get() == 2


class TestStress:
    def test_two_producers_one_consumer(self):
        # 80 items from 2 producers; every item arrives exactly once and
        # per-producer FIFO is preserved (a queue scrambling cross-producer
        # order is fine, scrambling WITHIN a producer is not).
        n_per, producers = 40, 2
        q = ExperienceQueue(maxsize=4)

        def produce(pid: int):
            for i in range(n_per):
                q.put((pid, i))
            q.put(None)

        for pid in range(producers):
            threading.Thread(target=produce, args=(pid,)).start()

        received: list[tuple[int, int]] = []
        sentinels = 0
        while sentinels < producers:
            item = q.get()
            if item is None:
                sentinels += 1
            else:
                received.append(item)

        assert len(received) == n_per * producers
        assert set(received) == {
            (pid, i) for pid in range(producers) for i in range(n_per)
        }
        for pid in range(producers):
            seq = [i for p, i in received if p == pid]
            assert seq == sorted(seq)
