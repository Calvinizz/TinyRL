"""MemoryProfiler: named CUDA-memory snapshots for Phase 3 benchmarking."""

import torch


class MemoryProfiler:
    """Records `snapshot()`s under labels and renders a markdown report.

    Usage:
        prof = MemoryProfiler()
        prof.mark("after model load")
        ... training steps ...
        prof.mark("after training")
        print(prof.report())
    """

    def __init__(self) -> None:
        self._marks: list[tuple[str, dict[str, float]]] = []

    def snapshot(self) -> dict[str, float]:
        """TODO P3.1: one CUDA memory snapshot.

        Returns a dict with exactly these float keys (bytes):
            {"allocated": ..., "reserved": ..., "peak": ...}

        - allocated: torch.cuda.memory_allocated()
        - reserved:  torch.cuda.memory_reserved()
        - peak:      torch.cuda.max_memory_allocated()

        All three are plain ints — cast with float(). They return 0 on a
        CPU-only box, which is what lets the unit tests run anywhere.
        """
        raise NotImplementedError("TODO P3.1: implement MemoryProfiler.snapshot")

    def mark(self, label: str) -> dict[str, float]:
        """Take a snapshot and store it under `label`."""
        snap = self.snapshot()
        self._marks.append((label, snap))
        return snap

    @property
    def peak(self) -> float:
        """Highest 'peak' value seen across all marks (bytes)."""
        return max((s["peak"] for _, s in self._marks), default=0.0)

    def report(self) -> str:
        """Markdown table of all marks; values in GiB."""
        lines = [
            "| section | allocated (GiB) | reserved (GiB) | peak (GiB) |",
            "|---|---|---|---|",
        ]
        for label, s in self._marks:
            gib = lambda v: f"{v / 1024 ** 3:.3f}"
            lines.append(
                f"| {label} | {gib(s['allocated'])} | {gib(s['reserved'])} "
                f"| {gib(s['peak'])} |"
            )
        lines.append(f"| **overall peak** | | | **{self.peak / 1024 ** 3:.3f}** |")
        return "\n".join(lines)
