"""Shape / dtype / device assertion helpers.

Phase 2 engineering task: every core component validates its tensors with
these instead of hand-rolled `assert` statements, so failures name the
offending tensor and what was expected.
"""

import torch


def assert_shape(tensor: torch.Tensor, expected: tuple, name: str) -> None:
    """expected entries: int for an exact size, None for "any".

    Example: assert_shape(x, (None, 3), "advantages") checks dim 1 == 3.
    """
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(f"{name}: expected torch.Tensor, got {type(tensor)}")
    if tensor.dim() != len(expected):
        raise AssertionError(
            f"{name}: expected {len(expected)} dims {expected}, "
            f"got shape {tuple(tensor.shape)}"
        )
    for i, want in enumerate(expected):
        if want is not None and tensor.shape[i] != want:
            raise AssertionError(
                f"{name}: expected shape {expected}, got {tuple(tensor.shape)}"
            )


def assert_dtype(tensor: torch.Tensor, dtypes, name: str) -> None:
    """dtypes: a torch dtype or a tuple of allowed dtypes."""
    if not isinstance(dtypes, tuple):
        dtypes = (dtypes,)
    if tensor.dtype not in dtypes:
        raise AssertionError(
            f"{name}: expected dtype in {[str(d) for d in dtypes]}, "
            f"got {tensor.dtype}"
        )


def assert_device(tensor: torch.Tensor, device, name: str) -> None:
    want = torch.device(device) if isinstance(device, str) else device
    if tensor.device != want:
        raise AssertionError(
            f"{name}: expected device {want}, got {tensor.device}"
        )
