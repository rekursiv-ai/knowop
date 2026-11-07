"""Tests for ``knowop.agent.errors``."""

from __future__ import annotations

from itertools import permutations
from typing import Final

import pytest

from knowop.agent.errors import (
    CiphertextDroppedError,
    LossyConversionError,
    NotResumableError,
)


_ERRORS: Final = (CiphertextDroppedError, LossyConversionError, NotResumableError)


@pytest.mark.parametrize("error", _ERRORS)
def test_each_error_is_a_class_a_handler_catches(error: type[Exception]) -> None:
    """An eagerly bound class, so the handler catches rather than raising TypeError."""
    with pytest.raises(error, match=r"^stored session$"):
        _raise(error)


def test_no_error_is_caught_as_another() -> None:
    for error, other in permutations(_ERRORS, 2):
        assert not issubclass(error, other)


def _raise(error: type[Exception]) -> None:
    raise error("stored session")


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
