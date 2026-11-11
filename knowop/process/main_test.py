"""The agent process's entry point, declared ahead of the process itself."""

from __future__ import annotations

from trackinizer.trax.profile import Profile

import pytest

from knowop.process.main import main


@pytest.mark.parametrize(
    "profile",
    [Profile(url="http://127.0.0.1:8765"), None],
    ids=["a person's profile", "started by a daemon"],
)
def test_the_agent_process_is_not_built_yet(profile: Profile | None) -> None:
    with pytest.raises(NotImplementedError) as raised:
        _ = main(["--headless"], profile=profile)

    assert str(raised.value) == "The agent process is not built yet."


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
