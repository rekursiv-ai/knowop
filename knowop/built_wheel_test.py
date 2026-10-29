"""The knowop wheel, built and run outside the checkout."""

from __future__ import annotations

import pytest

from knowop.lib.testing.wheel import built_wheel


pytestmark = pytest.mark.cli_uv


def test_the_wheel_builds_its_webrtc_peer_outside_the_checkout() -> None:
    wheel = built_wheel("knowop", anchor=__file__)

    wheel.run(
        "import aiortc; "
        "from knowop.shell.terminal_direct_aiortc import aiortc_peer_factory; "
        "assert aiortc_peer_factory() is not None",
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
