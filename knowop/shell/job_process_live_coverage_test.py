"""A runner's separate process group can be terminated and reaped."""

import os
import signal
import subprocess


def test_killpg_reaps_a_runner_started_in_its_own_session() -> None:
    with subprocess.Popen(["/bin/sleep", "30"], start_new_session=True) as child:
        try:
            assert os.getpgid(child.pid) == child.pid
            os.killpg(child.pid, signal.SIGTERM)
            assert child.wait(timeout=0.1) == -signal.SIGTERM
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=1)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
