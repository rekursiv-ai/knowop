"""Export-root pytest resource setup for knowop.

Binds the resource-marker hook and the per-user-directory isolation at the
repository root so they reach tests living OUTSIDE the ``knowop`` package --
the pre-commit gates' tests under ``tools/`` in particular. A conftest hook
applies to its own directory and below, so a package-level conftest alone
would leave a ``tools`` test's resource marker with no timeout, no CI skip
policy, and no error to say so. Every knowop test logs in and saves a
profile, so the isolation keeps each one out of the developer's own config.
"""

from knowop.lib.testing.resources import (
    isolate_user_dirs,
    pytest_collection_modifyitems,
    pytest_configure,
)


__all__ = ["isolate_user_dirs", "pytest_collection_modifyitems", "pytest_configure"]
