"""What ``knowop run``'s agent process needs to continue a stored session here.

``resume.session_journal`` picks the journal a process writes a session in, restoring
one from the stored parts (``materialize``) when this host holds none whole;
``truncate`` bounds what a record upload carries; ``placement`` asks the server where
a session sits.
"""

from __future__ import annotations
