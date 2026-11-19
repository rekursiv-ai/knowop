"""``knowop watch-pr``: watch GitHub pull requests for agent sessions, from any host.

A pull request is one WebResult row per account on the server (:mod:`.rows`); each
session that watches it is an edge to the row. ``knowop watch-pr serve`` reads every
watched pull request of the account through the host's own ``gh``
(:mod:`.github`), writes what GitHub says to its row, and tells each watching
session's agent what is news (:mod:`.news`, :mod:`.watch`). Every fact a pass needs
is on the row, so a serve may stop, start again elsewhere, or run on two hosts at
once: a compare-and-set on the row lets one of them tell each piece of news.
"""
