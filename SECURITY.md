# Security Policy

## Why this file exists

knowop holds an API key for your account, serves this computer's shell to the
browser, runs the commands the server's assistant sends, runs agents that act on
your files, and keeps the secrets your jobs use. Security reports need a private
path so exploit details are not published before review.

## Reporting a vulnerability

Please report suspected security vulnerabilities privately by emailing hello@rekursiv.ai.

Include:

- Affected version or commit.
- Steps to reproduce.
- Expected impact.
- Any suggested mitigation.

Please do not open public issues for vulnerabilities until we have investigated and coordinated disclosure.

## Scope

Security reports are especially useful for:

- Login: a key or enrollment token shown, logged or saved where another user
  can read it, or a login approved for the wrong computer.
- Logout leaving a usable key behind.
- `knowop connect` serving a terminal, or running commands, for anyone but the
  account's owner.
- A job secret shown, logged or sent anywhere but the job it was given to.
- `knowop` acting on a server or profile other than the one it logged in to.
- Dependency or packaging issues that affect installed users.
- Supply-chain concerns in the published wheel or its dependency set.
