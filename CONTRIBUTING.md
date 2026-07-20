# Contributing

OCR Model is a solo academic project maintained by Sithu Win San. The project
is not adding collaborators, co-maintainers, or team members. That boundary
does not prevent community contribution: focused issues and pull requests are
welcome.

## Before opening an issue

Search existing issues first. For a bug, include:

- the operating system and Python version;
- whether the portable Release or source checkout was used;
- the exact command or GUI action;
- the expected and actual behavior; and
- a minimal, non-sensitive sample or synthetic reproduction when possible.

Do not attach real invoices, contracts, receipts, identity documents, account
details, private OCR output, credentials, or model files.

## Pull requests

Good pull-request scopes include:

- reproducible bug fixes;
- focused OCR, extraction, GUI, portability, or accessibility improvements;
- tests for an existing behavior;
- corrections to setup or technical documentation; and
- privacy-preserving synthetic examples.

Keep each pull request narrow and avoid unrelated formatting or generated-file
changes.

1. Fork the repository and create a descriptive branch.
2. Make the smallest change that solves the problem.
3. Add or update regression tests where practical.
4. Run the affected tests. For broad Python changes, run:

   ```powershell
   python -m pytest -q
   python -m compileall -q src scripts tests
   ```

5. Update documentation only when behavior, setup, or interfaces change.
6. Open a pull request that explains the problem, approach, verification, and
   remaining limitations.

Acceptance is not guaranteed and does not grant collaborator, maintainer, or
team-member status.

## Privacy and repository hygiene

Never commit or upload:

- raw or private documents and their page renders;
- OCR text or predictions derived from private documents;
- `runtime.local.json`, `.env` files, tokens, passwords, or credentials;
- local virtual environments, caches, logs, or user outputs;
- large model checkpoints outside the established Release process; or
- material whose license does not permit redistribution.

Use synthetic or clearly redistributable public samples in tests. If a change
touches path validation, private-output handling, manifests, or packaging,
describe the privacy checks performed.

## Licensing

By contributing original source code or documentation, you agree that your
contribution may be distributed under this repository's MIT License. Trained
weights, datasets, and third-party components remain under their own upstream
licenses; see [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).
