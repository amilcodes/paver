# Public snapshot provenance

This is a portable derivative of the local research, not an unchanged signed
archive. Local research-root prefixes in text records were replaced with relative
paths. Artifact SHA-256 references were then recalculated through the dependency
graph. The ledger pairs original and exported hashes and identifies each
transformation. Numeric JSON payloads were checked for exact preservation after
normalizing only paths and SHA-256 strings. Source Python, image and NPZ bytes
were not changed. SQLite was exported through a consistent backup.

Historical audit verdicts describe the historical runs. Path/hash normalization
is packaging, not a new model experiment. `verify_snapshot.py` checks the exported
file ledger; `replay_solver_audit.py` additionally replays the latest scientific
audit without changing its historical output. The new replay result is recorded
separately.

Excluded: local crash/installer cleanup records, personal environment and live
automation/chat state, locks, Python caches, the old empty Git directory, and
copied external repository README captures. Source URLs are retained in research
notes. These are not omissions of unsuccessful scientific trials. Third-party
model weights and the environment lived outside the research directory and are
not redistributed. No secrets or personal account emails are needed to use the
repository.

Images and bulk evidence are in the attached release; Git includes the exact
manifest. Each archive is independently extractable, and the fetcher checks
both archive and payload SHA-256 values. Download only the parts you need with
`--part NAME`; the default fetches the complete frozen snapshot. Do not treat
an incomplete asset download as a completed audit or benchmark.
