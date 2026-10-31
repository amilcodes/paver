# PAVER research snapshot — October 31, 2026

Panoramic Alignment and View-based Evidence Recovery.

This release preserves the code, protocols, annotations, complete controls, negative results and cached numerical evidence through the solver-budget experiment. The public Git history contains 17 reconstructed research milestones plus publication work, with historical author dates based on recorded file modification times and actual import committer dates. It is not an original recovered Git history.

The snapshot includes three contiguous development areas (247 photospheres), nine reviewed anchors, 216 controlled conditions and the 353-collection catalog. The selected 100-sequence cohort is only partially acquired: 25 complete, one incomplete, 74 pending. No claim of reliable recovery across all areas is made: Nantes, Bourges and Alsace still abstain; the latest improvement was rejected as a default because some error tails worsened. Real vehicle footprints have no clean target truth.

Assets are 24 independently extractable tar.gz archives, totaling approximately 1.93 GB compressed / 2.27 GB unpacked. Use `python tools/fetch_assets.py` from the checkout to fetch and verify them. The archive and individual payload hashes are recorded in `publication/release-assets.json` and `publication/snapshot.jsonl`. Original/exported hashes are both retained for portable path normalization; numerical payloads are unchanged.

Validation: 138 tests passed with assets (two opt-in model checks skipped); 136 passed in a lightweight checkout (four explicitly explained skips). The latest scientific audit replay passed for 126 groups / 216 conditions with 1,128 artifact hashes checked and no new matching or optimization.

Code: MIT. Public Panoramax imagery: Etalab Open Licence 2.0, with producer/capture/source attribution preserved. External model weights and local environments are not included. See `DATA_LICENSE.md`, `data/attribution.jsonl`, and `docs/HISTORY.md`.
