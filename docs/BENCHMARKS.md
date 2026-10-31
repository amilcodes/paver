# Reading the benchmarks

`benchmark_v2` is the preserved early nine-road development benchmark.
`sequence_v1`–`sequence_v7` preserve successive geometry/matching experiments,
including rejected changes. `area_sets_v1` freezes the three contiguous areas;
`network_v1` separates retrieval links from query-specific visual evidence.
`controlled_v1` contains the first three anchors; `controlled_v2` adds six
spatially separated anchors. `sampling_v1` changes matcher input from 640 to
960 pixels; `solver_budget_v1` reuses those observations with a bounded solver
budget change. No default promotion followed either change.

Coverage is accepted pixels divided by the deliberately hidden region. RGB MAE
is measured only on accepted pixels, so raising abstention can lower error.
Always inspect both, including common/gained/lost pixel partitions, gradients,
error tails, textured subsets and the failed queries. Area-weighted results and
spherical solid-angle results are different quantities. A large square in a
downward projection can cover very little angular area near the horizon.

Synthetic damage is applied to reviewed visible static ground. Its clean RGB
can score the answer but cannot participate in reconstruction. Natural
vehicle/rig/graphic masks have no observed clean truth; their results contain
support statistics only. Road texture can be nearly uniform, so low RGB error
does not establish precise geometry. Moving traffic and shadows are not missing
static pavement. Shared masks across damage families do not create independent
geometry trials.

All repeatedly used roads and the three areas are development/train. The
catalog's route/geographic splits and frozen cohort are retained, but a
generalization claim requires new reviewed, independent evaluation collections.
The held-out catalog is metadata-only in this snapshot.
