# Data and third-party attribution

PAVER code is covered by the root MIT license. That license does not relicense
source imagery, model weights, third-party code or external datasets.

The bundled source photographs were obtained from public Panoramax assets with
explicit `etalab-2.0` metadata, preserving their existing privacy blurring.
They are covered by [Etalab Open Licence 2.0](https://ia.numerique.gouv.fr/licence-ouverte-open-licence/).
Keep the original producer and update/capture information when reusing them;
do not imply producer endorsement. Each source ID, producer, license, capture
date, source URL and checksum remains in its metadata/manifests. Main sources
include Bordeaux Métropole, Bourges Plus, IGN contributors and Eurométropole
de Strasbourg; the per-image record is authoritative.

`data/attribution.jsonl` indexes source photographs named in project manifests
and the cached Panoramax catalog. Full metadata is in the release under
`dataset_v1/metadata`, with early demo sources in `sample`, `road_sample` and
`benchmark_v2/data` manifests. Manifest records identify resizing or other
processing. Derived crops, projections, masks and repaired previews were
produced by PAVER; repaired previews are modified research outputs, not original
Panoramax photographs or independently verified scene truth.

The LightGlue license is preserved in `third_party/LightGlue-LICENSE`.
LoFTR/Kornia, PyTorch and other dependencies retain their respective licenses.
No pretrained model weights are included. External datasets discussed in the
research notes are references, not a claim that their files are bundled or
cleared for redistribution.
