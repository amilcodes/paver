# Panorama Repair

A local research prototype that transfers observed road pixels from neighboring 360° panoramas into the camera vehicle's footprint. It keeps donor provenance and abstains where the current rules cannot support a transfer.

**Start with [the matching benchmark](benchmark.html).** The [first demo](report.html) is retained. [Experiment 002 findings](docs/EXPERIMENT_002.md) explain the results and limitations.

## Current result

45 distinct licensed Panoramax images, nine neighborhoods, four donors per target. Five new car/van scenes, two new small-rig stress cases, and two original development scenes. This is a French convenience sample with manual masks, not a global benchmark.

On the five new car/van scenes, learned matching raises average candidate footprint coverage from **21.6% to 29.4%** under common inputs and gates. Requiring two agreeing donors reduces learned coverage to **3.9%**, or **2.9%** under the strict policy. These are proposed transfer rates; ground beneath the actual car has no reference.

The marked-lane development case reaches **83.5%** candidate coverage, but large artificial car-shaped occlusions on new roads mostly abstain. Smaller exploratory probes favor source transfer over Telea on accepted pixels, while the learned matcher does not beat SIFT's aggregate local-probe accuracy. Reliable full-car reconstruction remains unproven. All failures are retained in the report.

## View locally

From this directory:

```sh
python3 -m http.server 8766 --bind 127.0.0.1
```

Open [the report](http://127.0.0.1:8766/benchmark.html). It includes original/partial repair, support overlays, hidden references, SIFT/learned toggles, rejection diagnostics, all-scene tables and a 360° viewer. No network libraries or external services are used by the report. The sphere uses 2048×1024 preview exports. Unsupported regions retain the original vehicle.

## Reproduce

This run used Python 3.14 and pinned package versions. A local environment is under the parent workspace's `work/venv`; source and outputs are all inside this project directory. Install locally in your own environment if copying the project elsewhere:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-matching.txt
export TORCH_HOME="$PWD/.model-cache"

.venv/bin/python -m pytest tests -q
# Optional learned-feature exclusion integration test, downloads weights if absent:
RUN_LEARNED_TESTS=1 .venv/bin/python -m pytest tests -q

.venv/bin/python benchmark.py --method sift
.venv/bin/python benchmark.py --method disk_lightglue
.venv/bin/python local_probes.py --method sift
.venv/bin/python local_probes.py --method disk_lightglue
.venv/bin/python make_benchmark_report.py
```

The learned matcher uses Apple MPS when available, otherwise CPU; use `--device cpu` to override. Model weights total approximately 50 MB; the tensor runtime uses substantially more disk and memory. Run the jobs sequentially on memory-constrained machines. Full benchmark images and results are bundled. First model use downloads public weights; [recorded checksums](benchmark_v2/model_manifest.json) identify this run. Pretrained models are used without fine-tuning. Bitwise results may differ across devices or library versions.

`benchmark.py` verifies source checksums, duplicate IDs, protocol and implementation hashes. It resumes compatible completed cases. Use `--case <name>` to select a case and `--force` to explicitly replace incompatible core results. The supplemental script refuses incompatible existing results. Preserve existing results before deliberate protocol changes.

To fetch the 45 specifically pinned observations again, use `fetch_benchmark.py`. It checks licenses and collection boundaries and stores full metadata. It reuses existing prepared files and records. For an exact rerun, use bundled samples: the acquisition preparation switched to JPEG draft decoding partway through to reduce memory, and lossy resize/re-encode paths may not reproduce byte-identical samples on a fresh fetch. Every prepared image is hashed and shared by both compared methods. Raw acquisition methods do not alter the within-scene comparison.

## Pipeline and evidence

1. Project a 2:1 panorama downward at fixed focal length.
2. Manually exclude the vehicle, non-ground and distracting objects.
3. Zero excluded target/source content before feature extraction.
4. Match SIFT or DISK + LightGlue; fit a planar homography with MAGSAC.
5. Audit geometric support, transform plausibility, correspondence stability and visible-ground appearance.
6. Warp source observations only inside their safe masks and target inlier hulls.
7. Reject conflicting observations; select one observed donor, without averaging or generative filling.

Policies: candidate requires ≥1 donor and maximum pairwise RGB disagreement ≤18; corroborated requires ≥2 and ≤12; strict requires ≥2 and ≤8. These are heuristic labels, not confidence probabilities.

Each `*_evidence.npz` saves full-resolution `mask`, `provenance`, `count`, and `disagreement`. Provenance indexes `[prev1, next1, prev2, next2]`; −1 means no candidate. Evidence arrays are saved before policy filtering with threshold 255: derive policy acceptance from count/disagreement and the saved protocol, rather than treating every provenance entry as an approved repair. Metrics include every accepted/rejected homography, source/target point correspondences, accepted coverage and same-pixel Telea comparisons.

Core car-shaped hidden patches were specified before scoring new scenes. Smaller probes were added after the large-patch abstentions and are explicitly exploratory. Pixel errors are descriptive, spatially correlated RGB differences; no calibrated wrong-transfer rate or real-under-car accuracy is claimed.

## Project map

- `repair.py`: projection, fusion, scoring and equirectangular export.
- `matching_v2.py`: masked feature extractors and audited registration shared by both matchers.
- `benchmark.py`, `local_probes.py`: core and supplemental evaluation.
- `benchmark_v2/`: 45 source images, provenance, frozen protocols, full-resolution evidence, numeric results and model manifest.
- `make_benchmark_report.py`, `docs/benchmark_template.html`: interactive report and crops.
- `tests/`: geometry, abstention, source exclusions, hidden-pixel leakage and export preservation checks.
- `docs/EXPERIMENT_002.md`: findings, failed cases, methodological limits and next experiment.
- `docs/RESEARCH.md`: inherited-research corrections and related literature.
- `run_experiment.py`, `sample/`, `road_sample/`, `results/`, `road_results/`, `wide_results/`, `make_report.py`: original SIFT development experiments, preserved.

The original baseline can be rerun without tensor/model dependencies using `requirements.txt`, `run_experiment.py`, and `make_report.py`. It has separate masks and gates; do not compare its percentages directly with the new controlled matcher comparison.

## Attribution

Imagery is from **Eurométropole de Strasbourg, Bordeaux Métropole, Ecartip, Immergis and sig@bretagne.bzh**, distributed by **Panoramax IGN**, captured in 2023–2026. Every selected image's metadata specifies [Licence Ouverte / Open Licence 2.0](https://ia.numerique.gouv.fr/licence-ouverte-open-licence/). Exact producer, timestamp, source URL and identifiers are in `benchmark_v2/data/<scene>/manifest.json`. Existing privacy blurring is retained. Modifications include resize/re-encoding, projection, masks, diagnostic overlays and observed-pixel transfer. No provider endorsement is implied.

Learned matching uses the [official LightGlue repository](https://github.com/cvg/LightGlue) at the commit pinned in `requirements-matching.txt`, with DISK rather than SuperPoint. Its Apache-2.0 license is included under `third_party/`. No cloud deployment, remote repository or publication has been created.
