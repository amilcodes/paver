# Experiment 002 — matching, road variety, and recoverability

DISK + LightGlue increases proposed footprint coverage on several streets, but these tests do **not** establish reliable full-car removal. On five new car/van scenes, mean candidate coverage is **29.4%**, versus **21.6%** for SIFT under identical inputs and gates. Requiring at least two agreeing donors reduces learned coverage to **3.9%**, or **2.9%** with the tighter disagreement threshold. These are proposed transfers, not measured correctness under the vehicle.

[Open the interactive comparison](../benchmark.html). [Exact aggregates](../benchmark_v2/summary.json).

## What changed

- Pretrained DISK features and LightGlue matching; no training on these samples.
- Common MAGSAC planar fitting for learned and SIFT correspondence sets.
- Reject weak spatial support, unstable held-out-correspondence geometry, reflection, projective horizons, implausible scales, and poor appearance agreement on visible ground.
- Four distinct neighboring observations, with a two-neighbor ablation.
- Nine five-panorama neighborhoods: the original two development scenes, five new car/van scenes and two small-rig stress cases. All 45 images have distinct IDs. All seven new scenes come from different capture collections; the two original scenes share a collection.
- Full-resolution donor identity, support counts and disagreement arrays. UI overlays distinguish one donor, multiple agreeing donors and abstention. Partial 360 previews retain the original outside support.

The field of view is 163.30° at 2200 × 2200, maintaining the original focal length in pixels. Input panoramas are resized to at most 6144 × 3072. Each method sees identical prepared images and manual masks. Ground masks omit nearby vehicles, walls, vegetation and poles as conservatively as practical; the remaining annotation errors are a limitation. Padded rig masks include some visible pavement and are not pixel-perfect object silhouettes.

## All actual-footprint results

| New car / van scene | SIFT candidate | Learned candidate | Learned ≥2 donors, ΔRGB≤12 | Learned ≥2 donors, ΔRGB≤8 |
|---|---:|---:|---:|---:|
| Suburban asphalt, Strasbourg | 35.0% | 44.5% | 0.0% | 0.0% |
| Marked junction, Alsace | 0.0% | 0.0% | 0.0% | 0.0% |
| Narrow stone street, Bordeaux | 23.5% | 37.8% | 12.6% | 11.6% |
| Rural asphalt, Gironde | 0.0% | 0.0% | 0.0% | 0.0% |
| Shaded cycle street, Nantes | 49.3% | 64.7% | 6.8% | 2.7% |
| **Equal-scene mean, failures included** | **21.6%** | **29.4%** | **3.9%** | **2.9%** |

Both small-rig cases (Lyon paving/poles and Rennes crosswalk/rotation) fully abstain under both matchers. They are excluded from the car-scene mean. The original marked-lane development case reaches 83.5% learned candidate coverage versus 15.1% SIFT under this common protocol, with 19.6% corroborated coverage. The original repetitive paving reaches 17.6% versus 0%, all singly supported.

The exploratory ~98% learned transfer reported earlier used a different 1600-pixel development setup. It is not the standardized result above, nor verified accuracy. Likewise, the first demo's 55.6% SIFT result used different masks, field of view and gates. Only within-protocol comparisons isolate matching here.

Adding two more donors helps Bordeaux: candidate coverage rises from 25.1% with immediate neighbors to 37.8% with four donors. Most other scenes receive no additional coverage. Extra views alone do not solve sparse texture or bad geometry.

## Accuracy on pixels we can observe

The actual road under the vehicle has no reference. We therefore re-run registration with visible ground deliberately hidden. Both the real rig and the test patch are zeroed before feature extraction; target photometric audits exclude the patch. Donors remain their independent observed frames.

The core test shifts the complete padded rig polygon ahead and behind the vehicle, intersects it with annotated visible ground, and reports the retained area. Some masks are clipped substantially, especially on the narrow street. This tests an additional large occlusion while the real car remains present; it is harder than the actual-footprint problem and has different donor geometry. Its failure cannot be translated into a direct upper bound on actual repair accuracy, but it prevents us from claiming full-car validation.

Across the five new car scenes, those large masks contain 1,842,123 pixels. The learned candidate policy transfers only **0.36%**; SIFT transfers **0.18%**. Requiring two donors leaves **zero** large-mask transfers. Learned accepted-pixel MAE is 7.52/255, versus Telea's 3.12 on exactly those same pixels. The available large-mask evidence does not establish an advantage over inpainting.

After observing these abstentions, two smaller, fixed local probes were added per scene: 0.08 × 0.06 rectangles just ahead and behind the rig, clipped to allowed ground. These are explicitly exploratory. They are not an independent final test set, and their success does not validate an entire footprint.

| New car scenes, local probes | Coverage of tested pixels | Transfer MAE | Telea MAE on same pixels | Accepted pixels with mean RGB error >20 |
|---|---:|---:|---:|---:|
| SIFT candidate | 19.38% | 4.62 | 11.44 | 0.287% |
| Learned candidate | 19.31% | 5.36 | 12.59 | 0.948% |
| Learned corroborated | 2.02% | 3.67 | 12.26 | 0.085% |
| Learned strict | 1.75% | 3.15 | 11.20 | 0 / 4,087 pixels |

Learned matching therefore **does not outperform SIFT on aggregate local-probe accuracy or coverage** in this small sample, despite increasing actual-footprint candidate coverage. Each matcher accepts a different set of pixels; the SIFT versus learned MAEs are not a paired same-pixel comparison. The Telea comparisons are paired within each row.

All corroborated local-probe pixels in the new car scenes come from Bordeaux. Zero >20 errors among 4,087 strict pixels is not a zero-error guarantee: these pixels are highly correlated and come from one scene. Exposure and shadows also affect RGB error, so it is not a calibrated geometric wrong-transfer rate. Agreement between adjacent views is itself correlated.

## Interpreting failures

The rural asphalt lacks sufficiently distinctive accepted correspondences. The junction and repetitive surfaces can yield matches without enough spatially distributed, stable planar support. The small-rig examples add rotation, poles and very limited usable ground. Rejection diagnostics are retained for all donors and all masks; no failed site was dropped after matching.

The geometric cross-validation refits folds of the already selected inlier set. It measures stability, not fully independent validation. The visible-ground photometric audit is also a heuristic gate, not held-out ground truth for the car footprint. Both methods share these gates; this may suppress legitimate but difficult transfers.

The cohort is a geographically varied convenience sample from France, not a random road sample. It does not measure Google Street View camera-generation performance, global coverage, physical ground area, or spherical solid angle. Model training overlap with these places is unknown.

## Next experiment

The next bottleneck is geometry and evidence, not another visual fill. Keep the benchmark frozen and add a separate sequence-aware model that estimates camera motion/road plane jointly from several frames. Use that model to predict visibility, then independently audit alignment at road markings and boundaries. Compare a masked sequence alignment against pairwise homographies on the same sites. Avoid enlarging a trusted region merely because smooth asphalt looks plausible.

For a genuine full-footprint accuracy test, collect or identify a controlled scene with a verified clear reference of the same ground, then evaluate a centrally placed artificial occluder of realistic angular size without leaving a second real vehicle hole in the scene. Calibrate acceptance on development routes and assess it on new routes, reporting coverage, edge alignment and failure rate together. Only after that should “reliable recovery” become a product claim.

## Provenance and reproduction

[Core protocol](../benchmark_v2/protocol.json), [supplemental probe protocol](../benchmark_v2/local_probe_protocol.json), [source selection](../benchmark_v2/selection.json), and [model checksums](../benchmark_v2/model_manifest.json) are saved. Source manifests include all original/prepared image hashes, IDs, metadata, licenses and capture timestamps. Each result records implementation and protocol hashes. See the [README](../README.md) for commands.

Sources: [official LightGlue implementation](https://github.com/cvg/LightGlue/tree/5a650c444718caffc2669dc0094cde908f42dd73), [Panoramax API documentation](https://docs.panoramax.fr/backend/api/api/), and [Licence Ouverte 2.0](https://ia.numerique.gouv.fr/licence-ouverte-open-licence/). Source imagery credit is in every case manifest. The project uses no generative image fill in its proposed repairs.
