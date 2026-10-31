# Research direction and evidence ledger

Project restart: 31 October 2026. The user supplied prior planning and a long coverage summary as background, explicitly warning that it may be inaccurate. The underlying CSV, tally scripts and report files were not supplied. We have not reproduced their global percentages.

## The question worth testing first

For each masked region in a panorama, how much of the surface is visible in nearby observations, and when can we reliably transfer those observations without creating false scene content?

Start with camera-vehicle footprints on approximately planar roads, using neighboring panoramas from the same pass. This scope gives a cheap geometric baseline, avoids cross-year scene changes, and targets the actual visibility problem independently of camera-generation labels. Keep source-backed transfers, geometry-dependent estimates and plausible generated content distinct in both data and presentation.

This is a useful research direction, not a claim of novelty. Closely related methods already exist.

## Corrections to the inherited reasoning

1. **One capture date is not one spatial observation.** The pasted report treats some single-pass countries as having no reference imagery. That does not follow. Adjacent panoramas from one drive provide spatially distinct observations. Our six-image sample demonstrates the distinction directly: two neighborhoods, one capture day, neighboring cameras displaced along the road. Historical passes and same-pass neighbors must be separate axes in the dataset.

2. **Latest listed date is not a measured default selection.** Taking the maximum valid date is an appropriate description of the newest listed capture in a supplied history. Without recording the panorama actually returned by a specified viewer/query at a specified time, it does not independently establish what Google selects by default. Date histories at a location may also refer to nearby camera positions rather than exactly co-located captures.

3. **Country road-length reweighting does not fix the within-country sample.** If points come from an area sample snapped to a panorama network, multiplying country means by national road totals leaves the within-country inclusion bias intact. Total road length is not measured covered road length. An assumed coverage fraction produces a model-based scenario, not a design-based global estimate. Excluding India or the US is a sensitivity analysis; it is not a statistical upper bound or confidence interval.

4. **Road nodes are not uniform road-length samples.** Sampling OSM vertices overweights curvy or densely digitized segments. A defensible road-based survey would first define the road universe, split it into suitable segments, sample segments proportional to length (or sample cumulative arclength directly), then sample uniformly along the chosen segments. It must handle repeated snaps, unmatched roads, the provider's search radius, historical availability and inclusion probabilities. That is a separate measurement project.

5. **Date and image size provide noisy camera-class proxies.** Their overlap is already acknowledged in the supplied background. Do not promote the heuristic to ground-truth hardware identification. Retain an unknown class, record capture and release dates separately, and measure defect geometry directly. Actual repair inputs should include the masked angular area, neighbor baseline, usable ground visibility, pose uncertainty, seam location, effective resolution and dynamic occlusions.

6. **Missing histories do not establish that no newer drive exists.** Provider selection, deleted history, changing metadata, sampling and update lag all complicate that inference. Similarly, a stale generation-presence map cannot estimate a current generation share.

7. **A lower camera can hide more usable ground even with sharper sensors.** A simple flat-ground model clarifies the tradeoff. If camera height is h and the unavailable nadir cone has half-angle α from straight down, its footprint radius is h·tan(α). For a point directly under the target camera, a source camera at horizontal separation b sees it at depression angle atan(h/b). It must be outside the source's own nadir cone and within its usable lower field of view. Too little baseline retains the source vehicle occlusion; too much makes the ground oblique, low-resolution or out of the chosen crop. This is a geometric derivation under stated assumptions, not a claim about any named Google rig.

The current [Emily coverage-date page](https://geo.emily.bz/coverage-dates) was checked. It documents downloadable country maps, approximate update cadence, boundary issues, placeholder dates and sparse newly covered areas. It does not by itself verify the supplied 7.32-million tally or its road-weighted conclusions. Those quantities remain background claims pending a separate reproducible audit.

## Closest verified prior art

| Work | What it already establishes | Relevance to this project |
|---|---|---|
| [Thonat et al., Multi-View Inpainting for Image-Based Scene Editing and Rendering, 3DV 2016](https://www-sop.inria.fr/reves/Basilic/2016/TSPD16/) | Combines multi-view reconstruction, neighboring-view reprojection and patch-based inpainting for urban object removal; includes Google Street View examples. | Directly overlaps the initial idea. Reprojection from neighbors is not new. |
| [Uittenbogaard et al., Privacy Protection in Street-View Panoramas using Depth and Multi-View Imagery, CVPR 2019](https://arxiv.org/abs/1903.11532) | Uses depth consistency and segmentation to identify moving objects, then fills from other street-view images. | A strong task-specific reference for the later vehicle/occluder branch. Its reported numbers are not comparable with our tiny planar experiment. |
| [Mirzaei et al., SPIn-NeRF, CVPR 2023](https://arxiv.org/abs/2211.12254) | Multi-view-consistent 3D inpainting with NeRFs; includes scenes with and without target objects for evaluation. | Useful benchmark design and advanced baseline after registration/visibility are established. |
| [Yu et al., Panoramic Image Inpainting With Gated Convolution And Contextual Reconstruction Loss, ICASSP 2024](https://arxiv.org/abs/2402.02936) | A panorama-specific learned image-inpainting method evaluated on SUN360. | A possible single-image comparison, not evidence that the filled scene is observed. |

The primary sources above were checked; this is a targeted literature pass, not an exhaustive 2026 survey. A defensible contribution might combine a recoverability benchmark, explicit abstention and calibrated error estimates for sparse degraded panoramas. Whether that combination is novel requires further comparison with uncertainty-aware 3D inpainting and reconstruction work.

## Data decision

Use an accessible, licensed source for initial experiments. [Panoramax documents a STAC-style API](https://docs.panoramax.fr/backend/api/api/) with collections, images and metadata; its [compatibility documentation](https://docs.panoramax.fr/backend/dev/STAC_compatibility/) describes previous/next links. The six selected images' returned metadata explicitly lists the Strasbourg producer, capture time and `etalab-2.0` license. The [official licence page](https://ia.numerique.gouv.fr/licence-ouverte-open-licence/) describes adaptation and reuse with attribution.

The standard [Google Map Tiles API policy](https://developers.google.com/maps/documentation/tile/policies) expressly lists image analysis and machine interpretation among non-visualization uses that are not permitted under that API's standard policy, and restricts storage subject to the applicable agreement. This is a concrete reason not to make that API the default acquisition path for this prototype. A separately licensed provider dataset or user-owned capture can be added later. This note concerns that particular documented API policy, not a blanket conclusion about every Google agreement or research arrangement.

No unsupported private endpoints, mass metadata collection or unblurring were used. Public Panoramax images retain their existing privacy processing. Image metadata is treated as acquisition evidence, not calibrated six-degree-of-freedom camera pose.

## What the experiment established

The current pipeline is a computational baseline, not a trained model. Road-plane registration can recover observed markings from neighboring frames. In the marked-road neighborhood, widening the downward view allowed more ground to participate and increased proposed transfer coverage. The original repetitive-paving neighborhood supplied too few distinctive matches and was rejected. All results remain in the repository.

The wider-view experiment changes the view extent and feature population at fixed focal length in pixels. We inspected the first results before choosing this variant. The four hidden-patch measurements therefore remain development observations. Their full-mask errors favor the complete Telea baseline, while accepted-pixel errors favor transferred observations on two marking patches. Reporting only those two wins would overstate the result.

The actual vehicle experiment has no clean observation of the target from the same camera position, so it provides no factual accuracy score beneath the vehicle. Its repair mask conservatively covers the car, logo and some surrounding visible ground. The 55.6% number measures algorithmically accepted pixels in that mask. All have only one usable donor; the experiment does not yet validate multi-view agreement or calibrate uncertainty.

## Next experiment and decision gates

1. **Freeze a small benchmark before additional tuning.** Collect roughly 30 target neighborhoods from at least five distinct routes and several surface types. Use permissioned/open data. Keep route-level splits; nearby frames must not leak between development and test. Treat this size as a proposed pilot, not statistical sufficiency.
2. **Build observable ground truth.** Mask visible road patches with realistic projected footprint shapes; stratify by angular size, surface, baseline and exposure. Add a small controlled capture with the rig removed or repositioned to assess actual footprint reconstruction. Record moving shadows and changes instead of silently treating them as registration error.
3. **Compare correspondence methods at fixed inputs.** SIFT/RANSAC versus a stronger pretrained matcher, plus GPS/heading priors where available. Do not relax geometric acceptance solely to raise coverage. Evaluate residual alignment on held-out correspondences and on the actual transfer region.
4. **Measure both coverage and wrong transfers.** Report accepted-region MAE/PSNR, full-mask metrics, boundary error, line-structure preservation and a risk-versus-coverage curve. Keep single-donor and corroborated pixels separate. Calibrate any confidence score on development data and test it on held-out routes.
5. **Add depth and visibility before general object removal.** A planar homography cannot distinguish a pedestrian from the road behind them. Require source masks, reprojection/depth consistency, visibility tests and a recorded reason for rejection. Only then consider a generative finishing branch, clearly labeled as inferred content.

Proceed to a larger system when transfers remain accurate on unseen routes at useful coverage and failure gates reject bad matches. If coverage stays low despite correct geometry, the relevant work may be source selection, capture spacing or a hybrid observed/inferred output rather than another image model.
