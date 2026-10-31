# Academic context and next experiments

This is a research prototype and benchmark effort, not a claim of inventing
multiview panorama repair. The following approaches motivate future work;
none is silently substituted into the reported baseline.

## Reconstruct shared geometry, then reproject observed pixels

[COLMAP](https://colmap.github.io/tutorial.html) is the classical structure-from-motion
and multiview-stereo route: estimate cameras and scene points, then denser
surfaces. It provides the right geometric abstraction but depends on texture,
overlap and appropriate camera models. PAVER currently fits a shared road plane,
which cannot explain a full street scene.

[VGGT](https://arxiv.org/abs/2503.11651) and
[MapAnything](https://map-anything.github.io/) predict multiview geometry and
cameras. [PanoVGGT](https://arxiv.org/abs/2603.17571), with
[released code](https://github.com/YijingGuo-June/PanoVGGT), is especially relevant:
it processes panoramic images for poses, depth and point clouds. Its PanoCity
dataset is synthetic; training overlap must be excluded from any independent
evaluation. CUDA-oriented dependencies do not imply it runs on this project's
MPS hardware. No checkpoint has been evaluated here.

[PaGeR](https://pager360.github.io/) estimates panoramic depth and normals.
[ZüriPano](https://huggingface.co/datasets/prs-eth/ZuriPano) provides real outdoor
LiDAR-based geometry references; it is not automatically a contiguous donor
network benchmark. [MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/) combines
learned point maps with tracking, fusion and loop closure.
[Argus](https://argus-paper.realsee.ai/) studies sparse panoramic reconstruction
and covisibility in indoor settings; road transfer remains to be established.

The useful experiment is to infer a shared surface map from allowed observations,
project a missing target ray onto it, test which source camera can actually see
that surface, and transfer only adequately sampled observed pixels. Depth/pose
uncertainty, occlusion checks and appearance changes must accompany coverage.

## Scene completion and rendering are different evidence

[Multiview inpainting work from 2016](https://www-sop.inria.fr/reves/Basilic/2016/TSPD16/)
and [Street View privacy protection, CVPR 2019](https://arxiv.org/abs/1903.11532)
are direct prior art for urban multiview repair.
[Street Gaussians](https://arxiv.org/abs/2401.01339) separates static streets and
dynamic objects for novel-view rendering. [GS-RoadPatching](https://arxiv.org/abs/2509.19937)
uses similar Gaussian patches to repair road appearance; a plausible substitute
is not necessarily the previously hidden physical surface.
[RoadGaussian](https://link.springer.com/article/10.1007/s11760-026-05546-5)
uses masks and image inpainting in a Gaussian reconstruction pipeline.
[SPIn-NeRF](https://spinnerf3d.github.io/) and
[Geometry-Aware Diffusion for Multiview Scene Inpainting](https://geomvi.github.io/)
address consistent scene completion. These can be useful when no donor saw the
surface, but synthesized content must remain distinguishable from evidence.

For PAVER, keep two outcomes separate: **observed-surface recovery**, with source
provenance and calibrated acceptance tests, and **plausible completion**, which
cannot be scored as recovery without independent truth. Current experiments
implement the first, restricted to a road plane, and retain abstentions.
