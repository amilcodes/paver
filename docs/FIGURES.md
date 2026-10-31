# README figures

The figures show one Beijing development example from
[PanoCity](https://huggingface.co/datasets/YijingGuo/PanoCity), by Yijing Guo
et al. (2026), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The source revision, five image URLs, checksums, processing and measured values
are recorded in [evidence.json](figures/beijing/evidence.json).

The target is frame 2 of a five-frame window. A synthetic 35° nadir disc is
removed from the target; each donor has its own 25° blocked nadir. GPT-6 Astra
uses masked images, capture order and tool-assisted ground registration.
Clean target pixels, depth and poses are withheld. Its donor-coordinate
submission is committed before scoring. Blinding is procedural on a shared
filesystem, not an OS-enforced boundary.

The figure preserves raw donor colors and abstentions. It is a selected expert
comparator, separate from the automated PAVER LoFTR system. It does not establish
an upper bound or performance on independent routes. The synthetic viewpoints
are elevated; this is not measured recovery under a real car.

## Reading the evidence

- Solid donor links identify the two sources used in the committed output.
  Dashed links are retrieved candidates, unused in that output. Node positions
  encode capture order, not a surveyed map or recovered camera trajectory.
- Fractions integrate the solid angle of the original panorama mask. They
  are not rectangular crop-pixel percentages.
- Geometry passes when the chosen source has depth-based visibility support
  and source-ray error is at most 0.5°. This is a tolerance, not exact alignment.
- The joint check also requires mean absolute RGB-channel error at most
  10 on a 0–255 scale. 62.70% passes both; 36.49% passes geometry but fails RGB;
  0.35% fails geometry/support; 0.45% is unfilled.
- The broader photometric integrity checks remain failed. These are conditional
  scores using provided synthetic geometry, not calibrated real-world accuracy.

The dark donor accounts for 86.79% of aligned RGB failures and averages 17.29
channel levels below the reference. That describes the observed difference;
the physical cause—exposure, lighting or rendering—has not been isolated.

## Reproduce the presentation

```sh
python tools/render_readme_figures.py
```

This uses the four committed crops and evidence JSON to regenerate the PNGs,
diagram and GIF. It does not rerun reconstruction or reproduce the complete
benchmark. Crop hashes and rendered-file hashes are recorded. No font downloads
or network access are needed; the renderer uses Pillow's bundled font.

The GIF contains three discrete assembly states: masked, donor 3, then donors
3 and 1. It is a playback of already committed pixels, not an optimizer trace.
Palette quantization affects only the GIF; PNGs retain RGB. All panels share
the same 90° downward projection, with no color correction or enhancement.

This visual supplement was added after the original research snapshot release.
It does not rewrite that release's ledger or imply the full PanoCity pilot is
included in its archives. The original scientific results remain unchanged.
