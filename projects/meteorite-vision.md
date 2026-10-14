---
layout: default
title: Meteorite Vision
---

# Meteorite Vision

**Classification under domain shift and an individual image-generation study**

[Back to portfolio](../) ·
[Source code](https://github.com/0zz0ruby/meteorite-vision-portfolio) ·
[Kaggle Stage 2](https://www.kaggle.com/competitions/meteorite-identification-stage-2)

## Context

This work connects a team Kaggle course competition in meteorite identification
with an individual DCGAN/WGAN-GP generation study.

The classification challenge was distribution shift: training images contained
recurring rulers, color cards, and controlled backgrounds, while test images
varied in lighting, viewpoint, scale, and scene context. Strong random-split
validation did not reliably transfer online.

## My contribution

- Built the ResNet-18 baseline and training-to-submission workflow.
- Developed DINOv2 frozen-feature classification and ablations.
- Collected and processed more than 100 external images.
- Proposed multi-model voting after single-model improvements plateaued.
- Independently conducted the complete DCGAN/WGAN-GP generation study.

The final four-model voting implementation and competition submission were team
work. This page does not attribute every team model or the final ensemble
implementation to individual work.

## Classification experiments

The ResNet-18 baseline reached 0.9852 local F1 but 0.72222 Stage-1 online F1.
The DINOv2 pipeline uses frozen features, concatenated CLS tokens from the last
two blocks, and a linear head. Average-hash grouping with
`StratifiedGroupKFold` reduces near-duplicate leakage; augmentation and
test-time augmentation are evaluated explicitly.

The best individual online F1 was **0.73**. Adding external images reduced
the score to **0.71**, providing negative evidence about transfer under
additional domain mismatch.

## Voting ensemble

The ensemble combines ConvNeXt-Tiny, Swin V2-S, DINOv2 ViT-S/14, and
EfficientNetV2-S. Each model casts a positive, negative, or abstaining vote
using lower and upper probability thresholds; weighted votes determine the
final prediction.

A preserved v2 configuration scored **0.78260** public F1. A separate course
leaderboard records the team's final score as **0.78284**. The final fitting
script and exact final parameters were not retained, so these are separate
evidence records rather than one reproducible configuration.

## Image generation

![Samples from the WGAN-GP meteorite generator](../assets/projects/meteorite-vision/wgan_gp_generated_samples.png)

| Experiment | FID |
|---|---:|
| DCGAN baseline | 245.9168 |
| WGAN-GP v1 | 189.7166 |
| Capacity-balanced v2 | 170.82 |
| SpectralNorm v3 | 137.25 |
| Final saved evaluation | 134.203409 |
| EMA and larger v4 | 400+ |

The final evaluation used 1,000 generated and 2,682 real images. The experiment
history retains failed changes, including the larger EMA model, rather than
only positive runs.

## Reproducibility and attribution

Code, recovered configurations, ablation results, FID records, and evidence
notes are public. Competition data and team checkpoints are not redistributed.
The author's DINOv2 checkpoint is available separately through the repository's
`v1.0-model` release, with a SHA-256 manifest.

[Original team repository](https://github.com/enhhhj/meteorite-identification-stage2/) ·
[Chinese project history](https://github.com/0zz0ruby/meteorite-vision-portfolio/blob/main/docs/project_history_zh.md)
