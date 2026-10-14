---
layout: default
title: DataAgent-SFT
---

# DataAgent-SFT

**Data curation, supervised fine-tuning, and a downstream analytical application**

[Back to portfolio](../) ·
[Source code and reproduction](https://github.com/0zz0ruby/data-agent-sft)

## Objective

Adapt a compact Qwen model to produce structured data-analysis responses from
curated analytical trajectories. The project covers selection, assistant-token
supervision, training and validation, checkpoint inference, and deployment.
CrossMetric AI applies this fine-tuned Data Agent in a commerce-analysis
prototype.

## Data curation

Processed 11,707 DataMind trajectories using quality scores combining
complexity, syntax, execution-consistency, structure, and text signals.
Execution-consistency signals come from existing traces, rather than
re-executing every program.

| Stage | Trajectories |
|---|---:|
| Input | 11,707 |
| After exact deduplication | 11,500 |
| After TF-IDF near-deduplication | 11,301 |
| Selected with ranking and diversity constraints | 2,500 |
| Train / validation | 2,000 / 500 |

The split has no recorded group, ID, or message-hash overlap. Selection uses
deterministic quality scoring, not a trained reward model or reinforcement
learning. Candidates not selected were not necessarily invalid.

## Fine-tuning

Fine-tuned the Qwen3.5-0.8B text backbone with the unused vision tower frozen.
This is neither LoRA nor an update of all multimodal parameters.
Each assistant turn becomes an example; system, user, and context tokens are
masked from loss. Ray TorchTrainer supports gradient accumulation,
checkpointing, and validation-scored checkpoint retention.

| Setting / result | Value |
|---|---:|
| Epochs / maximum sequence length | 1 / 2,048 |
| Learning rate / seed | 2e-5 / 42 |
| Per-device batch / accumulation | 1 / 8 |
| Optimizer steps | 1,475 |
| Baseline validation loss | 0.731607 |
| Final validation loss | 0.488760 |
| Final training loss | 0.504752 |

Held-out teacher-forced token loss decreased by about 33.2%. This is not a
measured task-success rate, code-runnability score, or business-performance
evaluation. Results come from the archived server run.

## CrossMetric AI

The bilingual commerce prototype combines deterministic Python calculations
with guarded model explanations, three decision workflows, and Monte Carlo
scenarios. Unsupported narratives fall back to deterministic text; fallback is
not counted as successful model reasoning. Historical transaction revenue is
observed, while operational cost, channel, and inventory fields are simulated.

## Demonstration and limitations

The model generates analysis plans and Python code. The current demonstration
does not execute generated code or validate its results.

Observed failures include invented metrics, contradictory financial reasoning,
and repetition. Guardrails are heuristic. Base-versus-fine-tuned generated-answer
evaluation remains future work, and architecture capacity targets are not
measured throughput.

## Reproducibility

The repository includes selection and training scripts, recorded metrics,
executable checks, a streaming demo, and reproduction notes. The recorded
training run used one RTX 4090 D. Raw datasets, weights, credentials, and
personal server paths are not committed; upstream terms apply separately.
Original code has no selected open-source license yet.

[Reproducibility report](https://github.com/0zz0ruby/data-agent-sft/blob/main/docs/REPRODUCIBILITY.md) ·
[Application README](https://github.com/0zz0ruby/data-agent-sft/blob/main/applications/crossmetric/README.md)
