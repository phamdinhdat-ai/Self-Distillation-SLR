# SMKD Enhancement Suite — Experiment Guide

## Quick Start

```bash
# Smoke test (synthetic data, 6 epochs, ~30 seconds)
python experiments/runner.py --config configs/experiments/baseline.yaml --epochs 6

# Full run (synthetic data, 100 epochs)
python experiments/runner.py --config configs/experiments/baseline.yaml

# Real data run
python experiments/runner.py --config configs/experiments/baseline.yaml --real --data_root /path/to/phoenix2014T

# Override specific settings
python experiments/runner.py --config configs/experiments/baseline.yaml --override loss.alpha=0.3 optimization.lr=0.001
```

## Compare Results

```bash
# Compare all runs
python experiments/compare.py

# Compare specific runs
python experiments/compare.py runs/baseline/20250101_120000 runs/loss_vac/20250101_130000

# Save comparison plot
python experiments/compare.py --output comparison.png
```

## Config Schema

| Section | Key | Default | Description |
|---|---|---|---|
| **data** | `synthetic` | `true` | Use synthetic random data (no dataset needed) |
| | `img_size` | `224` | Frame crop size (H×W) |
| | `max_frames` | `300` | Max frames per sequence |
| | `batch_size` | `2` | Per-GPU batch size |
| | `temporal_aug.enabled` | `false` | Enable temporal augmentation |
| | `temporal_aug.temporal_mask_prob` | `0.3` | Probability of frame masking |
| | `temporal_aug.temporal_mask_max_ratio` | `0.15` | Max fraction of frames to zero out |
| | `temporal_aug.temporal_jitter_range` | `0.0` | ± rate variation (e.g. 0.2 = ±20%) |
| **model** | `d_model` | `512` | Feature dimension |
| | `hidden_size` | `512` | BiLSTM hidden size |
| | `backbone` | `resnet18` | One of: resnet18, mobilenet_v3_small, efficientnet_b0 |
| | `temporal_conv_type` | `standard` | standard or depthwise_separable |
| | `multi_scale_temporal` | `false` | Enable multi-scale temporal branches |
| | `multi_scale_dilation_rates` | `[1,2]` | Dilation rates for extra branches |
| | `use_tsm` | `false` | Temporal Shift Module (ResNet-18 only) |
| **loss** | `alpha` | `0.5` | Visual loss weight |
| | `label_smoothing` | `0.2` | GSBA CE label smoothing |
| | `alpha_schedule` | `fixed` | fixed, linear, or cosine |
| | `alpha_start` / `alpha_end` | `0.1` / `0.5` | Curriculum range |
| | `feature_alignment.enabled` | `false` | VAC-style ||g_c − v_c|| penalty |
| | `feature_alignment.weight` | `0.1` | Alignment loss weight |
| | `feature_alignment.mode` | `cosine` | cosine or l2 |
| | `entropy_regularization.enabled` | `false` | Maximize blank entropy (anti-spike) |
| | `entropy_regularization.weight` | `0.01` | Entropy reg weight |
| | `focal_ctc.enabled` | `false` | Focal CTC modulation |
| | `focal_ctc.gamma` | `2.0` | Focal exponent |
| | `gsba_confidence_weight` | `false` | Weight CE-LS by GSBA confidence margin |
| **schedule** | `epochs` | `100` | Total epochs |
| | `stage1_end` | `30` | Last epoch of stage 1 |
| | `stage2_end` | `90` | Last epoch of stage 2 |
| | `eval_every` | `5` | Evaluate WER every N epochs |
| | `smooth_transitions` | `false` | Linear ramp at stage boundaries |
| | `smooth_transition_epochs` | `5` | Number of transition epochs |
| | `self_distill.enabled` | `false` | KD from pre-trained teacher |
| | `self_distill.teacher_ckpt` | `""` | Path to teacher checkpoint |
| | `self_distill.kd_temperature` | `4.0` | KD temperature |
| | `self_distill.kd_weight` | `0.5` | KD loss weight |
| **optimization** | `lr` | `1e-4` | Learning rate |
| | `lr_milestones` | `[40,60,80]` | LR decay epochs |
| | `lr_gamma` | `0.5` | LR decay factor |
| | `use_amp` | `false` | Automatic mixed precision |
| | `grad_accum_steps` | `1` | Gradient accumulation steps |
| | `use_ema` | `false` | Exponential moving average |
| | `ema_decay` | `0.999` | EMA decay rate |
| **hardware** | `multi_gpu` | `true` | Use DataParallel if multiple GPUs |
| **output** | `experiment_name` | `default` | Name for run directory |
| | `ckpt_dir` | `./checkpoints` | Checkpoint save directory |

## Experiment Presets

| Config File | What It Tests |
|---|---|
| `baseline.yaml` | Paper reproduction (no enhancements) |
| `loss_vac.yaml` | VAC-style visual alignment constraint |
| `loss_entropy.yaml` | Entropy regularization (anti-spike) |
| `loss_vac_entropy.yaml` | VAC + entropy combined |
| `loss_confidence_gsba.yaml` | Confidence-weighted GSBA |
| `loss_curriculum_alpha.yaml` | Curriculum schedule on α |
| `loss_focal_ctc.yaml` | Focal CTC modulation |
| `model_dwconv.yaml` | Depthwise-separable temporal convs |
| `model_multiscale.yaml` | Multi-scale temporal branches |
| `model_mobilenet.yaml` | MobileNetV3-Small backbone |
| `model_tsm.yaml` | Temporal Shift Module (TSM) |
| `train_amp.yaml` | AMP + gradient accumulation |
| `train_ema.yaml` | Exponential moving average |
| `train_temporal_aug.yaml` | Temporal data augmentation |
| `train_smooth_transition.yaml` | Smooth stage transitions |
| `combined_tier1.yaml` | All Tier 1 enhancements together |
| `combined_tier2.yaml` | All Tier 1 + Tier 2 together |

## Typical Experiment Workflow

1. **Smoke test the baseline**: `python experiments/runner.py --config configs/experiments/baseline.yaml --epochs 6`
2. **Run baseline for comparison**: `python experiments/runner.py --config configs/experiments/baseline.yaml`
3. **Run a few ablations**: e.g. VAC, entropy, AMP, temporal aug
4. **Compare**: `python experiments/compare.py`
5. **Run combined best config**: `python experiments/runner.py --config configs/experiments/combined_tier2.yaml`
6. **Compare all iterations**: `python experiments/compare.py --output final_comparison.png`

## Output Structure

```
runs/
  <experiment_name>/
    <YYYYMMDD_HHMMSS>/
      config.yaml          # Raw YAML config
      flat_config.yaml     # Flattened config (what Trainer received)
      history.csv          # Per-epoch loss/WER values
      wer_history.png      # WER + loss plots
      checkpoints/
        smkd_best_ep*.pt   # Best WER checkpoint(s)
```
