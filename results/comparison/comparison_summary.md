# DenseNet vs ViT (new recipe) vs ViT (baseline recipe) — APPA-REAL

## Final metrics (test, single seed=42)

| metric | DenseNet | ViT (new) | ViT (baseline) |
|---|---:|---:|---:|
| epochs_run | 25 | 50 | 25 |
| best_epoch_by_val_mae | 22 | 37 | 25 |
| best_val_mae | 4.1589 | 4.1270 | 10.6279 |
| best_val_rmse | 6.3876 | 6.1957 | 14.2557 |
| test_mae | 5.0639 | 4.8607 | 13.0091 |
| test_rmse | 7.1905 | 6.9433 | 17.6269 |
| pred_std | 15.2938 | 15.2739 | 3.0888 |
| pearson_r | 0.9196 | 0.9244 | 0.2856 |
| mean_epoch_time_sec | 9.3061 | 36.1805 | 35.9340 |
| gpu_mem_peak_mb | 4122.5786 | 2990.8071 | 2989.8071 |

## Per-age-group MAE (test)

| age bin | N | DenseNet | ViT (new) | ViT (baseline) | best of {DN, ViT-new} |
|---|---:|---:|---:|---:|:---:|
| [0, 10) | 222 | 3.03 | 2.62 | 22.32 | **ViT (new)** |
| [10, 20) | 147 | 5.44 | 5.57 | 11.67 | **DenseNet** |
| [20, 30) | 579 | 3.70 | 3.76 | 3.67 | **DenseNet** |
| [30, 40) | 465 | 4.64 | 4.23 | 5.52 | **ViT (new)** |
| [40, 50) | 244 | 5.90 | 6.11 | 15.30 | **DenseNet** |
| [50, 60) | 146 | 6.93 | 6.81 | 24.91 | **ViT (new)** |
| [60, 70) | 104 | 8.50 | 7.59 | 34.65 | **ViT (new)** |
| [70, inf) | 71 | 12.81 | 11.15 | 47.81 | **ViT (new)** |

## Recipe comparison

- **DenseNet121**: torchvision pretrained, head replaced with `Linear(1024, 1)`,
  Adam lr=1e-4, StepLR (γ=0.5 every 10 ep), weight_decay=0, batch_size=32,
  25 epochs, L1 loss.
- **ViT-B/16 (new, this run)**: torchvision pretrained, head replaced with
  `Linear(768, 1)`, AdamW base_lr=2e-4 at head with
  **LLRD (decay 0.75)** down to ~4.75e-6 at embedding, **linear warmup over
  5 epochs**, cosine to η_min=3e-7, weight_decay=0.01, batch_size=16,
  **50 epochs**, L1 loss.
- **ViT-B/16 (baseline)**: same backbone + head as the new run, but the
  default torchvision-style fine-tune recipe: AdamW lr=5e-5 flat across all
  params (no LLRD), cosine to 0 (no warmup), weight_decay=0.05, 25 epochs.

## Headline

- **ViT (new) slightly beats DenseNet** on this test set:
  MAE 4.861 vs 5.064 (gap -0.203), and wins in
  5 / 8 age bins including both extreme tails ([0,10) and [70,∞)) that
  DenseNet handles only moderately well.
- **ViT (baseline) collapses**: test MAE 13.009, Pearson r=0.286
  (vs 0.92 for the others), pred std 3.09 vs gt 17.67 — model is essentially
  predicting the training-set mean for all faces.
- **Recipe difference is decisive**: switching from the default
  fine-tune recipe to LLRD + warmup + 50 ep moves ViT's test MAE from
  13.01 down to 4.86, an improvement of
  8.15 years (factor ~2.68× lower error).

## Shared experimental controls (fairness)

Both ViT runs and DenseNet share: seed=42, APPA-REAL official splits
(train=4113, valid=1500, test=1978), the same `AppaRealDataset` with the
same train aug (Resize→RandomCrop→HFlip→ColorJitter→ToTensor→ImageNet-normalize)
and val/test transform, the same L1 loss, the same MAE/RMSE evaluation, and
the **same `run_training()` function** in `src/train.py` (no
`if model_name == "vit"` branching). Per-model differences (optimizer,
lr, scheduler, weight_decay, batch_size, epochs, LLRD) are entirely in the
YAML configs.
