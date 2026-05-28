# DenseNet vs ViT (new recipe) vs ViT (baseline recipe) — APPA-REAL

All three runs at **50 epochs** to remove the budget-asymmetry confound.

## Final metrics (test, single seed=42)

| metric | DenseNet | ViT (new) | ViT (baseline) |
|---|---:|---:|---:|
| epochs_run | 50 | 50 | 50 |
| best_epoch_by_val_mae | 31 | 37 | 45 |
| best_val_mae | 4.0983 | 4.1270 | 7.6288 |
| best_val_rmse | 6.3020 | 6.1957 | 10.6993 |
| test_mae | 4.9409 | 4.8607 | 9.4422 |
| test_rmse | 6.9429 | 6.9433 | 13.1345 |
| pred_std | 15.2599 | 15.2739 | 11.5439 |
| pearson_r | 0.9250 | 0.9244 | 0.6956 |
| mean_epoch_time_sec | 9.2574 | 36.1805 | 36.0929 |
| gpu_mem_peak_mb | 4122.5786 | 2990.8071 | 2989.8071 |

## Per-age-group MAE (test)

| age bin | N | DenseNet | ViT (new) | ViT (baseline) | best of {DN, ViT-new} |
|---|---:|---:|---:|---:|:---:|
| [0, 10) | 222 | 3.20 | 2.62 | 8.89 | **ViT (new)** |
| [10, 20) | 147 | 5.29 | 5.57 | 9.66 | **DenseNet** |
| [20, 30) | 579 | 3.65 | 3.76 | 5.21 | **DenseNet** |
| [30, 40) | 465 | 4.48 | 4.23 | 6.61 | **ViT (new)** |
| [40, 50) | 244 | 5.71 | 6.11 | 10.11 | **DenseNet** |
| [50, 60) | 146 | 6.77 | 6.81 | 14.64 | **DenseNet** |
| [60, 70) | 104 | 8.35 | 7.59 | 20.76 | **ViT (new)** |
| [70, inf) | 71 | 11.81 | 11.15 | 34.21 | **ViT (new)** |

## Recipe comparison

- **DenseNet121**: torchvision pretrained, head replaced with `Linear(1024, 1)`,
  Adam lr=1e-4, StepLR (γ=0.5 every 10 ep — triggers at ep 11/21/31/41
  within 50 ep, so lr decays 4 times: 1e-4 -> 6.25e-6 final), weight_decay=0,
  batch_size=32, **50 epochs**, L1 loss.
- **ViT-B/16 (new, this run)**: torchvision pretrained, head replaced with
  `Linear(768, 1)`, AdamW base_lr=2e-4 at head with
  **LLRD (decay 0.75)** down to ~4.75e-6 at embedding, **linear warmup over
  5 epochs**, cosine to η_min=3e-7, weight_decay=0.01, batch_size=16,
  **50 epochs**, L1 loss.
- **ViT-B/16 (baseline)**: same backbone + head as the new run, but the
  default torchvision-style fine-tune recipe: AdamW lr=5e-5 flat across all
  params (no LLRD), cosine to η_min=0 (no warmup), weight_decay=0.05,
  **50 epochs**, L1 loss.

## Headline (at equal 50-epoch budget)

- **DenseNet vs ViT (new) is essentially a tie**:
  MAE 4.861 vs 4.941 (gap -0.080), RMSE essentially
  identical (6.943 each). Within single-seed noise. The earlier "ViT slightly
  wins" framing at 50 vs 25-epoch budget was confounded by the asymmetry.
- **ViT (baseline) at 50 epochs is NOT collapsed**: test MAE
  9.442, Pearson r=0.696 (vs 0.92 for the other two), pred std
  11.54 (vs gt 17.67). Going from 25 to 50 epochs let it escape the
  mean-predictor local optimum (25ep had r=0.286, pred_std=3.09, MAE 13.01)
  — but it remains substantially worse than the new-recipe ViT.
- **Recipe contribution, at equal budget**: switching ViT-B/16 from the
  default recipe to LLRD + warmup (both at 50 ep) drops test MAE from
  9.44 down to 4.86 — an improvement of
  4.58 years (factor ~1.94× lower error).
  See `budget_ablation.csv` for the orthogonal "budget contribution".

## Shared experimental controls (fairness)

All three runs share: seed=42, APPA-REAL official splits
(train=4113, valid=1500, test=1978), the same `AppaRealDataset` with the
same train aug (Resize→RandomCrop→HFlip→ColorJitter→ToTensor→ImageNet-normalize)
and val/test transform, the same L1 loss, the same MAE/RMSE evaluation,
the **same `run_training()` function** in `src/train.py` (no
`if model_name == "vit"` branching), and **the same 50-epoch training budget**.
Per-model differences (optimizer, lr, scheduler, weight_decay, batch_size,
LLRD) are entirely in the YAML configs.
