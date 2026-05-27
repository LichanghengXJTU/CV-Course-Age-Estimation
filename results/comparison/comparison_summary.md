# DenseNet vs ViT — APPA-REAL Age Estimation, 25 epochs, seed=42

## Final metrics

| metric | DenseNet121 | ViT-B/16 |
|---|---:|---:|
| epochs_run | 25 | 25 |
| best_epoch_by_val_mae | 22 | 25 |
| best_val_mae | 4.1589 | 10.6279 |
| best_val_rmse | 6.3876 | 14.2557 |
| final_val_mae | 4.2916 | 10.6279 |
| test_mae | 5.0639 | 13.0091 |
| test_rmse | 7.1905 | 17.6269 |
| n_test | 1978 | 1978 |
| mean_epoch_time_sec | 9.3061 | 35.9340 |
| gpu_mem_peak_mb | 4122.5786 | 2989.8071 |

## Per-age-group MAE on test set

| age bin | N | DenseNet MAE | ViT MAE | ViT / DN |
|---|---:|---:|---:|---:|
| [0, 10) | 222 | 3.03 | 22.32 | 7.37× |
| [10, 20) | 147 | 5.44 | 11.67 | 2.15× |
| [20, 30) | 579 | 3.70 | 3.67 | 0.99× |
| [30, 40) | 465 | 4.64 | 5.52 | 1.19× |
| [40, 50) | 244 | 5.90 | 15.30 | 2.60× |
| [50, 60) | 146 | 6.93 | 24.91 | 3.59× |
| [60, 70) | 104 | 8.50 | 34.65 | 4.08× |
| [70, inf) | 71 | 12.81 | 47.81 | 3.73× |

## Notes

- **Single shared training loop** (`src/train.py:run_training`); per-model
  difference is only optimizer/LR/scheduler/batch_size, all in YAML configs.
- Same augmentation + same ImageNet mean/std + same L1Loss + same eval procedure
  for both models.
- Test set evaluated with each model's `best_val_mae` checkpoint (DenseNet
  ep 22, ViT ep
  25).
- DenseNet wins overall (test MAE
  5.06 vs ViT 13.01),
  driven entirely by the age tails — see `per_age_bar.png`. On the dominant
  [20,30) bin (N=579), ViT slightly edges DenseNet
  (3.67 vs 3.70).
