"""Compare DenseNet vs ViT formal-run results.

Reads:
  results/{densenet,vit}/{epoch_log.csv, per_age_group_mae.csv,
                          test_predictions.npy, test_ground_truth.npy,
                          test_summary.txt}

Writes (to results/comparison/):
  - mae_curves_overlay.png    train+val MAE per epoch, both models
  - loss_curves_overlay.png   train+val loss per epoch, both models
  - scatter_side_by_side.png  pred vs true on test set
  - per_age_bar.png           per-age-group MAE, grouped bars
  - comparison_table.csv      key metrics side-by-side
  - comparison_summary.md     human-readable summary
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = RES / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["densenet", "vit"]
COLORS = {"densenet": "#1f77b4", "vit": "#ff7f0e"}
LABELS = {"densenet": "DenseNet121", "vit": "ViT-B/16"}


def load_all():
    data = {}
    for m in MODELS:
        d = RES / m
        data[m] = {
            "epoch_log": pd.read_csv(d / "epoch_log.csv"),
            "per_age": pd.read_csv(d / "per_age_group_mae.csv"),
            "pred": np.load(d / "test_predictions.npy"),
            "gt": np.load(d / "test_ground_truth.npy"),
            "summary_raw": (d / "test_summary.txt").read_text(),
        }
        # Parse simple "key: value" lines from test_summary.txt
        s = {}
        for line in data[m]["summary_raw"].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                s[k.strip()] = v.strip()
        data[m]["summary"] = s
    return data


def plot_mae_curves(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MODELS:
        df = data[m]["epoch_log"]
        ax.plot(df["epoch"], df["val_mae"], "-",
                color=COLORS[m], label=f"{LABELS[m]} val MAE", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val MAE (years)")
    ax.set_title("Validation MAE — DenseNet vs ViT (APPA-REAL, 25 epochs, seed=42)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mae_curves_overlay.png", dpi=120)
    plt.close(fig)


def plot_loss_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, kind in zip(axes, ["train_loss_avg", "val_loss"]):
        for m in MODELS:
            df = data[m]["epoch_log"]
            ax.plot(df["epoch"], df[kind], "-",
                    color=COLORS[m], label=LABELS[m], linewidth=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(kind.replace("_", " "))
        ax.set_title(kind.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Loss curves — DenseNet vs ViT")
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves_overlay.png", dpi=120)
    plt.close(fig)


def plot_scatter(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, m in zip(axes, MODELS):
        pred = data[m]["pred"]
        gt = data[m]["gt"]
        mae = float(np.mean(np.abs(pred - gt)))
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        ax.scatter(gt, pred, s=6, alpha=0.35, color=COLORS[m])
        # y = x reference
        lo, hi = 0, max(gt.max(), pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7)
        ax.set_xlabel("true age")
        ax.set_ylabel("predicted age")
        ax.set_title(f"{LABELS[m]}\nMAE={mae:.2f}  RMSE={rmse:.2f}  N={len(pred)}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Test-set prediction vs ground truth (y=x is perfect)")
    fig.tight_layout()
    fig.savefig(OUT / "scatter_side_by_side.png", dpi=120)
    plt.close(fig)


def plot_per_age(data):
    dn = data["densenet"]["per_age"]
    vit = data["vit"]["per_age"]
    assert (dn["bin"].values == vit["bin"].values).all(), "bin mismatch"
    bins = dn["bin"].values
    counts = dn["count"].values
    x = np.arange(len(bins))
    width = 0.4

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.bar(x - width / 2, dn["mae"].values, width,
           color=COLORS["densenet"], label=LABELS["densenet"])
    ax.bar(x + width / 2, vit["mae"].values, width,
           color=COLORS["vit"], label=LABELS["vit"])
    ax.set_ylabel("MAE (years)")
    ax.set_title("Per-age-group test MAE — DenseNet vs ViT")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    ax2.bar(x, counts, color="grey", alpha=0.7)
    ax2.set_ylabel("# test samples")
    ax2.set_xlabel("age group")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bins, rotation=20)
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "per_age_bar.png", dpi=120)
    plt.close(fig)


def write_table(data):
    rows = []
    for m in MODELS:
        df = data[m]["epoch_log"]
        s = data[m]["summary"]
        best_idx = df["val_mae"].idxmin()
        rows.append({
            "model": LABELS[m],
            "epochs_run": int(df["epoch"].max()),
            "best_epoch_by_val_mae": int(df.loc[best_idx, "epoch"]),
            "best_val_mae": float(df.loc[best_idx, "val_mae"]),
            "best_val_rmse": float(df.loc[best_idx, "val_rmse"]),
            "final_val_mae": float(df["val_mae"].iloc[-1]),
            "test_mae": float(s["test_mae"]),
            "test_rmse": float(s["test_rmse"]),
            "n_test": int(s["n_test"]),
            "mean_epoch_time_sec": float(df["epoch_time_sec"].mean()),
            "gpu_mem_peak_mb": float(df["gpu_mem_peak_mb"].max()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "comparison_table.csv", index=False)
    return out


def write_markdown_summary(data, table_df):
    dn_age = data["densenet"]["per_age"].set_index("bin")["mae"]
    vit_age = data["vit"]["per_age"].set_index("bin")["mae"]
    counts = data["densenet"]["per_age"].set_index("bin")["count"]
    age_lines = ["| age bin | N | DenseNet MAE | ViT MAE | ViT / DN |",
                 "|---|---:|---:|---:|---:|"]
    for b in dn_age.index:
        ratio = vit_age[b] / dn_age[b]
        age_lines.append(
            f"| {b} | {int(counts[b])} | {dn_age[b]:.2f} | {vit_age[b]:.2f} | {ratio:.2f}× |"
        )

    table_lines = ["| metric | DenseNet121 | ViT-B/16 |",
                   "|---|---:|---:|"]
    for col in ["epochs_run", "best_epoch_by_val_mae", "best_val_mae",
                "best_val_rmse", "final_val_mae", "test_mae", "test_rmse",
                "n_test", "mean_epoch_time_sec", "gpu_mem_peak_mb"]:
        v_dn = table_df.iloc[0][col]
        v_vit = table_df.iloc[1][col]
        fmt = (lambda x: f"{x:.4f}") if isinstance(v_dn, float) else str
        table_lines.append(f"| {col} | {fmt(v_dn)} | {fmt(v_vit)} |")

    md = f"""# DenseNet vs ViT — APPA-REAL Age Estimation, 25 epochs, seed=42

## Final metrics

{chr(10).join(table_lines)}

## Per-age-group MAE on test set

{chr(10).join(age_lines)}

## Notes

- **Single shared training loop** (`src/train.py:run_training`); per-model
  difference is only optimizer/LR/scheduler/batch_size, all in YAML configs.
- Same augmentation + same ImageNet mean/std + same L1Loss + same eval procedure
  for both models.
- Test set evaluated with each model's `best_val_mae` checkpoint (DenseNet
  ep {int(table_df.iloc[0]['best_epoch_by_val_mae'])}, ViT ep
  {int(table_df.iloc[1]['best_epoch_by_val_mae'])}).
- DenseNet wins overall (test MAE
  {table_df.iloc[0]['test_mae']:.2f} vs ViT {table_df.iloc[1]['test_mae']:.2f}),
  driven entirely by the age tails — see `per_age_bar.png`. On the dominant
  [20,30) bin (N={int(counts['[20, 30)'])}), ViT slightly edges DenseNet
  ({vit_age['[20, 30)']:.2f} vs {dn_age['[20, 30)']:.2f}).
"""
    (OUT / "comparison_summary.md").write_text(md)


def main():
    data = load_all()
    plot_mae_curves(data)
    plot_loss_curves(data)
    plot_scatter(data)
    plot_per_age(data)
    table = write_table(data)
    write_markdown_summary(data, table)
    print("=== comparison_table.csv ===")
    print(table.to_string(index=False))
    print()
    print(f"All artifacts saved under {OUT}")


if __name__ == "__main__":
    main()
