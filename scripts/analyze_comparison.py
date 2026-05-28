"""Compare DenseNet vs ViT (LLRD+warmup) vs ViT baseline (default recipe).

Reads:
  results/{densenet, vit, vit_baseline}/{
      epoch_log.csv, per_age_group_mae.csv,
      test_predictions.npy, test_ground_truth.npy,
      test_summary.txt,
  }

Writes (to results/comparison/):
  - mae_curves_overlay.png       val MAE per epoch, 3 models overlaid
  - loss_curves_overlay.png      train + val loss per epoch, 3 models
  - scatter_three_panel.png      pred vs true, one panel per model
  - per_age_bar.png              per-age-group MAE, grouped bars
  - pred_distribution.png        prediction histogram per model vs ground truth
  - comparison_table.csv         3-row key-metrics table
  - comparison_summary.md        human-readable summary

The primary comparison is DenseNet vs ViT (the LLRD+warmup recipe).
vit_baseline is the original default recipe — kept for ablation, to make the
"recipe matters" story explicit and reproducible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = RES / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

# Order matters for plotting layers / table rows.
MODELS = ["densenet", "vit", "vit_baseline"]
COLORS = {
    "densenet": "#1f77b4",
    "vit": "#2ca02c",
    "vit_baseline": "#d62728",
}
LABELS = {
    "densenet": "DenseNet121 (50ep)",
    "vit": "ViT-B/16 (LLRD+warmup, 50ep)",
    "vit_baseline": "ViT-B/16 (default recipe, 50ep)",
}


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
        s = {}
        for line in data[m]["summary_raw"].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                s[k.strip()] = v.strip()
        data[m]["summary"] = s
    return data


def plot_mae_curves(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in MODELS:
        df = data[m]["epoch_log"]
        ax.plot(df["epoch"], df["val_mae"], "-",
                color=COLORS[m], label=LABELS[m], linewidth=2)
    # mark each model's best epoch
    for m in MODELS:
        df = data[m]["epoch_log"]
        idx = df["val_mae"].idxmin()
        ax.scatter([df.loc[idx, "epoch"]], [df.loc[idx, "val_mae"]],
                   color=COLORS[m], s=80, zorder=5, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val MAE (years)")
    ax.set_title("Validation MAE — DenseNet vs ViT (LLRD+warmup) vs ViT (baseline)\n"
                 "(dots mark each model's best epoch, used for test eval)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mae_curves_overlay.png", dpi=120)
    plt.close(fig)


def plot_loss_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, kind in zip(axes, ["train_loss_avg", "val_loss"]):
        for m in MODELS:
            df = data[m]["epoch_log"]
            ax.plot(df["epoch"], df[kind], "-",
                    color=COLORS[m], label=LABELS[m], linewidth=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(kind.replace("_", " "))
        ax.set_title(kind.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Loss curves — DenseNet vs ViT (new) vs ViT (baseline)")
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves_overlay.png", dpi=120)
    plt.close(fig)


def plot_scatter(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, m in zip(axes, MODELS):
        pred = data[m]["pred"]
        gt = data[m]["gt"]
        mae = float(np.mean(np.abs(pred - gt)))
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        ax.scatter(gt, pred, s=6, alpha=0.35, color=COLORS[m])
        lo, hi = 0, 100
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7)
        ax.set_xlabel("true age")
        ax.set_ylabel("predicted age")
        ax.set_title(f"{LABELS[m]}\nMAE={mae:.2f}  RMSE={rmse:.2f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Test-set prediction vs ground truth (y=x is perfect)")
    fig.tight_layout()
    fig.savefig(OUT / "scatter_three_panel.png", dpi=120)
    plt.close(fig)


def plot_per_age(data):
    bins_ref = data["densenet"]["per_age"]["bin"].values
    counts = data["densenet"]["per_age"]["count"].values
    x = np.arange(len(bins_ref))
    width = 0.27

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    for i, m in enumerate(MODELS):
        df = data[m]["per_age"]
        assert (df["bin"].values == bins_ref).all(), f"bin mismatch in {m}"
        offset = (i - 1) * width  # -width, 0, +width
        ax.bar(x + offset, df["mae"].values, width,
               color=COLORS[m], label=LABELS[m])
    ax.set_ylabel("MAE (years)")
    ax.set_title("Per-age-group test MAE — DenseNet vs ViT (new) vs ViT (baseline)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax2.bar(x, counts, color="grey", alpha=0.7)
    ax2.set_ylabel("# test samples")
    ax2.set_xlabel("age group")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bins_ref, rotation=20)
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "per_age_bar.png", dpi=120)
    plt.close(fig)


def plot_pred_distribution(data):
    """Histogram of predictions vs ground truth. Makes the v1 collapse
    visually obvious."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    gt = data["densenet"]["gt"]  # same gt across models
    bins = np.arange(0, 101, 5)
    for ax, m in zip(axes, MODELS):
        ax.hist(gt, bins=bins, alpha=0.45, color="grey",
                label="ground truth", edgecolor="black", linewidth=0.4)
        ax.hist(data[m]["pred"], bins=bins, alpha=0.7, color=COLORS[m],
                label=f"{LABELS[m]} prediction", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("age")
        ax.set_title(LABELS[m] + f"\npred std={data[m]['pred'].std():.2f}, "
                     f"r={np.corrcoef(data[m]['pred'], gt)[0,1]:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("# test samples")
    fig.suptitle("Prediction distribution vs ground truth — "
                 "narrow histogram = mean-collapse")
    fig.tight_layout()
    fig.savefig(OUT / "pred_distribution.png", dpi=120)
    plt.close(fig)


def write_table(data):
    rows = []
    for m in MODELS:
        df = data[m]["epoch_log"]
        s = data[m]["summary"]
        pred = data[m]["pred"]
        gt = data[m]["gt"]
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
            "pred_std": float(pred.std()),
            "pearson_r": float(np.corrcoef(pred, gt)[0, 1]),
            "mean_epoch_time_sec": float(df["epoch_time_sec"].mean()),
            "gpu_mem_peak_mb": float(df["gpu_mem_peak_mb"].max()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "comparison_table.csv", index=False)
    return out


def write_markdown_summary(data, table_df):
    bins_ref = data["densenet"]["per_age"]["bin"].values
    counts = data["densenet"]["per_age"]["count"].values

    age_lines = ["| age bin | N | DenseNet | ViT (new) | ViT (baseline) | best of {DN, ViT-new} |",
                 "|---|---:|---:|---:|---:|:---:|"]
    for i, b in enumerate(bins_ref):
        dn = data["densenet"]["per_age"].iloc[i]["mae"]
        vit = data["vit"]["per_age"].iloc[i]["mae"]
        base = data["vit_baseline"]["per_age"].iloc[i]["mae"]
        winner = "**ViT (new)**" if vit < dn else "**DenseNet**"
        age_lines.append(f"| {b} | {int(counts[i])} | {dn:.2f} | {vit:.2f} | {base:.2f} | {winner} |")

    # Build main table from columns we care about
    main_cols = ["epochs_run", "best_epoch_by_val_mae", "best_val_mae",
                 "best_val_rmse", "test_mae", "test_rmse", "pred_std",
                 "pearson_r", "mean_epoch_time_sec", "gpu_mem_peak_mb"]
    table_lines = ["| metric | DenseNet | ViT (new) | ViT (baseline) |",
                   "|---|---:|---:|---:|"]
    for col in main_cols:
        v_dn = table_df.iloc[0][col]
        v_vit = table_df.iloc[1][col]
        v_base = table_df.iloc[2][col]
        if isinstance(v_dn, float):
            line = f"| {col} | {v_dn:.4f} | {v_vit:.4f} | {v_base:.4f} |"
        else:
            line = f"| {col} | {v_dn} | {v_vit} | {v_base} |"
        table_lines.append(line)

    # Quick "story" interpretation
    vit_test = table_df.iloc[1]["test_mae"]
    dn_test = table_df.iloc[0]["test_mae"]
    base_test = table_df.iloc[2]["test_mae"]
    gap_vs_dn = vit_test - dn_test
    gap_vs_base = base_test - vit_test
    md = f"""# DenseNet vs ViT (new recipe) vs ViT (baseline recipe) — APPA-REAL

All three runs at **50 epochs** to remove the budget-asymmetry confound.

## Final metrics (test, single seed=42)

{chr(10).join(table_lines)}

## Per-age-group MAE (test)

{chr(10).join(age_lines)}

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
  MAE {vit_test:.3f} vs {dn_test:.3f} (gap {gap_vs_dn:+.3f}), RMSE essentially
  identical (6.943 each). Within single-seed noise. The earlier "ViT slightly
  wins" framing at 50 vs 25-epoch budget was confounded by the asymmetry.
- **ViT (baseline) at 50 epochs is NOT collapsed**: test MAE
  {base_test:.3f}, Pearson r=0.696 (vs 0.92 for the other two), pred std
  11.54 (vs gt 17.67). Going from 25 to 50 epochs let it escape the
  mean-predictor local optimum (25ep had r=0.286, pred_std=3.09, MAE 13.01)
  — but it remains substantially worse than the new-recipe ViT.
- **Recipe contribution, at equal budget**: switching ViT-B/16 from the
  default recipe to LLRD + warmup (both at 50 ep) drops test MAE from
  {base_test:.2f} down to {vit_test:.2f} — an improvement of
  {gap_vs_base:.2f} years (factor ~{base_test/vit_test:.2f}× lower error).
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
"""
    (OUT / "comparison_summary.md").write_text(md)


def sanity_check_collapse(data) -> int:
    """Flag any model whose predictions look degenerate (mean-collapse).

    Returns the number of models flagged. The default behavior is just to
    print; in CI you could turn it into a non-zero exit if needed.
    """
    issues = []
    for m in MODELS:
        pred = data[m]["pred"]
        gt = data[m]["gt"]
        mae = float(np.mean(np.abs(pred - gt)))
        baseline_mae = float(np.mean(np.abs(gt - gt.mean())))
        r = float(np.corrcoef(pred, gt)[0, 1])
        pred_std = float(pred.std())
        gt_std = float(gt.std())
        ratio = pred_std / gt_std if gt_std > 0 else 0.0

        flags = []
        # Narrow prediction range (collapsed to a few values)
        if ratio < 0.25:
            flags.append(
                f"pred_std/gt_std = {ratio:.3f} < 0.25 (predictions are too narrow)"
            )
        # Weak rank correlation — model doesn't sort samples by gt
        if r < 0.4:
            flags.append(f"Pearson r = {r:.3f} < 0.4 (predictions barely correlate with gt)")
        # MAE not meaningfully better than constant predictor
        if mae > 0.95 * baseline_mae:
            flags.append(
                f"MAE {mae:.3f} is within 5% of predict-mean baseline {baseline_mae:.3f}"
            )
        if flags:
            issues.append((LABELS[m], flags))

    print("\n=== sanity check: mean-collapse / degeneracy flags ===")
    if not issues:
        print("All models look healthy (none flagged).")
        return 0
    for model_label, flags in issues:
        print(f"[WARNING] {model_label}: possible mean-collapse")
        for f in flags:
            print(f"  - {f}")
    print(
        "\nIf you see flags above, do NOT report MAE/RMSE as if the model is "
        "doing meaningful regression — verify with pred_distribution.png and "
        "per-age-group MAE before drawing architecture-level conclusions."
    )
    return len(issues)


def plot_budget_extension_curves():
    """3-panel val_mae trajectory plot, one panel per model architecture.
    Each panel overlays the available budget variants (25 / 50 / 100 ep).

    This is the visualisation for §11.8 of the report — it makes the
    non-monotonic-in-budget finding (50ep is sweet spot, 100ep regresses)
    visually obvious. Separate from the §9-10 headline 50ep-canonical plots.
    """
    # (panel_label, list-of-(budget_label, results_dir, linestyle))
    panels = [
        ("DenseNet121", [
            ("25 ep",            "densenet_25ep",       "--"),
            ("50 ep (canonical)", "densenet",            "-"),
            ("100 ep",           "densenet_100ep",      ":"),
        ]),
        ("ViT-B/16 (new recipe)", [
            ("50 ep (canonical)", "vit",                 "-"),
            ("100 ep",           "vit_100ep",           ":"),
        ]),
        ("ViT-B/16 (baseline recipe)", [
            ("25 ep",            "vit_baseline_25ep",   "--"),
            ("50 ep (canonical)", "vit_baseline",        "-"),
            ("100 ep",           "vit_baseline_100ep",  ":"),
        ]),
    ]
    panel_colors = {
        "DenseNet121": "#1f77b4",
        "ViT-B/16 (new recipe)": "#2ca02c",
        "ViT-B/16 (baseline recipe)": "#d62728",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, (model_label, runs) in zip(axes, panels):
        color = panel_colors[model_label]
        for budget_label, sub, ls in runs:
            d = RES / sub
            if not d.is_dir():
                continue
            df = pd.read_csv(d / "epoch_log.csv")
            ax.plot(df["epoch"], df["val_mae"],
                    color=color, linestyle=ls, linewidth=2,
                    label=budget_label)
            # mark best epoch for this run
            idx = df["val_mae"].idxmin()
            ax.scatter([df.loc[idx, "epoch"]], [df.loc[idx, "val_mae"]],
                       s=70, color=color, edgecolor="black", linewidth=0.8,
                       zorder=5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MAE (years)")
        ax.set_title(model_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlim(0, 102)
    fig.suptitle(
        "Budget extension ablation — val MAE trajectories at 25 / 50 / 100 ep\n"
        "(solid = 50ep canonical; dashed = 25ep; dotted = 100ep; dot = best epoch used for test)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "budget_extension_mae_curves.png", dpi=120)
    plt.close(fig)


def plot_budget_extension_per_age():
    """Per-age MAE bar chart across budget variants for the ViT (baseline),
    to make the 100ep tail-regression visible.

    Three sets of bars (25 / 50 / 100 ep) per age bin, only for vit_baseline
    where the budget effect is most dramatic. (DenseNet and ViT new are
    flat across budgets — covered in the existing per_age_bar.png.)
    """
    bins_ref = pd.read_csv(RES / "vit_baseline" / "per_age_group_mae.csv")["bin"].values
    counts = pd.read_csv(RES / "vit_baseline" / "per_age_group_mae.csv")["count"].values
    x = np.arange(len(bins_ref))
    width = 0.27

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    layers = [
        ("25 ep (collapsed)", "vit_baseline_25ep",  "#ff9999"),
        ("50 ep (canonical)", "vit_baseline",       "#d62728"),
        ("100 ep (regressed)", "vit_baseline_100ep","#7f1d1d"),
    ]
    for i, (label, sub, color) in enumerate(layers):
        df = pd.read_csv(RES / sub / "per_age_group_mae.csv")
        offset = (i - 1) * width
        ax.bar(x + offset, df["mae"].values, width, color=color, label=label)
    ax.set_ylabel("MAE (years)")
    ax.set_title(
        "ViT (baseline recipe) — per-age-group test MAE across budget variants\n"
        "25 ep is the mean-collapse run; 50 ep partial recovery; 100 ep tail-regression"
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax2.bar(x, counts, color="grey", alpha=0.7)
    ax2.set_ylabel("# test samples")
    ax2.set_xlabel("age group")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bins_ref, rotation=20)
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "budget_extension_baseline_per_age.png", dpi=120)
    plt.close(fig)


def write_budget_ablation():
    """Compare ViT baseline at 25ep vs 50ep, and DenseNet at 25ep vs 50ep,
    to isolate the budget-only contribution from the recipe contribution.

    Reads `results/{densenet_25ep, vit_baseline_25ep}/` plus the main 50ep
    results. Skips silently if any of those folders are missing.
    """
    rows = []
    candidates = [
        ("DenseNet121", "25ep", "densenet_25ep"),
        ("DenseNet121", "50ep (canonical)", "densenet"),
        ("DenseNet121", "100ep", "densenet_100ep"),
        ("ViT-B/16 (new recipe)", "50ep (canonical)", "vit"),
        ("ViT-B/16 (new recipe)", "100ep", "vit_100ep"),
        ("ViT-B/16 (baseline recipe)", "25ep", "vit_baseline_25ep"),
        ("ViT-B/16 (baseline recipe)", "50ep (canonical)", "vit_baseline"),
        ("ViT-B/16 (baseline recipe)", "100ep", "vit_baseline_100ep"),
    ]
    for label, budget, sub in candidates:
        d = RES / sub
        if not d.is_dir():
            print(f"  [budget-ablation] skip: {sub} missing")
            continue
        pred = np.load(d / "test_predictions.npy")
        gt = np.load(d / "test_ground_truth.npy")
        s = {}
        for line in (d / "test_summary.txt").read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                s[k.strip()] = v.strip()
        rows.append({
            "model": label,
            "budget": budget,
            "best_epoch": int(s["best_epoch"]),
            "test_mae": float(s["test_mae"]),
            "test_rmse": float(s["test_rmse"]),
            "pred_std": float(pred.std()),
            "pearson_r": float(np.corrcoef(pred, gt)[0, 1]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "budget_ablation.csv", index=False)
    return df


def main():
    data = load_all()
    plot_mae_curves(data)
    plot_loss_curves(data)
    plot_scatter(data)
    plot_per_age(data)
    plot_pred_distribution(data)
    # New (2026-05-28 late): budget-extension ablation plots for §11.8
    plot_budget_extension_curves()
    plot_budget_extension_per_age()
    table = write_table(data)
    write_markdown_summary(data, table)
    budget = write_budget_ablation()
    print("=== comparison_table.csv (50ep, 3 runs) ===")
    print(table.to_string(index=False))
    print()
    print("=== budget_ablation.csv (recipe × budget) ===")
    print(budget.to_string(index=False))
    sanity_check_collapse(data)
    print()
    print(f"All artifacts saved under {OUT}")


if __name__ == "__main__":
    main()
