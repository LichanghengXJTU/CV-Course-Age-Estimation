"""Test-time evaluation and artifact production.

Loads best checkpoint, runs test set, saves: predictions, ground truth,
file names, per-age-group MAE CSV, loss/MAE curves, pred-vs-true scatter.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # noqa: E402  no display on server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import evaluate
from .utils import ensure_dir, per_age_group_mae


def _load_state_dict(model: nn.Module, ckpt_path: str, device: torch.device) -> dict:
    payload = torch.load(ckpt_path, map_location=device)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        # state_dict is strict so this branch is essentially unreachable —
        # but keep defensive note.
        print(f"[eval] load warnings: missing={missing}, unexpected={unexpected}")
    return payload if isinstance(payload, dict) else {}


def _plot_loss_curves(history_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = history_df["epoch"].values
    ax.plot(epochs, history_df["train_loss_avg"].values, label="train loss", marker="o")
    ax.plot(epochs, history_df["val_loss"].values, label="val loss", marker="s")
    ax.set_xlabel("epoch")
    ax.set_ylabel("L1 loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_mae_curve(history_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        history_df["epoch"].values,
        history_df["val_mae"].values,
        label="val MAE", marker="o", color="tab:orange",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (years)")
    ax.set_title("Validation MAE over epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_pred_vs_true(preds: np.ndarray, gts: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gts, preds, alpha=0.4, s=12)
    lo = float(min(gts.min(), preds.min()))
    hi = float(max(gts.max(), preds.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel("ground-truth age")
    ax.set_ylabel("predicted age")
    ax.set_title("Predicted vs. ground-truth age (test)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate_and_save_artifacts(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    out_dir: str,
    ckpt_path: str,
    history_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Run best-checkpoint evaluation on the test set, save all artifacts.

    Returns a dict of headline numbers: {test_mae, test_rmse, test_loss, ...}.
    """
    ensure_dir(out_dir)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Train first, or pass --ckpt_dir pointing to the right place."
        )
    payload = _load_state_dict(model, ckpt_path, device)
    model.to(device)

    test_loss, test_mae, test_rmse, preds, gts, names = evaluate(
        model, test_loader, criterion, device, return_preds=True
    )
    print(
        f"[{model_name}] TEST loss={test_loss:.4f} | "
        f"MAE={test_mae:.4f} | RMSE={test_rmse:.4f}  (from {ckpt_path})"
    )

    # Save raw arrays / names
    np.save(os.path.join(out_dir, "test_predictions.npy"), preds)
    np.save(os.path.join(out_dir, "test_ground_truth.npy"), gts)
    with open(os.path.join(out_dir, "test_file_names.txt"), "w") as f:
        f.write("\n".join(names) + "\n")

    # Per-age-group MAE
    per_bin_df = per_age_group_mae(preds, gts)
    per_bin_df.to_csv(os.path.join(out_dir, "per_age_group_mae.csv"), index=False)

    # Curves require history
    if history_df is not None and len(history_df) > 0:
        _plot_loss_curves(history_df, os.path.join(out_dir, "loss_curve.png"))
        _plot_mae_curve(history_df, os.path.join(out_dir, "mae_curve.png"))
    else:
        # Fall back to reading epoch_log.csv if it's there
        epoch_csv = os.path.join(out_dir, "epoch_log.csv")
        if os.path.exists(epoch_csv):
            df = pd.read_csv(epoch_csv)
            if len(df) > 0:
                _plot_loss_curves(df, os.path.join(out_dir, "loss_curve.png"))
                _plot_mae_curve(df, os.path.join(out_dir, "mae_curve.png"))

    _plot_pred_vs_true(preds, gts, os.path.join(out_dir, "pred_vs_true_scatter.png"))

    # Quick test summary file (human-readable)
    summary_path = os.path.join(out_dir, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"model: {model_name}\n")
        f.write(f"ckpt: {ckpt_path}\n")
        if isinstance(payload, dict) and "epoch" in payload:
            f.write(f"best_epoch: {payload['epoch']}\n")
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"test_mae: {test_mae:.6f}\n")
        f.write(f"test_rmse: {test_rmse:.6f}\n")
        f.write(f"n_test: {len(preds)}\n")

    return {
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "n_test": int(len(preds)),
        "ckpt_path": ckpt_path,
    }
