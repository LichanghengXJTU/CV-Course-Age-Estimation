"""Shared training & validation loop.

The same `run_training` function is used for DenseNet and ViT — the only
differences come from the YAML config (optimizer, lr, scheduler, batch_size,
weight_decay). Do NOT add `if model_name == 'vit'` branches here.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import (
    AverageMeter,
    build_criterion,
    build_optimizer,
    build_scheduler,
    ensure_dir,
    gpu_mem_peak_mb,
    reset_gpu_mem_peak,
)


# Sampling cadence for step_log.csv
STEP_LOG_EVERY = 20


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    step_log_rows: list,
) -> float:
    """One epoch of training. Returns average loss across the epoch."""
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(loader, desc=f"epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar):
        imgs, ages, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True).float()

        optimizer.zero_grad()
        out = model(imgs)
        # head outputs (B, 1) -> (B,)
        out = out.squeeze(1)
        loss = criterion(out, ages)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        loss_meter.update(batch_loss, imgs.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if step % STEP_LOG_EVERY == 0:
            step_log_rows.append({
                "epoch": epoch,
                "step": step,
                "batch_loss": batch_loss,
                "running_avg_loss": loss_meter.avg,
            })

    return loss_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_preds: bool = False,
) -> Tuple[float, float, float, Optional[np.ndarray], Optional[np.ndarray], Optional[list]]:
    """Evaluate. Returns (loss_avg, mae, rmse, preds, gts, names).

    preds/gts/names are None when return_preds=False.
    """
    model.eval()
    loss_meter = AverageMeter()
    abs_errs = []
    sq_errs = []
    all_preds = [] if return_preds else None
    all_gts = [] if return_preds else None
    all_names = [] if return_preds else None

    for batch in tqdm(loader, desc="eval", leave=False):
        imgs, ages, names = batch
        imgs = imgs.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True).float()
        out = model(imgs).squeeze(1)
        loss = criterion(out, ages)
        loss_meter.update(float(loss.item()), imgs.size(0))
        diff = (out - ages).detach().cpu().numpy()
        abs_errs.append(np.abs(diff))
        sq_errs.append(diff ** 2)
        if return_preds:
            all_preds.append(out.detach().cpu().numpy())
            all_gts.append(ages.detach().cpu().numpy())
            all_names.extend(list(names))

    abs_errs = np.concatenate(abs_errs) if abs_errs else np.zeros(0)
    sq_errs = np.concatenate(sq_errs) if sq_errs else np.zeros(0)
    mae = float(abs_errs.mean()) if abs_errs.size else float("nan")
    rmse = float(math.sqrt(sq_errs.mean())) if sq_errs.size else float("nan")

    if return_preds:
        preds = np.concatenate(all_preds, axis=0)
        gts = np.concatenate(all_gts, axis=0)
        return loss_meter.avg, mae, rmse, preds, gts, all_names
    return loss_meter.avg, mae, rmse, None, None, None


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def run_training(
    model: nn.Module,
    model_name: str,
    cfg: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,  # carried through for symmetry; not used here
    device: torch.device,
    out_dir: str,
    ckpt_dir: str,
) -> Tuple[float, pd.DataFrame, str]:
    """Train `model` per `cfg`, log everything, save best & last checkpoints.

    Returns (best_val_mae, history_df, best_ckpt_path).
    """
    del test_loader  # explicitly unused here; test happens in eval module
    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)

    epochs = int(cfg["epochs"])
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    model.to(device)

    step_log_rows: list = []
    epoch_log_rows: list = []

    best_val_mae = float("inf")
    best_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
    last_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_last.pth")

    step_log_csv = os.path.join(out_dir, "step_log.csv")
    epoch_log_csv = os.path.join(out_dir, "epoch_log.csv")

    for epoch in range(1, epochs + 1):
        reset_gpu_mem_peak()
        t0 = time.time()
        lr_now = _current_lr(optimizer)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, step_log_rows
        )

        val_loss, val_mae, val_rmse, *_ = evaluate(
            model, val_loader, criterion, device, return_preds=False
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        peak_mem = gpu_mem_peak_mb()

        epoch_log_rows.append({
            "epoch": epoch,
            "lr": lr_now,
            "train_loss_avg": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "epoch_time_sec": elapsed,
            "gpu_mem_peak_mb": peak_mem,
        })

        # Flush logs every epoch so partial runs are still useful.
        pd.DataFrame(step_log_rows).to_csv(step_log_csv, index=False)
        pd.DataFrame(epoch_log_rows).to_csv(epoch_log_csv, index=False)

        # Save last
        torch.save({
            "model_name": model_name,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "cfg": cfg,
        }, last_ckpt_path)

        # Save best by val MAE
        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            torch.save({
                "model_name": model_name,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "cfg": cfg,
            }, best_ckpt_path)

        print(
            f"[{model_name}] epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f} | "
            f"lr={lr_now:.2e} | time={elapsed:.1f}s | "
            f"gpu_mem_peak_mb={peak_mem:.1f}"
            + ("  <- best" if improved else "")
        )

    history_df = pd.DataFrame(epoch_log_rows)
    return best_val_mae, history_df, best_ckpt_path
