"""Utilities: seeds, snapshots, optimizer/scheduler/criterion factories, metrics.

Shared by both DenseNet and ViT pipelines. No model-specific branching here.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml


# --------------------------------------------------------------------------- #
# Config I/O
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")
    return cfg


def save_config_snapshot(cfg: Dict[str, Any], out_path: str) -> None:
    """Persist the post-CLI-override config as YAML."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def save_env_snapshot(out_path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Persist python/torch/cuda/gpu/git-rev/cmdline/timestamp."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []
    lines.append(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"python: {sys.version.split()[0]}")
    lines.append(f"executable: {sys.executable}")
    lines.append(f"cmdline: {' '.join(sys.argv)}")
    lines.append(f"cwd: {os.getcwd()}")
    try:
        lines.append(f"torch: {torch.__version__}")
        lines.append(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"torch.version.cuda: {torch.version.cuda}")
            lines.append(f"cudnn: {torch.backends.cudnn.version()}")
            lines.append(f"gpu_count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                lines.append(f"gpu[{i}]: {torch.cuda.get_device_name(i)}")
    except Exception as e:  # noqa: BLE001 - just record, don't crash
        lines.append(f"torch_info_error: {e}")
    # git rev
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        lines.append(f"git_rev: {rev}")
    except Exception:
        lines.append("git_rev: <not a git repo or git missing>")
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        lines.append(f"git_dirty: {'yes' if dirty else 'no'}")
    except Exception:
        pass
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_seed(seed: int, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{seed}\n")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Set all RNGs and enable deterministic cudnn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker init: seed each worker from torch's initial seed."""
    base_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


# --------------------------------------------------------------------------- #
# Meters and GPU memory
# --------------------------------------------------------------------------- #
class AverageMeter:
    """Running average meter."""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def reset_gpu_mem_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_mem_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# --------------------------------------------------------------------------- #
# Factories (optimizer / scheduler / criterion)
# --------------------------------------------------------------------------- #
def build_criterion(cfg: Dict[str, Any]) -> nn.Module:
    """Build loss from cfg. L1 is the default for the fair-experiment protocol;
    MSE and SmoothL1 are also supported for ablation runs (the protocol is
    enforced at the config layer, not here)."""
    name = str(cfg.get("loss", "l1")).lower()
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name in ("smooth_l1", "smoothl1", "huber"):
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss '{name}'. Allowed: l1, mse, smooth_l1.")


def _vit_param_depth(name: str, n_blocks: int = 12) -> int:
    """Return LLRD depth for a torchvision ViT-B param name.

    Depth 0 = closest to output (head, final LayerNorm) -> highest lr.
    Depth n_blocks+1 = embeddings -> lowest lr (base_lr * decay^(n_blocks+1)).
    """
    if "heads" in name:
        return 0
    if name.startswith("encoder.ln"):
        return 0
    if "encoder.layers.encoder_layer_" in name:
        idx = int(name.split("encoder_layer_")[1].split(".")[0])
        return n_blocks - idx  # block_{n-1} -> 1, block_0 -> n
    # conv_proj (patch embed), encoder.pos_embedding, class_token, etc.
    return n_blocks + 1


def _build_param_groups_llrd(
    model: nn.Module,
    base_lr: float,
    decay: float,
    weight_decay: float,
    n_blocks: int = 12,
) -> list:
    """Per-layer param groups for layer-wise LR decay on torchvision ViT-B/16.

    - lr_i = base_lr * decay^depth_i
    - LayerNorm / bias / class_token / pos_embedding params get weight_decay=0
      (standard transformer convention).
    """
    groups: Dict = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        depth = _vit_param_depth(name, n_blocks=n_blocks)
        no_wd = (
            name.endswith(".bias")
            or "ln" in name.lower()
            or "norm" in name.lower()
            or name == "class_token"
            or name.endswith("pos_embedding")
        )
        key = (depth, no_wd)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * (decay ** depth),
                "weight_decay": 0.0 if no_wd else weight_decay,
                "depth": depth,
                "no_wd": no_wd,
            }
        groups[key]["params"].append(p)
    return [groups[k] for k in sorted(groups.keys())]


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer from cfg. Allowed: adam, adamw, sgd.

    If `cfg['llrd']` is true, build per-layer param groups with decayed lr
    (`llrd_decay` defaults 0.75, `llrd_n_blocks` defaults 12 for ViT-B/16).
    """
    name = str(cfg.get("optimizer", "adam")).lower()
    base_lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))

    if bool(cfg.get("llrd", False)):
        decay = float(cfg.get("llrd_decay", 0.75))
        n_blocks = int(cfg.get("llrd_n_blocks", 12))
        param_groups = _build_param_groups_llrd(
            model, base_lr, decay, wd, n_blocks=n_blocks
        )
    else:
        param_groups = [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": wd,
        }]

    if name == "adam":
        return torch.optim.Adam(param_groups)
    if name == "adamw":
        return torch.optim.AdamW(param_groups)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(param_groups, momentum=momentum)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build LR scheduler from cfg['scheduler']. Returns None if absent."""
    sch = cfg.get("scheduler", None)
    if sch is None:
        return None
    name = str(sch.get("name", "")).lower()
    if name in ("", "none"):
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch["step_size"]),
            gamma=float(sch.get("gamma", 0.1)),
        )
    if name == "cosine":
        t_max = int(sch.get("t_max", cfg.get("epochs", 25)))
        eta_min = float(sch.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    if name == "cosine_warmup":
        warmup_epochs = int(sch.get("warmup_epochs", 5))
        t_max = int(sch.get("t_max", cfg.get("epochs", 50)))
        eta_min = float(sch.get("eta_min", 0.0))
        start_factor = float(sch.get("warmup_start_factor", 0.01))
        # Phase 1: linear warmup from start_factor * lr to lr over warmup_epochs.
        # Phase 2: cosine from lr down to eta_min over (t_max - warmup_epochs).
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=max(1, warmup_epochs),
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max - warmup_epochs),
            eta_min=eta_min,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    if name == "multistep":
        milestones = [int(m) for m in sch["milestones"]]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=float(sch.get("gamma", 0.1))
        )
    raise ValueError(f"Unknown scheduler '{name}'")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def per_age_group_mae(
    preds: np.ndarray, gts: np.ndarray, bin_edges: Optional[list] = None
) -> pd.DataFrame:
    """Compute MAE within age bins: [0,10), [10,20), ..., [70, inf).

    Returns a DataFrame with columns: bin, low, high, count, mae.
    """
    preds = np.asarray(preds).reshape(-1)
    gts = np.asarray(gts).reshape(-1)
    if bin_edges is None:
        bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, np.inf]
    rows = []
    abs_err = np.abs(preds - gts)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if np.isinf(hi):
            mask = gts >= lo
            label = f"[{int(lo)}, inf)"
        else:
            mask = (gts >= lo) & (gts < hi)
            label = f"[{int(lo)}, {int(hi)})"
        n = int(mask.sum())
        mae = float(abs_err[mask].mean()) if n > 0 else float("nan")
        rows.append({
            "bin": label,
            "low": float(lo),
            "high": float(hi) if not np.isinf(hi) else float("inf"),
            "count": n,
            "mae": mae,
        })
    return pd.DataFrame(rows)


def ensure_dir(path: str) -> str:
    """mkdir -p path; return path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
