"""Age estimation on APPA-REAL — shared pipeline for DenseNet & ViT."""
from .dataset import AppaRealDataset, build_dataloaders, build_transform
from .eval import evaluate_and_save_artifacts
from .models import build_model
from .train import evaluate, run_training, train_one_epoch
from .utils import (
    AverageMeter,
    build_criterion,
    build_optimizer,
    build_scheduler,
    ensure_dir,
    gpu_mem_peak_mb,
    load_config,
    per_age_group_mae,
    reset_gpu_mem_peak,
    save_config_snapshot,
    save_env_snapshot,
    save_seed,
    set_seed,
    worker_init_fn,
)

__all__ = [
    "AppaRealDataset",
    "AverageMeter",
    "build_criterion",
    "build_dataloaders",
    "build_model",
    "build_optimizer",
    "build_scheduler",
    "build_transform",
    "ensure_dir",
    "evaluate",
    "evaluate_and_save_artifacts",
    "gpu_mem_peak_mb",
    "load_config",
    "per_age_group_mae",
    "reset_gpu_mem_peak",
    "run_training",
    "save_config_snapshot",
    "save_env_snapshot",
    "save_seed",
    "set_seed",
    "train_one_epoch",
    "worker_init_fn",
]
