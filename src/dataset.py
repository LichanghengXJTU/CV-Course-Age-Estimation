"""APPA-REAL dataset and dataloaders.

Splits: train / valid / test (matches official APPA-REAL layout).
Labels come from gt_avg_<split>.csv (apparent_age_avg).
Optional ignore_list.csv at the dataset root drops listed file_names.

Same transform pipeline (train aug / val-test eval) used by both models —
this is part of the fair-experiment protocol.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(img_size: int, train: bool) -> Callable:
    """Train augmentation vs. eval transform. Shared across both models."""
    if train:
        # Resize a bit larger, crop, flip, jitter — standard light aug.
        return T.Compose([
            T.Resize((int(img_size * 1.15), int(img_size * 1.15))),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class AppaRealDataset(Dataset):
    """APPA-REAL face dataset.

    Returns (image_tensor, age_float_tensor, file_name_str) per item.
    """

    SPLITS = ("train", "valid", "test")

    def __init__(
        self,
        data_root: str,
        split: str,
        img_size: int = 224,
        train: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split}")
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.transform = transform if transform is not None else build_transform(
            img_size, train=train
        )

        csv_path = os.path.join(data_root, f"gt_avg_{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Label CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        required = {"file_name", "apparent_age_avg"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {missing}"
            )

        # Optional ignore list at dataset root.
        ignore_path = os.path.join(data_root, "ignore_list.csv")
        ignore: set = set()
        if os.path.exists(ignore_path):
            ig_df = pd.read_csv(ignore_path)
            if "img_name" in ig_df.columns:
                ignore = set(ig_df["img_name"].astype(str).values)

        img_dir = os.path.join(data_root, split)
        self.file_names: List[str] = []
        self.img_paths: List[str] = []
        self.ages: List[float] = []
        skipped_missing = 0
        for _, row in df.iterrows():
            fname = str(row["file_name"])
            if fname in ignore:
                continue
            img_path = os.path.join(img_dir, fname + "_face.jpg")
            if not os.path.exists(img_path):
                skipped_missing += 1
                continue
            self.file_names.append(fname)
            self.img_paths.append(img_path)
            self.ages.append(float(row["apparent_age_avg"]))

        if len(self.img_paths) == 0:
            raise RuntimeError(
                f"No samples loaded for split={split} from {data_root}. "
                f"Check that '{img_dir}' exists and has *_face.jpg files."
            )
        if skipped_missing > 0:
            print(
                f"[AppaRealDataset:{split}] skipped {skipped_missing} rows "
                f"with missing image files."
            )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        return img, age, self.file_names[idx]


def build_dataloaders(
    cfg: dict,
    data_root: str,
    worker_init_fn: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from a config dict.

    Both DenseNet and ViT pipelines call this with their own cfg —
    splits / transforms / shuffling are model-agnostic.
    """
    img_size = int(cfg.get("img_size", 224))
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 4))
    pin = torch.cuda.is_available()

    train_set = AppaRealDataset(data_root, "train", img_size=img_size, train=True)
    val_set = AppaRealDataset(data_root, "valid", img_size=img_size, train=False)
    test_set = AppaRealDataset(data_root, "test", img_size=img_size, train=False)

    # For deterministic shuffling we use the global torch seed (set_seed at startup)
    # and reseed workers via worker_init_fn.
    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 42)))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    print(
        f"[dataloaders] train={len(train_set)} | val={len(val_set)} | "
        f"test={len(test_set)} | bs={batch_size} | workers={num_workers}"
    )
    return train_loader, val_loader, test_loader
