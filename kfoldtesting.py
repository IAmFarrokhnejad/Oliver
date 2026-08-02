"""
Definitive ConvNeXt Testing with Attention Mechanisms (5-Fold CV)
======================================================
Authors: Morteza Farrokhnejad, Ali Hakan Ulusoy, Ahmet Rizaner

- Trains a ConvNeXt-Base model with an SE attention mechanism on a series of
  Olive datasets.
- Implements Early Stopping (patience=10) with up to 50 epochs.
- Runs every dataset x attention combination using 5-Fold Cross-Validation.
- Records all critical validation and test metrics tracking configuration differences.
- Writes per-fold and averaged CSV files next to this script:
    convnext_results_runs.csv
    convnext_results_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
import traceback
import warnings
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Requested output location
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(r"") #Specifiy this path
RUNS_CSV = SCRIPT_DIR / "convnext_results_runs.csv"
SUMMARY_CSV = SCRIPT_DIR / "convnext_results_summary.csv"
STATE_PATH = SCRIPT_DIR / "convnext_run_state.json"

DEFAULT_FOLDS = 5
SEED_BASE = 1000
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset definitions
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS: List[Dict] = [
    {
        "name": "dataset1_oliver",
        "kind": "presplit",
        "root": Path(r""), # Specifiy this path
        "train_split_name": "train",
        "test_split_name": "test",
        "val_split_names": ["val", "valid", "validation"],
        "val_split_from_train": 0.20,
        "image_size": 224,
    },
    {
        "name": "dataset2_uguz",
        "kind": "presplit",
        "root": Path(r""), # Specifiy this path
        "train_split_name": "train",
        "test_split_name": "test",
        "val_split_names": ["val", "valid", "validation"],
        "val_split_from_train": 0.20,
        "image_size": 224,
    },
    {
        "name": "dataset3_diker",
        "kind": "flat",
        "root": Path(r""), # Specifiy this path
        "test_split": 0.20,
        "val_split": 0.20,
        "split_seed": 42,
        "image_size": 224,
    },
    {
        "name": "dataset4_zeytin_augmented",
        "kind": "presplit",
        "root": Path(r""), # Specifiy this path
        "train_split_name": "train",
        "test_split_name": "test",
        "val_split_names": ["val", "valid", "validation"],
        "val_split_from_train": 0.20,
        "image_size": 224,
    },
    {
        "name": "dataset5_osco_mamani",
        "kind": "presplit",
        "root": Path(r""), # Specifiy this path
        "train_split_name": "train",
        "test_split_name": "test",
        "val_split_names": ["valid", "val", "validation"],
        "val_split_from_train": 0.20,
        "image_size": 224,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# ConvNeXt model variant to test
# ─────────────────────────────────────────────────────────────────────────────

MODEL_VARIANTS = ["convnext_base"]

# ─────────────────────────────────────────────────────────────────────────────
# ConvNeXt hyperparameters (shared, unchanged, across all model variants)
# ─────────────────────────────────────────────────────────────────────────────

CONVNEXT_HP = {
    "combo_id": 1,
    "pretrained": True,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "batch_size": 16,
    "epochs": 50,            
    "patience": 10,          
    "optimizer": "adamw",
    "lr_scheduler": "cosine",
    "dropout": 0.0,
    "use_weighted_sampler": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# Attention Configurations to Explore
# ─────────────────────────────────────────────────────────────────────────────

ATTENTION_CONFIGS = [
    {"type": "SE", "param": 8},
]

# ─────────────────────────────────────────────────────────────────────────────
# CSV columns
# ─────────────────────────────────────────────────────────────────────────────

RUN_COLUMNS = [
    "run_id", "dataset", "model_variant", "attention_type", "attention_param", "fold_idx", "seed", "timestamp",
    "num_classes", "class_names", "num_train", "num_val", "num_test",
    "val_accuracy", "val_f1_macro",
    "test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro",
    "test_precision_weighted", "test_recall_weighted", "test_f1_weighted",
    "per_class_f1", "confusion_matrix",
    "train_time_sec", "best_epoch", "peak_gpu_memory_mb", 
    "status", "error_message",
]

SUMMARY_COLUMNS = [
    "dataset", "model_variant", "attention_type", "attention_param", "folds_completed", "folds_expected",
    "mean_test_accuracy", "std_test_accuracy",
    "mean_test_f1_macro", "std_test_f1_macro",
    "mean_test_f1_weighted", "std_test_f1_weighted",
    "mean_test_precision_macro", "std_test_precision_macro",
    "mean_test_recall_macro", "std_test_recall_macro",
    "mean_val_accuracy", "std_val_accuracy",
    "mean_val_f1_macro", "std_val_f1_macro",
    "mean_train_time_sec", "mean_best_epoch", "last_updated",
]

# ─────────────────────────────────────────────────────────────────────────────
# Attention Module Implementations
# ─────────────────────────────────────────────────────────────────────────────

class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out.view(b, c, 1, 1))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAMModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ECAModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SelfAttentionModule(nn.Module):
    def __init__(self, channels: int, heads: int = 1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        out = x.flatten(2).transpose(1, 2)
        out, _ = self.mha(out, out, out)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out

# ─────────────────────────────────────────────────────────────────────────────
# ConvNeXt Network Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ConvNeXtWithAttention(nn.Module):
    def __init__(self, base_model: nn.Module, attention_layer: Optional[nn.Module]):
        super().__init__()
        self.features = base_model.features
        self.attention = attention_layer
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CVDataset:
    dataset_name: str
    class_names: List[str]
    paths: np.ndarray
    labels: np.ndarray
    image_size: int

@dataclass
class SplitData:
    dataset_name: str
    class_names: List[str]
    train_paths: List[str]
    train_labels: np.ndarray
    val_paths: List[str]
    val_labels: np.ndarray
    test_paths: List[str]
    test_labels: np.ndarray
    image_size: int

@dataclass
class ConvNeXtOutputs:
    val_probs: np.ndarray
    val_preds: np.ndarray
    val_labels: np.ndarray
    test_probs: np.ndarray
    test_preds: np.ndarray
    test_labels: np.ndarray
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_time_sec: float
    best_epoch: int
    peak_gpu_memory_mb: float

# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_state() -> Dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": []}

def save_state(state: Dict) -> None:
    ensure_dir(SCRIPT_DIR)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def make_run_id(dataset_name: str, model_variant: str, att_type: str, att_param: int, fold_idx: int) -> str:
    return f"{dataset_name}|{model_variant}|{att_type}|{att_param}|fold{fold_idx}"

def init_csvs(force: bool = False) -> None:
    ensure_dir(SCRIPT_DIR)
    if force:
        for p in (RUNS_CSV, SUMMARY_CSV, STATE_PATH):
            if p.exists():
                p.unlink()
    if not RUNS_CSV.exists():
        with open(RUNS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=RUN_COLUMNS).writeheader()
    if not SUMMARY_CSV.exists():
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS).writeheader()

def append_run_rows(rows: List[Dict]) -> None:
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS, extrasaction="ignore")
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in RUN_COLUMNS})

def _mean(vals: List[float]) -> float | str:
    return round(statistics.mean(vals), 6) if vals else ""

def _std(vals: List[float]) -> float | str:
    return round(statistics.stdev(vals), 6) if len(vals) > 1 else (0.0 if vals else "")

def refresh_summary(folds_expected: int) -> None:
    groups: Dict[Tuple[str, str, str, str], List[Dict]] = {}
    if not RUNS_CSV.exists():
        return

    with open(RUNS_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "success":
                continue
            key = (row["dataset"], row["model_variant"], row["attention_type"], row["attention_param"])
            groups.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []
    for (dataset, model_variant, att_type, att_param), rows in sorted(groups.items()):
        def col(name: str) -> List[float]:
            out: List[float] = []
            for r in rows:
                try:
                    if r.get(name) not in ("", None, "nan"):
                        out.append(float(r[name]))
                except Exception:
                    pass
            return out

        summary_rows.append({
            "dataset": dataset,
            "model_variant": model_variant,
            "attention_type": att_type,
            "attention_param": att_param,
            "folds_completed": len(rows),
            "folds_expected": folds_expected,
            "mean_test_accuracy": _mean(col("test_accuracy")),
            "std_test_accuracy": _std(col("test_accuracy")),
            "mean_test_f1_macro": _mean(col("test_f1_macro")),
            "std_test_f1_macro": _std(col("test_f1_macro")),
            "mean_test_f1_weighted": _mean(col("test_f1_weighted")),
            "std_test_f1_weighted": _std(col("test_f1_weighted")),
            "mean_test_precision_macro": _mean(col("test_precision_macro")),
            "std_test_precision_macro": _std(col("test_precision_macro")),
            "mean_test_recall_macro": _mean(col("test_recall_macro")),
            "std_test_recall_macro": _std(col("test_recall_macro")),
            "mean_val_accuracy": _mean(col("val_accuracy")),
            "std_val_accuracy": _std(col("val_accuracy")),
            "mean_val_f1_macro": _mean(col("val_f1_macro")),
            "std_val_f1_macro": _std(col("val_f1_macro")),
            "mean_train_time_sec": _mean(col("train_time_sec")),
            "mean_best_epoch": _mean(col("best_epoch")),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery and splitting
# ─────────────────────────────────────────────────────────────────────────────

def list_class_dirs(root: Path) -> List[str]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def collect_split_from_class_dirs(split_root: Path, class_names: List[str]) -> Tuple[List[str], np.ndarray]:
    paths: List[str] = []
    labels: List[int] = []
    for idx, cls in enumerate(class_names):
        folder = split_root / cls
        if not folder.exists():
            print(f"  WARNING: missing class folder: {folder}")
            continue
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        paths.extend(str(p) for p in files)
        labels.extend([idx] * len(files))
    return paths, np.asarray(labels, dtype=np.int64)

def can_stratify(y: np.ndarray) -> bool:
    if len(y) == 0:
        return False
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min()) >= 2

def resolve_validation_split(root: Path, val_names: Sequence[str]) -> Optional[Path]:
    for name in val_names:
        candidate = root / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None

def load_cv_data(cfg: Dict) -> CVDataset:
    root = Path(cfg["root"])
    image_size = int(cfg.get("image_size", 224))

    all_paths: List[str] = []
    all_labels: List[int] = []

    if cfg["kind"] == "presplit":
        train_root = root / cfg.get("train_split_name", "train")
        class_names = list_class_dirs(train_root)
        
        splits_to_check = [cfg.get("train_split_name", "train")]
        if "test_split_name" in cfg:
            splits_to_check.append(cfg["test_split_name"])
            
        val_root = resolve_validation_split(root, cfg.get("val_split_names", ["val", "valid", "validation"]))
        if val_root:
            splits_to_check.append(val_root.name)

        # Merge pre-split folders into one unified dataset
        for sp in set(splits_to_check):
            sp_root = root / sp
            if sp_root.exists():
                p, l = collect_split_from_class_dirs(sp_root, class_names)
                all_paths.extend(p)
                all_labels.extend(l)

    elif cfg["kind"] == "flat":
        class_names = list_class_dirs(root)
        p, l = collect_split_from_class_dirs(root, class_names)
        all_paths.extend(p)
        all_labels.extend(l)
    else:
        raise ValueError(f"Unknown dataset kind: {cfg['kind']}")

    # Ensure absolute path uniqueness
    unique_paths, indices = np.unique(all_paths, return_index=True)
    unique_labels = np.array(all_labels)[indices]

    if len(unique_paths) == 0:
        raise RuntimeError(f"Empty split combination in {cfg['name']}")

    return CVDataset(
        dataset_name=cfg["name"],
        class_names=class_names,
        paths=unique_paths,
        labels=np.asarray(unique_labels, dtype=np.int64),
        image_size=image_size,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ConvNeXt pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], labels: np.ndarray, class_names: List[str], transform=None):
        self.paths = list(paths)
        self.targets = [int(x) for x in labels]
        self.classes = list(class_names)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]

def get_transforms(image_size: int, is_train: bool):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def make_weighted_sampler(dataset: ImagePathDataset) -> WeightedRandomSampler:
    targets = np.asarray(dataset.targets, dtype=np.int64)
    class_counts = np.bincount(targets, minlength=len(dataset.classes)).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

_MODEL_MAP = {
    "convnext_base": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1),
}

def build_model(variant: str, pretrained: bool, num_classes: int, dropout: float, attention_type: str, attention_param: int):
    fn, weights = _MODEL_MAP[variant]
    base_model = fn(weights=weights if pretrained else None)
    in_features = base_model.classifier[-1].in_features
    
    # Build chosen custom attention layer
    attention_layer = None
    if attention_type == "SE":
        attention_layer = SEModule(channels=in_features, reduction=attention_param)
    
    if dropout > 0:
        base_model.classifier[-1] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    else:
        base_model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    model = ConvNeXtWithAttention(base_model, attention_layer)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    return model, round(total_params, 2)

def build_optimizer(model, hp: Dict):
    if hp["optimizer"] == "adamw":
        return optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    return optim.SGD(model.parameters(), lr=hp["learning_rate"], momentum=0.9, weight_decay=hp["weight_decay"])

def build_scheduler(optimizer, hp: Dict):
    name = hp["lr_scheduler"]
    epochs = hp["epochs"]
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    return None

def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None, num_workers: int = 4):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)

def amp_context(device: torch.device):
    return autocast() if device.type == "cuda" else nullcontext()

def train_one_epoch(model, loader, optimizer, criterion, scaler, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)

@torch.no_grad()
def evaluate_convnext_probs(model, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with amp_context(device):
            outputs = model(imgs)
        probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.vstack(all_probs), np.concatenate(all_preds), np.concatenate(all_labels)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float | str]:
    average_macro = "binary" if num_classes == 2 else "macro"
    average_weighted = "binary" if num_classes == 2 else "weighted"
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average=average_macro, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average=average_macro, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average=average_macro, zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average=average_weighted, zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average=average_weighted, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average=average_weighted, zero_division=0)),
        "per_class_f1": json.dumps([round(float(x), 6) for x in per_class]),
        "confusion_matrix": json.dumps(cm.tolist()),
    }

def train_convnext(split: SplitData, seed: int, device: torch.device, num_workers: int, model_variant: str, attention_type: str, attention_param: int) -> ConvNeXtOutputs:
    set_seed(seed)
    hp = CONVNEXT_HP
    num_classes = len(split.class_names)

    train_ds = ImagePathDataset(split.train_paths, split.train_labels, split.class_names, get_transforms(split.image_size, True))
    val_ds = ImagePathDataset(split.val_paths, split.val_labels, split.class_names, get_transforms(split.image_size, False))
    test_ds = ImagePathDataset(split.test_paths, split.test_labels, split.class_names, get_transforms(split.image_size, False))

    if hp["use_weighted_sampler"]:
        sampler = make_weighted_sampler(train_ds)
        train_loader = make_loader(train_ds, hp["batch_size"], sampler=sampler, num_workers=num_workers)
    else:
        train_loader = make_loader(train_ds, hp["batch_size"], shuffle=True, num_workers=num_workers)

    val_loader = make_loader(val_ds, hp["batch_size"] * 2, shuffle=False, num_workers=max(0, num_workers // 2))
    test_loader = make_loader(test_ds, hp["batch_size"] * 2, shuffle=False, num_workers=max(0, num_workers // 2))

    model, _ = build_model(model_variant, hp["pretrained"], num_classes, hp["dropout"], attention_type, attention_param)
    model = model.to(device)

    class_counts = np.bincount(split.train_labels, minlength=num_classes).astype(float)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = build_optimizer(model, hp)
    scheduler = build_scheduler(optimizer, hp)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = -1.0
    best_val_f1 = -1.0
    best_epoch = 0
    best_weights = None
    
    patience = hp.get("patience", 10)
    epochs_no_improve = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter()
    for epoch in range(1, hp["epochs"] + 1):
        ep_start = time.perf_counter()
        train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_probs, val_preds, val_labels = evaluate_convnext_probs(model, val_loader, device)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        if scheduler:
            scheduler.step()

        elapsed = time.perf_counter() - ep_start
        print(
            f"    {model_variant} ({attention_type}-{attention_param}) epoch {epoch:03d}/{hp['epochs']:03d} | "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={elapsed:.1f}s"
        )
        
        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_val_f1 = float(val_f1)
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"    Early stopping triggered after {epoch} epochs (patience {patience}).")
            break

    train_time = time.perf_counter() - train_start
    if best_weights is not None:
        model.load_state_dict(best_weights)

    val_probs, val_preds, val_labels = evaluate_convnext_probs(model, val_loader, device)
    test_probs, test_preds, test_labels = evaluate_convnext_probs(model, test_loader, device)

    peak_gpu = 0.0
    if device.type == "cuda":
        peak_gpu = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    val_metrics = compute_metrics(val_labels, val_preds, num_classes)
    test_metrics = compute_metrics(test_labels, test_preds, num_classes)

    del model, optimizer, scaler, best_weights
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ConvNeXtOutputs(
        val_probs=val_probs,
        val_preds=val_preds,
        val_labels=val_labels,
        test_probs=test_probs,
        test_preds=test_preds,
        test_labels=test_labels,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_time_sec=train_time,
        best_epoch=best_epoch,
        peak_gpu_memory_mb=peak_gpu,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Run orchestration
# ─────────────────────────────────────────────────────────────────────────────

def build_run_row(
    run_id: str,
    split: SplitData,
    model_variant: str,
    attention_type: str,
    attention_param: int,
    fold_idx: int,
    seed: int,
    conv: ConvNeXtOutputs,
) -> Dict:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "run_id": run_id,
        "dataset": split.dataset_name,
        "model_variant": model_variant,
        "attention_type": attention_type,
        "attention_param": attention_param,
        "fold_idx": fold_idx,
        "seed": seed,
        "timestamp": timestamp,
        "num_classes": len(split.class_names),
        "class_names": json.dumps(split.class_names, ensure_ascii=False),
        "num_train": len(split.train_paths),
        "num_val": len(split.val_paths),
        "num_test": len(split.test_paths),
        "val_accuracy": round(float(conv.val_metrics["accuracy"]), 6),
        "val_f1_macro": round(float(conv.val_metrics["f1_macro"]), 6),
        "test_accuracy": round(float(conv.test_metrics["accuracy"]), 6),
        "test_precision_macro": round(float(conv.test_metrics["precision_macro"]), 6),
        "test_recall_macro": round(float(conv.test_metrics["recall_macro"]), 6),
        "test_f1_macro": round(float(conv.test_metrics["f1_macro"]), 6),
        "test_precision_weighted": round(float(conv.test_metrics["precision_weighted"]), 6),
        "test_recall_weighted": round(float(conv.test_metrics["recall_weighted"]), 6),
        "test_f1_weighted": round(float(conv.test_metrics["f1_weighted"]), 6),
        "per_class_f1": conv.test_metrics["per_class_f1"],
        "confusion_matrix": conv.test_metrics["confusion_matrix"],
        "train_time_sec": round(float(conv.train_time_sec), 2),
        "best_epoch": conv.best_epoch,
        "peak_gpu_memory_mb": round(float(conv.peak_gpu_memory_mb), 1),
        "status": "success",
        "error_message": "",
    }


def run_one_dataset_fold(split: SplitData, model_variant: str, attention_type: str, attention_param: int, fold_idx: int, seed: int, device: torch.device, args) -> List[Dict]:
    print(
        f"  Classes={len(split.class_names)} | "
        f"train={len(split.train_paths)} val={len(split.val_paths)} test={len(split.test_paths)}"
    )
    print(f"  Class order: {split.class_names}")

    print(f"  Training {model_variant} with {attention_type} (param={attention_param})...")
    conv = train_convnext(split, seed, device, num_workers=args.num_workers, model_variant=model_variant, attention_type=attention_type, attention_param=attention_param)
    print(
        f"  {model_variant} done: val_acc={conv.val_metrics['accuracy']:.4f} "
        f"test_acc={conv.test_metrics['accuracy']:.4f} best_epoch={conv.best_epoch}"
    )

    row = build_run_row(
        make_run_id(split.dataset_name, model_variant, attention_type, attention_param, fold_idx),
        split, model_variant, attention_type, attention_param, fold_idx, seed, conv,
    )
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser(description="Definitive ConvNeXt 5-Fold Cross Validation Testing.")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help=f"Folds per combo. Default: {DEFAULT_FOLDS}.")
    parser.add_argument("--force", action="store_true", help="Delete existing CSV/state and rerun from scratch.")
    parser.add_argument("--num-workers", type=int, default=4, help="PyTorch DataLoader workers. Use 0 if Windows multiprocessing complains.")
    args = parser.parse_args()

    init_csvs(force=args.force)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("  Definitive ConvNeXt Testing - 5-Fold Cross Validation")
    print(f"  Script dir : {SCRIPT_DIR}")
    print(f"  Runs CSV   : {RUNS_CSV}")
    print(f"  Summary CSV: {SUMMARY_CSV}")
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"  Datasets   : {len(DATASET_CONFIGS)}")
    print(f"  Variants   : {len(MODEL_VARIANTS)} ({', '.join(MODEL_VARIANTS)})")
    print(f"  Combos/ds  : {len(ATTENTION_CONFIGS)}")
    print(f"  Folds      : {args.folds}")
    print("=" * 78)

    state = load_state()
    completed = set(state.get("completed", []))

    all_jobs: List[Tuple[Dict, str, str, int, int, int, str, SplitData]] = []
    
    # Pre-calculate data splits and folds to ensure deterministic behavior across identical seeds
    for cfg in DATASET_CONFIGS:
        cv_data = load_cv_data(cfg)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED_BASE)
        
        for model_variant in MODEL_VARIANTS:
            for att in ATTENTION_CONFIGS:
                for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(cv_data.paths, cv_data.labels), 1):
                    seed = SEED_BASE + fold_idx
                    run_id = make_run_id(cfg["name"], model_variant, att["type"], att["param"], fold_idx)
                    
                    train_val_paths = cv_data.paths[train_val_idx]
                    train_val_labels = cv_data.labels[train_val_idx]
                    test_paths = cv_data.paths[test_idx]
                    test_labels = cv_data.labels[test_idx]
                    
                    # Allocate 20% of the active fold's training pool to validation for early stopping
                    strat_labels = train_val_labels if can_stratify(train_val_labels) else None
                    train_paths, val_paths, train_labels, val_labels = train_test_split(
                        train_val_paths, 
                        train_val_labels, 
                        test_size=0.20, 
                        random_state=seed, 
                        stratify=strat_labels
                    )
                    
                    split = SplitData(
                        dataset_name=cv_data.dataset_name,
                        class_names=cv_data.class_names,
                        train_paths=train_paths.tolist(),
                        train_labels=train_labels,
                        val_paths=val_paths.tolist(),
                        val_labels=val_labels,
                        test_paths=test_paths.tolist(),
                        test_labels=test_labels,
                        image_size=cv_data.image_size
                    )
                    
                    all_jobs.append((cfg, model_variant, att["type"], att["param"], fold_idx, seed, run_id, split))

    pending = [job for job in all_jobs if job[6] not in completed]
    print(f"Total jobs: {len(all_jobs)} | completed: {len(all_jobs) - len(pending)} | pending: {len(pending)}")

    for job_i, (cfg, model_variant, att_type, att_param, fold_idx, seed, run_id, split) in enumerate(pending, 1):
        print("\n" + "━" * 78)
        print(f"[{job_i}/{len(pending)}] {cfg['name']} | Model: {model_variant} | Attention: {att_type} (p={att_param}) | Fold {fold_idx}/{args.folds} | seed={seed}")
        print("━" * 78)
        try:
            rows = run_one_dataset_fold(split, model_variant, att_type, att_param, fold_idx, seed, device, args)
            append_run_rows(rows)
            state.setdefault("completed", []).append(run_id)
            save_state(state)
            refresh_summary(args.folds)
        except Exception as exc:
            print(f"  ERROR: {cfg['name']} | {model_variant} | {att_type} | fold {fold_idx} failed: {exc}")
            err_row = {col: "" for col in RUN_COLUMNS}
            err_row.update({
                "run_id": run_id,
                "dataset": cfg["name"],
                "model_variant": model_variant,
                "attention_type": att_type,
                "attention_param": att_param,
                "fold_idx": fold_idx,
                "seed": seed,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "error",
                "error_message": traceback.format_exc()[:1200],
            })
            append_run_rows([err_row])
            refresh_summary(args.folds)

    refresh_summary(args.folds)
    print("\n" + "=" * 78)
    print("Done.")
    print(f"Per-fold results : {RUNS_CSV}")
    print(f"Summary results : {SUMMARY_CSV}")
    print("=" * 78)

if __name__ == "__main__":
    main()