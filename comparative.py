# Authors: Morteza Farrokhnejad, Ali Hakan Ulusoy, Ahmet Rizaner
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Import the untouched pipeline from kfoldtesting.py (same directory)
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfoldtesting import (  # noqa: E402
    DATASET_CONFIGS,
    CONVNEXT_HP,
    MODEL_VARIANTS,
    SEED_BASE,
    SCRIPT_DIR, #CHANGE THIS
    SplitData,
    CVDataset,
    ConvNeXtOutputs,
    load_cv_data,
    can_stratify,
    train_convnext,
    ensure_dir,
)

# ─────────────────────────────────────────────────────────────────────────────
# Output locations (separate files -- original kfoldtesting.py CSVs untouched)
# ─────────────────────────────────────────────────────────────────────────────

RUNS_CSV = SCRIPT_DIR / "convnext_variation_results_runs.csv"
SUMMARY_CSV = SCRIPT_DIR / "convnext_variation_results_summary.csv"
STATE_PATH = SCRIPT_DIR / "convnext_variation_run_state.json"

DEFAULT_RUNS = 5

# ─────────────────────────────────────────────────────────────────────────────
# The 3 requested variations
# ─────────────────────────────────────────────────────────────────────────────
# cv_mode:
#   "repeated_holdout" -> 5 independent stratified resamples (same split
#                          ratios as the pipeline already uses), NOT disjoint
#   "stratified_kfold"  -> 5 disjoint folds via StratifiedKFold, exactly the
#                          construction kfoldtesting.py's main() already does
#
# To flip a variation to strict k-fold instead of repeated holdout, just
# change its "cv_mode" value below -- nothing else needs to change.

VARIATIONS = [
    {
        "variation_id": 1,
        "variation_name": "V1_no_attention",
        "attention_type": "none",
        "attention_param": 0,
        "cv_mode": "repeated_holdout",
        "seed_offset": 100,
    },
    {
        "variation_id": 2,
        "variation_name": "V2_SE_attention",
        "attention_type": "SE",
        "attention_param": 8,
        "cv_mode": "repeated_holdout",
        "seed_offset": 200,
    },
    {
        "variation_id": 3,
        "variation_name": "V3_SE_attention_kfold",
        "attention_type": "SE",
        "attention_param": 8,
        "cv_mode": "stratified_kfold",
        "seed_offset": 0,  # matches kfoldtesting.py's own seed_base exactly
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CSV columns
# ─────────────────────────────────────────────────────────────────────────────

RUN_COLUMNS = [
    "run_id", "dataset", "variation_id", "variation_name", "cv_mode",
    "model_variant", "attention_type", "attention_param", "run_idx", "seed", "timestamp",
    "num_classes", "class_names", "num_train", "num_val", "num_test",
    "val_accuracy", "val_f1_macro",
    "test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro",
    "test_precision_weighted", "test_recall_weighted", "test_f1_weighted",
    "per_class_f1", "confusion_matrix",
    "train_time_sec", "best_epoch", "peak_gpu_memory_mb",
    "status", "error_message",
]

SUMMARY_COLUMNS = [
    "dataset", "variation_id", "variation_name", "cv_mode",
    "model_variant", "attention_type", "attention_param",
    "runs_completed", "runs_expected",
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
# State / CSV helpers (same pattern kfoldtesting.py already uses)
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> Dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": []}

def save_state(state: Dict) -> None:
    ensure_dir(SCRIPT_DIR)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def make_run_id(dataset_name: str, variation_name: str, run_idx: int) -> str:
    return f"{dataset_name}|{variation_name}|run{run_idx}"

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

def refresh_summary(runs_expected: int) -> None:
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    if not RUNS_CSV.exists():
        return

    with open(RUNS_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "success":
                continue
            key = (row["dataset"], row["variation_id"])
            groups.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []
    for (dataset, variation_id), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        first = rows[0]

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
            "variation_id": variation_id,
            "variation_name": first["variation_name"],
            "cv_mode": first["cv_mode"],
            "model_variant": first["model_variant"],
            "attention_type": first["attention_type"],
            "attention_param": first["attention_param"],
            "runs_completed": len(rows),
            "runs_expected": runs_expected,
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
# Split builders
# ─────────────────────────────────────────────────────────────────────────────
# Both builders reuse the exact same split calls (train_test_split /
# StratifiedKFold, same ratios, same stratification guard via can_stratify)
# that kfoldtesting.py's main() already uses -- nothing about how a split is
# carved out of the data has changed, only how many independent times it's
# done and whether the test folds are forced to be disjoint.

def build_kfold_splits(cv_data: CVDataset, n_runs: int, seed_base: int) -> List[Tuple[int, int, SplitData]]:
    """Identical construction to kfoldtesting.py main(): disjoint stratified folds."""
    skf = StratifiedKFold(n_splits=n_runs, shuffle=True, random_state=seed_base)
    out: List[Tuple[int, int, SplitData]] = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(cv_data.paths, cv_data.labels), 1):
        seed = seed_base + fold_idx

        train_val_paths = cv_data.paths[train_val_idx]
        train_val_labels = cv_data.labels[train_val_idx]
        test_paths = cv_data.paths[test_idx]
        test_labels = cv_data.labels[test_idx]

        strat_labels = train_val_labels if can_stratify(train_val_labels) else None
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=0.20, random_state=seed, stratify=strat_labels,
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
            image_size=cv_data.image_size,
        )
        out.append((fold_idx, seed, split))
    return out

def build_repeated_holdout_splits(cv_data: CVDataset, n_runs: int, seed_base: int) -> List[Tuple[int, int, SplitData]]:
    """5 independent stratified resamples using the same 80/20 -> 80/20 ratios
    the pipeline already uses per fold. Not a disjoint partition: each run
    draws its own random stratified test set from the full pool."""
    out: List[Tuple[int, int, SplitData]] = []
    for run_idx in range(1, n_runs + 1):
        seed = seed_base + run_idx

        strat_all = cv_data.labels if can_stratify(cv_data.labels) else None
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            cv_data.paths, cv_data.labels,
            test_size=0.20, random_state=seed, stratify=strat_all,
        )

        strat_tv = train_val_labels if can_stratify(train_val_labels) else None
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=0.20, random_state=seed, stratify=strat_tv,
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
            image_size=cv_data.image_size,
        )
        out.append((run_idx, seed, split))
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Run orchestration
# ─────────────────────────────────────────────────────────────────────────────

def build_run_row(
    run_id: str,
    split: SplitData,
    variation: Dict,
    model_variant: str,
    run_idx: int,
    seed: int,
    conv: ConvNeXtOutputs,
) -> Dict:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "run_id": run_id,
        "dataset": split.dataset_name,
        "variation_id": variation["variation_id"],
        "variation_name": variation["variation_name"],
        "cv_mode": variation["cv_mode"],
        "model_variant": model_variant,
        "attention_type": variation["attention_type"],
        "attention_param": variation["attention_param"],
        "run_idx": run_idx,
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

def run_one(split: SplitData, variation: Dict, model_variant: str, run_idx: int, seed: int, device: torch.device, args) -> List[Dict]:
    print(
        f"  Classes={len(split.class_names)} | "
        f"train={len(split.train_paths)} val={len(split.val_paths)} test={len(split.test_paths)}"
    )
    print(f"  Class order: {split.class_names}")
    print(f"  Training {model_variant} | {variation['variation_name']} (attention={variation['attention_type']})...")

    # Unmodified pipeline call -- identical to what kfoldtesting.py itself calls.
    conv = train_convnext(
        split, seed, device,
        num_workers=args.num_workers,
        model_variant=model_variant,
        attention_type=variation["attention_type"],
        attention_param=variation["attention_param"],
    )
    print(
        f"  {model_variant} done: val_acc={conv.val_metrics['accuracy']:.4f} "
        f"test_acc={conv.test_metrics['accuracy']:.4f} best_epoch={conv.best_epoch}"
    )

    row = build_run_row(
        make_run_id(split.dataset_name, variation["variation_name"], run_idx),
        split, variation, model_variant, run_idx, seed, conv,
    )
    return [row]

def main() -> None:
    parser = argparse.ArgumentParser(description="ConvNeXt ablation: 3 variations x 5 runs x 5 datasets.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"Runs per dataset x variation. Default: {DEFAULT_RUNS}.")
    parser.add_argument("--force", action="store_true", help="Delete existing CSV/state and rerun from scratch.")
    parser.add_argument("--num-workers", type=int, default=4, help="PyTorch DataLoader workers. Use 0 if Windows multiprocessing complains.")
    args = parser.parse_args()

    init_csvs(force=args.force)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("  ConvNeXt Ablation - 3 Variations x 5 Runs x 5 Datasets")
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
    print(f"  Variations : {len(VARIATIONS)} ({', '.join(v['variation_name'] for v in VARIATIONS)})")
    print(f"  Runs/combo : {args.runs}")
    print("=" * 78)

    state = load_state()
    completed = set(state.get("completed", []))

    all_jobs: List[Tuple[Dict, Dict, str, int, int, str, SplitData]] = []

    # Load each dataset's image pool once, then build splits per variation.
    for cfg in DATASET_CONFIGS:
        cv_data = load_cv_data(cfg)

        for model_variant in MODEL_VARIANTS:
            for variation in VARIATIONS:
                seed_base = SEED_BASE + variation["seed_offset"]

                if variation["cv_mode"] == "stratified_kfold":
                    splits = build_kfold_splits(cv_data, args.runs, seed_base)
                else:
                    splits = build_repeated_holdout_splits(cv_data, args.runs, seed_base)

                for run_idx, seed, split in splits:
                    run_id = make_run_id(cfg["name"], variation["variation_name"], run_idx)
                    all_jobs.append((cfg, variation, model_variant, run_idx, seed, run_id, split))

    pending = [job for job in all_jobs if job[5] not in completed]
    print(f"Total jobs: {len(all_jobs)} | completed: {len(all_jobs) - len(pending)} | pending: {len(pending)}")

    for job_i, (cfg, variation, model_variant, run_idx, seed, run_id, split) in enumerate(pending, 1):
        print("\n" + "━" * 78)
        print(
            f"[{job_i}/{len(pending)}] {cfg['name']} | {variation['variation_name']} "
            f"(attention={variation['attention_type']}, cv_mode={variation['cv_mode']}) | "
            f"run {run_idx}/{args.runs} | seed={seed}"
        )
        print("━" * 78)
        try:
            rows = run_one(split, variation, model_variant, run_idx, seed, device, args)
            append_run_rows(rows)
            state.setdefault("completed", []).append(run_id)
            save_state(state)
            refresh_summary(args.runs)
        except Exception as exc:
            print(f"  ERROR: {cfg['name']} | {variation['variation_name']} | run {run_idx} failed: {exc}")
            err_row = {col: "" for col in RUN_COLUMNS}
            err_row.update({
                "run_id": run_id,
                "dataset": cfg["name"],
                "variation_id": variation["variation_id"],
                "variation_name": variation["variation_name"],
                "cv_mode": variation["cv_mode"],
                "model_variant": model_variant,
                "attention_type": variation["attention_type"],
                "attention_param": variation["attention_param"],
                "run_idx": run_idx,
                "seed": seed,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "error",
                "error_message": traceback.format_exc()[:1200],
            })
            append_run_rows([err_row])
            refresh_summary(args.runs)

    refresh_summary(args.runs)
    print("\n" + "=" * 78)
    print("Done.")
    print(f"Per-run results : {RUNS_CSV}")
    print(f"Summary results : {SUMMARY_CSV}")
    print("=" * 78)

if __name__ == "__main__":
    main()