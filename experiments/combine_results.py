"""
Combine evaluation results from multiple checkpoints into a single table.

Usage:
    python experiments/combine_results.py --eval_dir outputs/evaluation
    python experiments/combine_results.py --eval_dir outputs/evaluation --metric hit@1
    python experiments/combine_results.py --eval_dir outputs/evaluation --output results.csv
"""

import argparse
import json
import os
import csv
from pathlib import Path

# Task → category mapping for grouping
TASK_CATEGORIES = {
    # Image Classification
    "ImageNet-1K":   "Image CLS",
    "ImageNet-A":    "Image CLS",
    "ImageNet-R":    "Image CLS",
    "ObjectNet":     "Image CLS",
    "N24News":       "Image CLS",
    "HatefulMemes":  "Image CLS",
    "VOC2007":       "Image CLS",
    "SUN397":        "Image CLS",
    "Place365":      "Image CLS",
    "Country211":    "Image CLS",
    # Image QA
    "OK-VQA":           "Image QA",
    "A-OKVQA":          "Image QA",
    "DocVQA":           "Image QA",
    "InfographicsVQA":  "Image QA",
    "ChartQA":          "Image QA",
    "Visual7W":         "Image QA",
    "ScienceQA":        "Image QA",
    "VizWiz":           "Image QA",
    "GQA":              "Image QA",
    "TextVQA":          "Image QA",
    # Image Retrieval (I→T)
    "MSCOCO_i2t":       "Image Ret I→T",
    "VisualNews_i2t":   "Image Ret I→T",
    # Image Retrieval (T→I)
    "MSCOCO_t2i":       "Image Ret T→I",
    "VisualNews_t2i":   "Image Ret T→I",
    "VisDial":          "Image Ret T→I",
    "WebQA":            "Image Ret T→I",
    "EDIS":             "Image Ret T→I",
    "Wiki-SS-NQ":       "Image Ret T→I",
    # Image Retrieval (I→I) / Visual Grounding
    "CIRR":                 "Image Ret I→I / VG",
    "NIGHTS":               "Image Ret I→I / VG",
    "OVEN":                 "Image Ret I→I / VG",
    "FashionIQ":            "Image Ret I→I / VG",
    "MSCOCO":               "Image Ret I→I / VG",
    "RefCOCO":              "Image Ret I→I / VG",
    "RefCOCO-Matching":     "Image Ret I→I / VG",
    "Visual7W-Pointing":    "Image Ret I→I / VG",
    # Video Classification
    "SmthSmthV2":   "Video CLS",
    "HMDB51":       "Video CLS",
    "UCF101":       "Video CLS",
    "K700":         "Video CLS",
    "Breakfast":    "Video CLS",
    # Video Retrieval
    "MSR-VTT":      "Video Ret",
    "MSVD":         "Video Ret",
    "DiDeMo":       "Video Ret",
    "VATEX":        "Video Ret",
    "YouCook2":     "Video Ret",
    # Moment Retrieval
    "QVHighlight":  "Moment Ret",
    "Charades-STA": "Moment Ret",
    "MomentSeeker": "Moment Ret",
    # Video QA
    "Video-MME":    "Video QA",
    "NExTQA":       "Video QA",
    "EgoSchema":    "Video QA",
    "MVBench":      "Video QA",
    "ActivityNetQA": "Video QA",
}

PRIMARY_METRIC = "hit@1"

def load_scores(eval_dir: Path, metric: str) -> dict[str, dict[str, float]]:
    """Returns {checkpoint_name: {dataset_name: score}}"""
    results = {}
    for ckpt_dir in sorted(eval_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        ckpt_name = ckpt_dir.name
        scores = {}
        for score_file in sorted(ckpt_dir.glob("*_score.json")):
            dataset = score_file.name.replace("_score.json", "")
            with open(score_file) as f:
                data = json.load(f)
            if metric in data:
                scores[dataset] = data[metric]
        if scores:
            results[ckpt_name] = scores
    return results


def build_table(results: dict, metric: str) -> tuple[list[str], list[str], dict]:
    """Returns (sorted checkpoints, sorted datasets, data dict)"""
    def _sort_key(name):
        suffix = name.split("-")[-1]
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)
    checkpoints = sorted(results.keys(), key=_sort_key)
    all_datasets = sorted(set(d for ckpt in results.values() for d in ckpt))
    return checkpoints, all_datasets, results


def category_avg(ckpt_scores: dict[str, float]) -> dict[str, float]:
    """Compute per-category average for one checkpoint."""
    cat_sums = {}
    cat_counts = {}
    for dataset, score in ckpt_scores.items():
        cat = TASK_CATEGORIES.get(dataset, "Other")
        cat_sums[cat] = cat_sums.get(cat, 0.0) + score
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    return {cat: cat_sums[cat] / cat_counts[cat] for cat in cat_sums}


def print_markdown_table(checkpoints, datasets, results, metric):
    header = f"| {'Dataset':<30} | {'Category':<22} | " + " | ".join(f"{c:<16}" for c in checkpoints) + " |"
    sep = f"| {'-'*30} | {'-'*22} | " + " | ".join(f"{'-'*16}" for _ in checkpoints) + " |"
    print(f"\n### Per-Dataset Results ({metric})\n")
    print(header)
    print(sep)

    prev_cat = None
    for ds in datasets:
        cat = TASK_CATEGORIES.get(ds, "Other")
        if cat != prev_cat and prev_cat is not None:
            print(f"| {'':<30} | {'':<22} | " + " | ".join(f"{'':<16}" for _ in checkpoints) + " |")
        prev_cat = cat
        row_vals = []
        for ckpt in checkpoints:
            val = results.get(ckpt, {}).get(ds)
            row_vals.append(f"{val:.4f}" if val is not None else "N/A")
        print(f"| {ds:<30} | {cat:<22} | " + " | ".join(f"{v:<16}" for v in row_vals) + " |")

    # Category averages
    print(f"\n### Category Averages ({metric})\n")
    all_cats = sorted(set(TASK_CATEGORIES.values()))
    header2 = f"| {'Category':<22} | " + " | ".join(f"{c:<16}" for c in checkpoints) + " |"
    sep2 = f"| {'-'*22} | " + " | ".join(f"{'-'*16}" for _ in checkpoints) + " |"
    print(header2)
    print(sep2)
    for cat in all_cats:
        row_vals = []
        for ckpt in checkpoints:
            ckpt_scores = results.get(ckpt, {})
            cat_scores = [v for ds, v in ckpt_scores.items() if TASK_CATEGORIES.get(ds) == cat]
            if cat_scores:
                row_vals.append(f"{sum(cat_scores)/len(cat_scores):.4f}")
            else:
                row_vals.append("N/A")
        print(f"| {cat:<22} | " + " | ".join(f"{v:<16}" for v in row_vals) + " |")

    # Overall average (across all tasks that appear in ALL checkpoints)
    common_datasets = set.intersection(*[set(results[c].keys()) for c in checkpoints]) if checkpoints else set()
    print(f"\n### Overall Average ({metric}, tasks present in all checkpoints)\n")
    header3 = f"| {'Overall Avg':<30} | " + " | ".join(f"{c:<16}" for c in checkpoints) + " |"
    sep3 = f"| {'-'*30} | " + " | ".join(f"{'-'*16}" for _ in checkpoints) + " |"
    print(header3)
    print(sep3)
    row_vals = []
    for ckpt in checkpoints:
        vals = [results[ckpt][ds] for ds in common_datasets if ds in results[ckpt]]
        row_vals.append(f"{sum(vals)/len(vals):.4f}" if vals else "N/A")
    print(f"| {'All tasks (common)':<30} | " + " | ".join(f"{v:<16}" for v in row_vals) + " |")


def write_csv(checkpoints, datasets, results, metric, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Category"] + checkpoints)
        for ds in datasets:
            cat = TASK_CATEGORIES.get(ds, "Other")
            row = [ds, cat]
            for ckpt in checkpoints:
                val = results.get(ckpt, {}).get(ds)
                row.append(f"{val:.4f}" if val is not None else "")
            writer.writerow(row)
        # Category averages
        writer.writerow([])
        writer.writerow(["--- Category Averages ---", ""] + [""] * len(checkpoints))
        all_cats = sorted(set(TASK_CATEGORIES.values()))
        for cat in all_cats:
            row = [cat, ""]
            for ckpt in checkpoints:
                ckpt_scores = results.get(ckpt, {})
                cat_scores = [v for ds, v in ckpt_scores.items() if TASK_CATEGORIES.get(ds) == cat]
                row.append(f"{sum(cat_scores)/len(cat_scores):.4f}" if cat_scores else "")
            writer.writerow(row)
        # Overall
        common_datasets = set.intersection(*[set(results[c].keys()) for c in checkpoints]) if checkpoints else set()
        row = ["Overall (common tasks)", ""]
        for ckpt in checkpoints:
            vals = [results[ckpt][ds] for ds in common_datasets if ds in results[ckpt]]
            row.append(f"{sum(vals)/len(vals):.4f}" if vals else "")
        writer.writerow(row)
    print(f"\nCSV written to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="outputs/evaluation", help="Directory with checkpoint subdirs")
    parser.add_argument("--metric", default=PRIMARY_METRIC, help="Metric key to extract (default: hit@1)")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    results = load_scores(eval_dir, args.metric)
    checkpoints, datasets, results = build_table(results, args.metric)

    print(f"Loaded {len(checkpoints)} checkpoints, {len(datasets)} unique datasets")
    print(f"Checkpoints: {checkpoints}")
    print(f"Metric: {args.metric}\n")

    print_markdown_table(checkpoints, datasets, results, args.metric)

    csv_path = args.output or str(eval_dir / f"combined_results_{args.metric.replace('@','_at_')}.csv")
    write_csv(checkpoints, datasets, results, args.metric, csv_path)


if __name__ == "__main__":
    main()
