#!/usr/bin/env python

import argparse
from pathlib import Path
import zipfile
import shutil

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a minimal local MMEB snapshot for VS Code debug runs.")
    parser.add_argument("--output-dir", default="data", help="Base directory used by --data_basedir in launch.json.")
    parser.add_argument("--subset", default="HatefulMemes", help="MMEB-train subset to mirror locally.")
    parser.add_argument("--split", default="original", help="Dataset split to export.")
    parser.add_argument("--rows", type=int, default=32, help="Number of rows to keep in the local parquet.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    dataset_root = output_dir / "vlm2vec_debug" / "MMEB-train"
    subset_dir = dataset_root / args.subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("TIGER-Lab/MMEB-train", args.subset, split=f"{args.split}[:{args.rows}]")
    parquet_path = subset_dir / f"{args.split}-debug.parquet"
    dataset.to_parquet(str(parquet_path))

    image_paths = sorted(
        {
            row[path_key]
            for row in dataset
            for path_key in ("qry_image_path", "pos_image_path", "neg_image_path")
            if path_key in row and row[path_key]
        }
    )

    archive_path = hf_hub_download(
        repo_id="TIGER-Lab/MMEB-train",
        repo_type="dataset",
        filename=f"images_zip/{args.subset}.zip",
    )

    with zipfile.ZipFile(archive_path) as zf:
        archive_members = {
            f"images/{member}": member
            for member in zf.namelist()
            if member and not member.endswith("/") and not member.startswith("__MACOSX/") and "/._" not in member
        }
        missing = [path for path in image_paths if path not in archive_members]
        if missing:
            missing_preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing {len(missing)} images in archive: {missing_preview}")
        for image_path in image_paths:
            archive_member = archive_members[image_path]
            target_path = dataset_root / image_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(archive_member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    print(f"Wrote parquet: {parquet_path}")
    print(f"Extracted {len(image_paths)} images under: {dataset_root}")


if __name__ == "__main__":
    main()
