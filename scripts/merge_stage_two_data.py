"""Merge real samples with generated Stage-II samples at ratios 0..5."""

import argparse
import re
import shutil
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-data-dir", type=Path, required=True)
    p.add_argument("--real-data-dir", type=Path, default=Path(
        "/data3/leics/dataset/teeth/sample512_merged"))
    p.add_argument("--generated-data-dir", type=Path, default=Path(
        "/data3/leics/dataset/GenDental/stage_two_samples"))
    p.add_argument("--source-index-dir", type=Path, default=Path("files/zj"))
    p.add_argument("--output-index-dir", type=Path,
                   default=Path("files/target_ratios"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", path.stem)]


def load_index(path):
    values = np.asarray(np.load(path))
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{path} must contain a 1-D integer array")
    return values


def main():
    args = parse_args()
    real_dir = args.real_data_dir.expanduser().resolve()
    generated_dir = args.generated_data_dir.expanduser().resolve()
    source_index_dir = args.source_index_dir.expanduser().resolve()
    output_data_dir = args.output_data_dir.expanduser().resolve()
    output_index_dir = args.output_index_dir.expanduser().resolve()

    for path in (real_dir, generated_dir, source_index_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")
    if output_data_dir in (real_dir, generated_dir):
        raise ValueError("--output-data-dir must be a separate directory")

    original_train = load_index(source_index_dir / "train.npy")
    original_test = load_index(source_index_dir / "test.npy")
    real_train = original_train[original_train != 145]
    if len(real_train) == 0:
        raise ValueError("No train indexes remain after excluding 145")

    all_real = np.unique(np.concatenate((real_train, original_test)))
    needed = 5 * len(real_train)
    generated_files = sorted(
        generated_dir.glob("*.npz"), key=natural_key)
    if len(generated_files) < needed:
        raise ValueError(
            f"Need {needed} generated NPZ files, found {len(generated_files)}")
    generated_files = generated_files[:needed]

    start = int(all_real.max()) + 1
    generated_indexes = np.arange(
        start, start + needed, dtype=original_train.dtype)

    copies = [
        (real_dir / f"{int(index)}.npz",
         output_data_dir / f"{int(index)}.npz")
        for index in all_real
    ]
    copies.extend(
        (source, output_data_dir / f"{int(index)}.npz")
        for source, index in zip(generated_files, generated_indexes)
    )
    index_outputs = [
        output_index_dir / f"train{ratio}.npy" for ratio in range(6)
    ] + [output_index_dir / "test.npy"]

    # Validate everything before copying any data.
    for source, destination in copies:
        if not source.is_file():
            raise FileNotFoundError(f"Missing source sample: {source}")
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"{destination} exists; pass --overwrite to replace it")
    for destination in index_outputs:
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"{destination} exists; pass --overwrite to replace it")

    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_index_dir.mkdir(parents=True, exist_ok=True)
    for source, destination in copies:
        shutil.copy2(source, destination)

    # trainX = 1x real training data + Xx generated data.
    for ratio in range(6):
        generated_count = ratio * len(real_train)
        train_indexes = np.concatenate(
            (real_train, generated_indexes[:generated_count]))
        np.save(output_index_dir / f"train{ratio}.npy", train_indexes)

    # Copy test.npy byte-for-byte; it remains the original real-only split.
    shutil.copy2(
        source_index_dir / "test.npy",
        output_index_dir / "test.npy",
    )

    print(f"real train samples: {len(real_train)}")
    print(f"generated samples: {needed}")
    print(f"generated index range: {generated_indexes[0]}..{generated_indexes[-1]}")
    print(f"merged data: {output_data_dir}")
    print(f"ratio indexes: {output_index_dir}")


if __name__ == "__main__":
    main()
