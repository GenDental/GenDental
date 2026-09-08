"""Visualize before/after states from Stage II NPZ samples."""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render side-by-side before/after point clouds from "
            "stage_two_samples."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data3/leics/dataset/teeth/merged_alignment_5x"),
        help="Directory containing Stage II NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stage_two_visualizations"),
        help="Directory for rendered PNG files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="Number of samples to render. Use 0 to render all.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional NPZ stems to render, for example: 0 10 25.",
    )
    parser.add_argument(
        "--max-points-per-tooth",
        type=int,
        default=512,
        help="Maximum rendered points per valid tooth.",
    )
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--azimuth", type=float, default=-70.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def natural_key(path: Path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def load_sample(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as sample:
        missing = {
            key
            for key in ("before_pts", "after_pts", "mask")
            if key not in sample
        }
        if missing:
            raise KeyError(f"{path} is missing fields: {sorted(missing)}")
        before_points = sample["before_pts"].astype(np.float32)
        after_points = sample["after_pts"].astype(np.float32)
        masks = sample["mask"].astype(bool)

    if before_points.ndim != 3 or before_points.shape[-1] != 3:
        raise ValueError(
            f"{path}: before_pts must have shape [N,P,3], "
            f"got {before_points.shape}."
        )
    if after_points.shape != before_points.shape:
        raise ValueError(
            f"{path}: after_pts shape {after_points.shape} does not match "
            f"before_pts shape {before_points.shape}."
        )
    if masks.shape != (before_points.shape[0],):
        raise ValueError(
            f"{path}: mask must have shape {(before_points.shape[0],)}, "
            f"got {masks.shape}."
        )
    if not masks.any():
        raise ValueError(f"{path}: mask contains no valid teeth.")
    return before_points, after_points, masks


def axis_limits(
    before_points: np.ndarray,
    after_points: np.ndarray,
    masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(
        [before_points[masks].reshape(-1, 3), after_points[masks].reshape(-1, 3)],
        axis=0,
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float((maximum - minimum).max()) / 2.0, 1e-6) * 1.05
    return center - radius, center + radius


def draw_state(
    axis,
    points: np.ndarray,
    masks: np.ndarray,
    title: str,
    max_points_per_tooth: int,
    lower: np.ndarray,
    upper: np.ndarray,
    elevation: float,
    azimuth: float,
) -> None:
    color_map = plt.get_cmap("turbo", points.shape[0])
    for tooth_index in range(points.shape[0]):
        if not masks[tooth_index]:
            continue
        tooth_points = points[tooth_index]
        if len(tooth_points) > max_points_per_tooth:
            indexes = np.linspace(
                0,
                len(tooth_points) - 1,
                max_points_per_tooth,
                dtype=np.int64,
            )
            tooth_points = tooth_points[indexes]
        axis.scatter(
            tooth_points[:, 0],
            tooth_points[:, 1],
            tooth_points[:, 2],
            s=0.7,
            color=color_map(tooth_index),
            linewidths=0,
            depthshade=False,
        )

    axis.set_title(title)
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()


def render_sample(
    input_path: Path,
    output_path: Path,
    max_points_per_tooth: int,
    elevation: float,
    azimuth: float,
    dpi: int,
) -> None:
    before_points, after_points, masks = load_sample(input_path)
    lower, upper = axis_limits(before_points, after_points, masks)
    figure = plt.figure(figsize=(12, 6), constrained_layout=True)
    before_axis = figure.add_subplot(1, 2, 1, projection="3d")
    after_axis = figure.add_subplot(1, 2, 2, projection="3d")

    draw_state(
        before_axis,
        before_points,
        masks,
        "Before",
        max_points_per_tooth,
        lower,
        upper,
        elevation,
        azimuth,
    )
    draw_state(
        after_axis,
        after_points,
        masks,
        "After",
        max_points_per_tooth,
        lower,
        upper,
        elevation,
        azimuth,
    )
    before_centers = before_points.mean(axis=1)
    after_centers = after_points.mean(axis=1)
    center_motion = np.linalg.norm(
        after_centers[masks] - before_centers[masks],
        axis=-1,
    )
    figure.suptitle(
        f"Sample {input_path.stem} | valid teeth: {int(masks.sum())} | "
        f"center motion mean/max: "
        f"{center_motion.mean():.4f}/{center_motion.max():.4f}"
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = get_args()
    if args.max_points_per_tooth <= 0:
        raise ValueError("--max-points-per-tooth must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if args.num_samples < 0:
        raise ValueError("--num-samples must be non-negative.")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    sample_paths = sorted(input_dir.glob("*.npz"), key=natural_key)
    if not sample_paths:
        raise FileNotFoundError(f"No NPZ samples found in: {input_dir}")

    if args.sample_ids:
        requested = {str(sample_id) for sample_id in args.sample_ids}
        sample_paths = [
            path for path in sample_paths if path.stem in requested
        ]
        missing = sorted(requested - {path.stem for path in sample_paths})
        if missing:
            raise FileNotFoundError(
                f"Requested sample IDs not found: {missing}"
            )
    elif args.num_samples:
        sample_paths = sample_paths[: args.num_samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(sample_paths)} samples to {output_dir}")

    for sample_path in sample_paths:
        output_path = output_dir / f"{sample_path.stem}_before_after.png"
        render_sample(
            sample_path,
            output_path,
            args.max_points_per_tooth,
            args.elevation,
            args.azimuth,
            args.dpi,
        )
        print(output_path)


if __name__ == "__main__":
    main()
