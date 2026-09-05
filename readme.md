# 3D Generated Data Improve AI Models in Digital Orthodontics

This is the PyTorch implementation of our paper *"3D Generated Data Improve AI Models in Digital Orthodontics."*

### Installation

First create a conda environment:

```shell
conda create --name gendental
conda activate gendental
```

Pytorch / Python combination that was verified to work is:

- Python 3.10, Pytorch 2.3.1, CUDA 11.8

To install python requirements:

```shell
pip install -r requirements.txt
```



## Dataset

To ensure reproducibility and ease of use, we standardize the input data as `.npz` files. Each patient/case should be saved as an individual file.

### Directory Structure

Organize your dataset directory as follows:

```
/path/to/your/dataset/
├── 1.npz
├── 2.npz
└── ...
```

### Data Schema

Each `.npz` file contains point cloud data and transformation metadata. The requirements differ between tooth arrangement data and stage prediction data.

#### A. Tooth Arrangement

Required for initial and final states. We use N=32(tooth number) and P=512 (points per tooth).

| **Key**      | **Type** | **Shape** | **Description**                                              |
| ------------ | -------- | --------- | ------------------------------------------------------------ |
| `before_pts` | float32  | (N, P, 3) | Point cloud coordinates of the pre-treatment (initial) state. |
| `after_pts`  | float32  | (N, P, 3) | Point cloud coordinates of the post-treatment (target) state. |
| `mask`       | int32    | (N,)      | Binary mask indicating valid tooth units (1 for valid, 0 for missing/invalid). |



#### B.Stage prediction

Required for initial and final states, as well as the orthodontic trajectory. We use L=20.

| **Key**        | **Type** | **Shape**    | **Description**                                              |
| -------------- | -------- | ------------ | ------------------------------------------------------------ |
| `before_pts`   | float32  | (N, P, 3)    | Point cloud coordinates of the pre-treatment (initial) state. |
| `after_pts`    | float32  | (N, P, 3)    | Point cloud coordinates of the post-treatment (target) state. |
| `mask`         | int32    | (N,)         | Binary mask indicating valid tooth units (1 for valid, 0 for missing/invalid). |
| `matrices`     | float32  | (L, N, 4, 4) | Transformation matrices from **initial** to each intermediate step $S$. |
| `inv_matrices` | float32  | (L, N, 4, 4) | Inverse transformation matrices from **final** to each intermediate step $S$. |



## Stage I

To train Stage I:

```shell
bash scripts/train_stage_one.sh
```

Runtime paths are configured in the script and can be overridden without
editing YAML, for example:

```shell
DATA_PATH=/path/to/data INDEX_PATH=/path/to/splits \
OUTPUT_DIR=/path/to/checkpoints bash scripts/train_stage_one.sh
```



To get synthetic post-orthodontic data:

 ```shell
 bash scripts/test_stage_one.sh
 ```

Stage I generation does not load a dataset:

```shell
CKPT_PATH=/path/to/ckpt OUTPUT_DIR=/path/to/generated \
NUM_SAMPLES=100 BATCH_SIZE=8 bash scripts/test_stage_one.sh
```



## Stage II

To train Stage II:

```shell
bash scripts/train_stage_two.sh
```



To get synthetic pre-orthodontic data:

 ```shell
bash scripts/sample_stage_two.sh
 ```

Stage II sampling reads directories directly and does not construct a
DataLoader. For every file in gpt_samples it randomly selects one style NPZ
from reference_data, so the output count always equals the data count:

```shell
STYLE_DIR=/path/to/reference_npz \
DATA_DIR=/path/to/stage_one_samples \
OUTPUT_DIR=/path/to/output \
bash scripts/sample_stage_two.sh
```

Style files must be NPZ files containing before_pts and mask. Stage I now
saves a structured NPZ beside each merged PLY; Stage II prefers this NPZ
because it preserves all 32 anatomical tooth slots. Set SEED to make style
selection reproducible; each output NPZ records the selected style_id.

## Configuration overrides

The entry point has explicit train, test, and generate modes. Any YAML field
can be overridden repeatedly from a script:

```shell
python main.py --mode train --config configs/stage_one.yaml \
  --set dataset.params.data_path=/path/to/data \
  --set dataset.params.batch_size=8
```

Generate builds only the model and checkpoint; train and test also build the
configured DataModule. The old test and sample flags remain supported.

## Sampling Guide

This section describes the complete two-stage sampling workflow. Both sampling
scripts use generate mode and do not construct a DataLoader.

### Stage I sampling

Stage I generates synthetic post-treatment tooth point clouds:

```shell
bash scripts/test_stage_one.sh
```

The commonly changed options can be provided as environment variables:

```shell
CUDA_VISIBLE_DEVICES=0 \
CKPT_PATH=/path/to/stage_one/ckpt \
OUTPUT_DIR=gpt_samples \
NUM_SAMPLES=100 \
BATCH_SIZE=8 \
SAVE_MERGED=true \
bash scripts/test_stage_one.sh
```

The checkpoint may be either an exact `.ckpt` file or a checkpoint
directory containing `last.ckpt`. For each generated sample, Stage I
writes:

- `<id>.ply`: all predicted valid teeth merged into one point cloud.
- `<id>.npz`: the preferred structured representation containing
  `after_pts[32,512,3]` and `mask[32]`.

The NPZ representation preserves anatomical tooth slots and should be used as
the Stage II data input whenever available. A legacy merged PLY can still be
used for a quick test, but missing tooth positions cannot be recovered exactly.

Example Stage I output:

```text
gpt_samples/
├── 0.npz
├── 0.ply
├── 1.npz
├── 1.ply
└── ...
```

### The purpose of reference_data

`reference_data` is the Stage II style pool. Each NPZ supplies a real
pre-treatment reference whose geometry and valid-tooth pattern guide the
generated pre-treatment state.

Every reference NPZ must contain at least:

| Key | Shape | Purpose |
| --- | --- | --- |
| `before_pts` | `(32, 512, 3)` | Pre-treatment tooth geometry used as the style reference. |
| `mask` | `(32,)` | Identifies valid and missing teeth in the style reference. |

The style pool and Stage I data directory do not need to contain the same
number of files and their filenames do not need to match. For every data
sample, Stage II independently selects one style NPZ at random, with
replacement. Therefore:

- the number of Stage II outputs always equals the number of data samples;
- one style may be selected for multiple data samples;
- unused styles are allowed;
- setting the same `SEED` reproduces the same style assignments.

Example style directory:

```text
reference_data/
├── 0.npz
├── 1.npz
└── ...
```

### Stage II sampling

By default, Stage II reads styles from `reference_data`, reads
post-treatment data from `gpt_samples`, and writes results to
`stage_two_samples`:

```shell
bash scripts/sample_stage_two.sh
```

Custom directories and sampling settings can be supplied without editing the
YAML configuration:

```shell
CUDA_VISIBLE_DEVICES=0 \
CKPT_PATH=/path/to/stage_two/ckpt \
STYLE_DIR=reference_data \
DATA_DIR=gpt_samples \
OUTPUT_DIR=stage_two_samples \
BATCH_SIZE=1 \
SEED=3407 \
bash scripts/sample_stage_two.sh
```

Stage II processes every `.npz` or `.ply` data sample. If
both formats have the same stem, for example `0.npz` and
`0.ply`, they represent one sample and the structured NPZ is
preferred. Output filenames follow the data filenames, not the randomly
selected style filenames.

Each Stage II output NPZ contains:

| Key | Shape | Description |
| --- | --- | --- |
| `before_pts` | `(32, 512, 3)` | Generated pre-treatment state. |
| `after_pts` | `(32, 512, 3)` | Stage I post-treatment input. |
| `style_pts` | `(32, 512, 3)` | Randomly selected reference style. |
| `mask` | `(32,)` | Valid-tooth mask from the selected style. |
| `style_id` | scalar string | Filename stem of the selected style NPZ. |
| `matrices` | `(21, 32, 4, 4)` | Relative treatment transformations. |
| `inv_matrices` | `(21, 32, 4, 4)` | Predicted inverse transformations. |

### Visualizing Stage II samples

Run the visualization script after Stage II sampling:

```shell
python vis.py
```

It reads `stage_two_samples/*.npz` by default and saves side-by-side
`<id>_before_after.png` images in the current directory. Custom paths
and view settings are also supported:

```shell
python vis.py \
  --input-dir stage_two_samples \
  --output-dir . \
  --elevation 20 \
  --azimuth -70 \
  --dpi 180
```
