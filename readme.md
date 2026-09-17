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
 bash scripts/sample_stage_one.sh
 ```

The configurable options are:

- `CKPT_PATH`: Path to the trained Stage I checkpoint used for generation.
- `OUTPUT_DIR`: Directory for saving the generated synthetic samples.
- `NUM_SAMPLES`: Number of synthetic samples to generate.
- `BATCH_SIZE`: Batch size used during sampling.
- `SAVE_MERGED`: Whether to save merged synthetic data (`true` or `false`).



## Stage II

To train Stage II:

```shell
bash scripts/train_stage_two.sh
```



To get synthetic pre-orthodontic data:

```shell
bash scripts/sample_stage_two.sh
```

For each generated sample, it randomly selects one style NPZ
 from `STYLE_DIR` and generates one synthetic pre-orthodontic sample.
 Therefore, the number of output samples always equals the number of samples
 in `DATA_DIR`.

The configurable options are:

- `TASK_MODE`: Generation mode, including `target` and `motion`. The
   corresponding checkpoint is selected automatically if `CKPT_PATH` is not
   specified.
- `CKPT_PATH`: Path to the Stage II checkpoint used for generation.
- `STYLE_DIR`: Directory containing reference style NPZ files.
- `DATA_DIR`: Directory containing Stage I generated samples.
- `OUTPUT_DIR`: Directory for saving generated pre-orthodontic samples.
- `BATCH_SIZE`: Sampling batch size.
- `SEED`: Random seed for reproducible style selection.

Style files must be NPZ files containing `before_pts` and `mask`. Stage I now
 saves structured NPZ files alongside each merged PLY file. Stage II prefers
 these NPZ files because they preserve all 32 anatomical tooth slots. Each
 output NPZ records the selected `style_id`.



## Quick Start with Pretrained Models

We provide pretrained checkpoints for both Stage I and Stage II to quickly
reproduce the data generation pipeline without training the models from
scratch.

### 1. Download Checkpoints

Download the pretrained checkpoints from:

[Google Drive](https://drive.google.com/drive/folders/12ZxKM3uC3xg4g38XZWImZYlvG8rga7OL?usp=drive_link)

Please place the downloaded checkpoints at the corresponding paths or specify
the checkpoint paths through the `CKPT_PATH` option in the sampling scripts.

### 2. Generate Synthetic Post-Orthodontic Data

Run Stage I sampling:

    bash scripts/sample_stage_one.sh

The generated synthetic post-orthodontic samples will be saved to:

    ./stage_one_samples

### 3. Generate Synthetic Pre-Orthodontic Data

The repository provides reference style data in:

    ./reference_data

Run Stage II sampling with the generated Stage I samples as input:

    STYLE_DIR=./reference_data \
    DATA_DIR=./stage_one_samples \
    bash scripts/sample_stage_two.sh

The generated synthetic pre-orthodontic samples will be saved to:

    ./stage_two_samples

### 4. Visualization

To visualize the generated samples, run:

    python vis.py

The visualization results will be saved to:

    ./stage_two_visualizations

