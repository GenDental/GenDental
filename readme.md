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
