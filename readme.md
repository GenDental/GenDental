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

you need to modify the data path in configuration files.



To get synthetic post-orthodontic data:

 ```shell
 bash scripts/test_stage_one.sh
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

