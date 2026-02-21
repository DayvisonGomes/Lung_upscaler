# lung_upscaler

Repository containing all code for performing super-resolution on lung computed tomography (CT) images using an AutoencoderKL and Latent Diffusion Models (UNet-based).

This code is part of my dissertation and the paper:
**Importance of conditioning in latent diffusion models for image generation and super-resolution**

```bibtex
@article{10.1117/1.JMI.13.S1.S11203,
author = {Dayvison Gomes de Oliveira and Franklin Anthony Ramos Co{\^e}lho and Tha{\'i}s Gaudencio do R{\^e}go and Yuri de Almeida Malheiros Barbosa and Telmo de Menezes Silva Filho and Bruno Barufaldi},
title = {{Importance of conditioning in latent diffusion models for image generation and super-resolution}},
volume = {13},
journal = {Journal of Medical Imaging},
number = {S1},
publisher = {SPIE},
pages = {S11203},
keywords = {latent diffusion model, image generation, image super-resolution, computed tomography, artificial intelligence, Diffusion, Education and training, Data modeling, Image processing, Medical imaging, Image segmentation, Anatomy, Artificial intelligence, Computed tomography, Super resolution},
year = {2026},
doi = {10.1117/1.JMI.13.S1.S11203},
URL = {https://doi.org/10.1117/1.JMI.13.S1.S11203}
}
```

## Important notes

- This README provides a general workflow for preprocessing, training and evaluation. Parts of the repository contain legacy code kept for reference.
- Some scripts and utilities are not required for the main pipelines — they are kept as historical examples. Contact the author if in doubt.
- Scripts under `src/train/bash/` and `src/test/bash/` are examples intended for Docker usage; you can also work locally using a Conda/Miniconda environment and `requirements.txt`.
- Multiple configuration versions are available in `configs/` — pick the YAML file that best fits your experiment (e.g., `configs/ldm_cond_configs/`, `configs/aekl_configs/`).

## Essential files and functions

- `src/train/util_training.py`: training utilities, loops and validation.
- `src/train/util_transformations.py`: transformations and dataloader creation; the `get_upsampler_dataloader` function is particularly important and used by `src/train/train_aekl.py` and `src/train/train_ldm.py`.

## How to run (summary)

### 1. Recommended: use Conda/Miniconda to create the environment (example):

```bash
conda create -n lung_upscaler python=3.10 -y
conda activate lung_upscaler
pip install -r requirements.txt
```

### 2. Quick start with Docker (example):

```bash
docker build -t lung_upscaler -f Dockerfile .
docker run -it --ipc=host -v ${PWD}:/project/lung_upscaler --gpus all \
    DOCKER_IMAGE python ...
```

### 3. Example commands used (Windows paths — show flags and arguments):

```bash
python D:\Users\dayv\lung_upscaler\mamografIA_upscaler\src\train\train_aekl.py --seed 42 --training_ids D:\Users\dayv\4d-lung-data\manifest-...\train.tsv --validation_ids D:\Users\dayv\4d-lung-data\manifest-...\validation.tsv --config_file D:\Users\dayv\lung_upscaler\mamografIA_upscaler\configs\aekl_configs\aekl_pulmao.yaml --batch_size 2 --n_epochs 50 --autoencoder_warm_up_n_epochs 10 --val_interval 1 --num_workers 4
```

```bash
python D:\Users\dayv\lung_upscaler\mamografIA_upscaler\src\train\train_ldm.py --seed 42 --training_ids D:\Users\dayv\4d-lung-data\manifest-...\train.tsv --validation_ids D:\Users\dayv\4d-lung-data\manifest-...\validation.tsv --config_file D:\Users\dayv\lung_upscaler\mamografIA_upscaler\configs\ldm_cond_configs\ldm_pulmao_v2.yaml --stage1_config_file_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\configs\aekl_configs\aekl_pulmao.yaml --stage1_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\outputs\runs_final\aekl\model_aekl_best_4_latent_new_lung.pth --scale_factor 1.0 --batch_size 1 --n_epochs 10 --val_interval 1 --num_workers 4
```

```bash
python D:\Users\dayv\lung_upscaler\mamografIA_upscaler\src\test\upscale_test_set.py --seed 42 --output_dir D:\Users\dayv\lung_upscaler\mamografIA_upscaler\outputs\runs_final\upscale_test_new_lung_Agen --stage1_config_file_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\configs\aekl_configs\aekl_pulmao.yaml --stage1_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\outputs\runs_final\aekl\model_aekl_best_4_latent_new_lung.pth --diffusion_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\outputs\runs_final\ldm\model_ldm_pulmao_Agen.pth --diffusion_config_file_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\configs\ldm_cond_configs\ldm_pulmao_v2.yaml --x_size 256 --y_size 256 --scale_factor 1.0 --num_inference_steps 1000 --noise_level 1 --test_ids D:\Users\dayv\4d-lung-data\manifest-...\test.tsv --training_ids D:\Users\dayv\4d-lung-data\manifest-...\train.tsv
```

### 4. Training AutoencoderKL using Docker (example):

```bash
chmod +x ./src/train/bash/train_aekl.sh
sh -xe ./src/train/bash/train_aekl.sh
```

### 5. Training Latent Diffusion Model using Docker (example):

```bash
chmod +x ./src/train/bash/train_ldm.sh
sh -xe ./src/train/bash/train_ldm.sh
```

### 6. Upscaling test images and computing metrics using Docker (example):

```bash
chmod +x ./src/test/bash/upscale_test_set.sh
sh -xe ./src/test/bash/upscale_test_set.sh
```

## Metrics used in the paper

- PSNR
- MS-SSIM (Multi-Scale SSIM)

The scripts under `src/test/` implement these metrics (see `calculate_metrics.py`).

## Preprocessing

- The preprocessing pipeline in `src/preprocessing_data/` is partially outdated. If you need to run or adapt preprocessing scripts, contact me — some routines may require adjustments for current data.

## Important files (quick reference)

- [src/train/train_ldm.py](src/train/train_ldm.py)
- [src/train/train_aekl.py](src/train/train_aekl.py)
- [src/train/util_training.py](src/train/util_training.py)
- [src/train/util_transformations.py](src/train/util_transformations.py)
- [requirements.txt](requirements.txt)

## Explanation of main files

### Configs

- `aekl_configs/`: AutoencoderKL hyperparameters and configs.
- `ldm_configs/`: Base Latent Diffusion configurations.
- `ldm_cond_configs/`: Conditioned LDM variations (e.g., with lung mask).
- `vae_configs/` and `vqvae_configs/`: configs for VAE and VQ-VAE experiments.

### Preprocessing (`src/preprocessing_data`)

Key data preparation scripts (some may be outdated):

- `process_medical_images.py`: generates `.tsv` manifests for train/val/test sets.
- `create_reference_img.py`: creates a reference image from DICOM.
- `create_images_label.py`: labels images by intensity/brightness.
- `create_multiclass_mask.py` / `creating_lung_mask.py`: generate binary/multiclass lung masks.
- `crop_imgs.py`, `convert_32.py`, `create_viz.py`, `get_radiomic_features.py`, `best_100.py`, `creating_csv_a_part.py`, `4d_lung_data_analise.py`: various utilities for normalization, visualization and image selection.

> Note: the preprocessing pipeline is partially outdated; contact the author for guidance if you plan to run it.

### Training (`src/train`)

- `train_aekl.py`: trains the AutoencoderKL (stage1).
- `train_ldm.py`: trains the Latent Diffusion Model (stage2), optionally using a pretrained `stage1` via flags.
- `train_ldm_without_low_res.py`, `train_vae.py`, `train_vqvae.py`: variants and alternative models.
- `util_training.py`, `util_training_vae.py`, `util_training_vqvae.py`: common training/validation utilities and metrics computation.
- `util_transformations.py`: `DataLoader` creation and transformations — `get_upsampler_dataloader` is used by `train_aekl.py` and `train_ldm.py`.

### Testing & Evaluation (`src/test`)

- `upscale_test_set.py` / `upscale_test_set_without_low_res.py`: pipelines to run upscaling on the test set and save metrics.
- `upscale_imgs.py`, `recons_vqvae.py`, `generation_and_recons.py`: scripts for generation, reconstruction and inspection.
- `calculate_fid.py`, `calculate_metrics.py`: compute FID, PSNR, MAE, MSSIM, SSIM.

### Notebooks

- `notebooks/code_tcc_test.ipynb` and `notebooks/data_processing.ipynb`: interactive examples and experiments.

## Configuration files and useful scripts

- `.dockerignore`, `.gitignore`: development support files.
- `Dockerfile` and `create_image_docker.sh`: example Docker setup to reproduce the environment.
- `requirements.txt`: Python dependencies.

## Final notes and contact

- The repository contains legacy code and several config versions — to reproduce experiments, verify paths and config files before running.
- For help running preprocessing, adjusting YAML configs, or reproducing experiments, contact: **ddayvisongomes11@gmail.com** — I can assist with execution, configuration and result interpretation.

---

For more details, inspect the scripts in `src/` and read docstrings in the source files.
