
# lung_upscaler

Repositório contendo todos os códigos para a realização da super-resolução em imagens de tomografia computadorizada de pulmão, utilizando AutoencoderKL e modelos de difusão latente UNet.

Este código faz parte da minha dissertação e do artigo:
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

## Observações importantes

- O README contém um fluxo geral para executar pré-processamento, treinamento e avaliação, porém partes do repositório contêm código legado mantido como referência.
- Alguns scripts e utilitários não são necessários para os pipelines principais — servem como histórico/backup. Consulte o autor em caso de dúvida.
- Os scripts em `src/train/bash/` e `src/test/bash/` são exemplos para uso com Docker; também é possível trabalhar localmente com Conda/Miniconda e instalar as dependências via `requirements.txt`.
- Existem várias versões de arquivos de configuração em `configs/` — escolha a que melhor se adequa ao seu experimento (por ex.: `configs/ldm_cond_configs/`, `configs/aekl_configs/`).

## Arquivos e funções essenciais

- `src/train/util_training.py`: utilitários de treinamento e laços de treino/validação.
- `src/train/util_transformations.py`: transformações e criação de dataloaders; o método `get_upsampler_dataloader` é especialmente importante e é utilizado por `src/train/train_aekl.py` e `src/train/train_ldm.py`.

## Primeiros passos rápidos

### 1. Recomendado: usar Conda/Miniconda para criar o ambiente (exemplo):

```bash
conda create -n lung_upscaler python=3.10 -y
conda activate lung_upscaler
pip install -r requirements.txt
```

### 2. Se preferir Docker (início rápido):

```bash
docker build -t lung_upscaler -f Dockerfile .
docker run -it --ipc=host -v ${PWD}:/project/lung_upscaler --gpus all \
    DOCKER_IMAGE python ...
```

### 3. Exemplos de comandos usados (Windows paths — servem como exemplo de flags e argumentos):

```bash
python D:\Users\dayv\lung_upscaler\mamografIA_upscaler\src\train\train_aekl.py --seed 42 --training_ids D:\Users\dayv\4d-lung-data\manifest-...\train.tsv --validation_ids D:\Users\dayv\4d-lung-data\manifest-...\validation.tsv --config_file D:\Users\dayv\lung_upscaler\mamografIA_upscaler\configs\aekl_configs\aekl_pulmao.yaml --batch_size 2 --n_epochs 50 --autoencoder_warm_up_n_epochs 10 --val_interval 1 --num_workers 4
```

```bash
python D:\Users\dayv\lung_upscaler\mamografIA_upscaler\src\train\train_ldm.py --seed 42 --training_ids D:\Users\dayv\4d-lung-data\manifest-...\train.tsv --validation_ids D:\Users\dayv\4d-lung-data\manifest-...\validation.tsv --config_file D:\Users\dayv\lung_upscaler\mamografIA_upscaler\configs\ldm_cond_configs\ldm_pulmao_v2.yaml --stage1_config_file_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\configs\aekl_configs\aekl_pulmao.yaml --stage1_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\outputs\runs_final\aekl\model_aekl_best_4_latent_new_lung.pth --scale_factor 1.0 --batch_size 1 --n_epochs 10 --val_interval 1 --num_workers 4
```

```bash
python D:\Users\dayv\lung_upscaler\mamografIA_upscaler\src\test\upscale_test_set.py --seed 42 --output_dir D:\Users\dayv\lung_upscaler\mamografIA_upscaler\outputs\runs_final\upscale_test_new_lung_Agen --stage1_config_file_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\configs\aekl_configs\aekl_pulmao.yaml --stage1_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\outputs\runs_final\aekl\model_aekl_best_4_latent_new_lung.pth --diffusion_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\outputs\runs_final\ldm\model_ldm_pulmao_Agen.pth --diffusion_config_file_path D:\Users\dayv\lung_upscaler\mamografIA_upscaler\configs\ldm_cond_configs\ldm_pulmao_v2.yaml --x_size 256 --y_size 256 --scale_factor 1.0 --num_inference_steps 1000 --noise_level 1 --test_ids D:\Users\dayv\4d-lung-data\manifest-...\test.tsv --training_ids D:\Users\dayv\4d-lung-data\manifest-...\train.tsv
```

### 4. Treinamento do AutoencoderKL utilizando Docker:

```bash
chmod +x ./src/train/bash/train_aekl.sh
sh -xe ./src/train/bash/train_aekl.sh
```

### 5. Treinamento do Modelo de Difusão Latente utilizando Docker:

```bash
chmod +x ./src/train/bash/train_ldm.sh
sh -xe ./src/train/bash/train_ldm.sh
```
### 6. Realizar upscale nas imagens de teste e calcular as métricas utilizando Docker: 

```bash
chmod +x ./src/test/bash/upscale_test_set.sh
sh -xe ./src/test/bash/upscale_test_set.sh
```

## Métricas utilizadas no artigo

- PSNR
- MS-SSIM (Multi-Scale SSIM)

Os scripts em `src/test/` implementam o cálculo destas métricas (veja `calculate_metrics.py`).

## Pré-processamento

- O pipeline de pré-processamento (`src/preprocessing_data/`) está parcialmente desatualizado. Se precisar executar ou adaptar scripts de pré-processamento, entre em contato comigo — algumas rotinas podem precisar de ajustes para os dados atuais.

## Contatos

- E-mail: ddayvisongomes11@gmail.com

## Arquivos importantes (referência rápida)

- [src/train/train_ldm.py](src/train/train_ldm.py)
- [src/train/train_aekl.py](src/train/train_aekl.py)
- [src/train/util_training.py](src/train/util_training.py)
- [src/train/util_transformations.py](src/train/util_transformations.py)
- [requirements.txt](requirements.txt)

## Explicação dos Principais Arquivos

### Configs

- **aekl_configs/**: configurações e hiperparâmetros do AutoencoderKL.
- **ldm_configs/**: configurações base dos modelos Latent Diffusion.
- **ldm_cond_configs/**: variações condicionadas (por exemplo, com máscara de pulmão).
- **vae_configs/** e **vqvae_configs/**: configurações para VAE e VQ-VAE quando usados.

### Pré-processamento (`src/preprocessing_data`)

Principais scripts de preparação de dados (atenção: alguns podem estar desatualizados):

- `process_medical_images.py`: gera arquivos `.tsv` com caminhos para treino/validação/teste.
- `create_reference_img.py`: cria imagem de referência a partir de DICOM.
- `create_images_label.py`: classifica imagens por brilho.
- `create_multiclass_mask.py` / `creating_lung_mask.py`: geram máscaras binárias/multiclasse para pulmão.
- `crop_imgs.py`, `convert_32.py`, `create_viz.py`, `get_radiomic_features.py`, `best_100.py`, `creating_csv_a_part.py`, `4d_lung_data_analise.py` — utilitários diversos para normalização, visualização e seleção de imagens.

> Observação: o pipeline de pré-processamento está parcialmente desatualizado; entre em contato por e-mail para orientação caso precise rodá-lo.

### Treinamento (`src/train`)

- `train_aekl.py`: treina o AutoencoderKL (stage1).
- `train_ldm.py`: treina o Latent Diffusion Model (stage2), pode usar um `stage1` pré-treinado via flags.
- `train_ldm_without_low_res.py`, `train_vae.py`, `train_vqvae.py`: variantes e outros modelos.
- `util_training.py`, `util_training_vae.py`, `util_training_vqvae.py`: funções comuns de treino/validação e métricas.
- `util_transformations.py`: criação de `DataLoader`s e transformações — o método `get_upsampler_dataloader` é utilizado por `train_aekl.py` e `train_ldm.py`.

### Teste e Avaliação (`src/test`)

- `upscale_test_set.py` / `upscale_test_set_without_low_res.py`: pipelines para realizar upscale no conjunto de teste e salvar métricas.
- `upscale_imgs.py`, `recons_vqvae.py`, `generation_and_recons.py`: scripts para geração, reconstrução e inspeção de imagens.
- `calculate_fid.py`, `calculate_metrics.py`: cálculo de métricas (FID, PSNR, MAE, MSSIM, SSIM).

### Notebooks

- `notebooks/code_tcc_test.ipynb` e `notebooks/data_processing.ipynb`: exemplos e experimentos interativos.

## Arquivos de Configuração e Scripts Úteis

- `Dockerfile` e `create_image_docker.sh` — exemplo de criação de imagem Docker para reproduzir ambiente.
- `requirements.txt` — dependências Python.

## Observações finais e contato

- O repositório contém código legado e várias versões de configs; para reproduzir experimentos, confirme os caminhos e os arquivos de configuração antes de rodar.
- Se tiver dúvidas, precisar de orientação para rodar o pré-processamento ou ajustar os `yaml`, entre em contato: **ddayvisongomes11@gmail.com** — respondo a dúvidas sobre execução, configuração e interpretação de resultados.

---
