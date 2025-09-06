# lung_upscaler

Repositório contendo todos os códigos para a realização da super-resolução em imagens de tomografia computadorizada de pulmão, utilizando AutoencoderKL e modelos de difusão latente UNet.

Este código faz parte da minha dissertação e do artigo:  
**Importance of Conditioning in Latent Diffusion Models for Lung Computed Tomography Image Generation**

## Estrutura de Pastas

```bash
.
├── configs
│   ├── aekl_configs
│   │   ├── aekl_artigo.yaml
│   │   ├── aekl_clinical.yaml
│   │   └── ...
│   ├── ldm_configs
│   │   ├── ldm_v0.yaml
│   │   ├── ldm_teste.yaml
│   │   └── ...
│   ├── ldm_cond_configs
│   │   ├── ldm_pulmao_v2.yaml
│   │   └── ...
│   ├── vae_configs
│   │   └── ...
│   ├── vqvae_configs
│   │   └── ...
├── src
│   ├── preprocessing_data
│   │   ├── 4d_lung_data_analise.py
│   │   ├── best_100.py
│   │   ├── convert_32.py
│   │   ├── create_images_label.py
│   │   ├── create_multiclass_mask.py
│   │   ├── create_reference_img.py
│   │   ├── creating_csv_a_part.py
│   │   ├── creating_lung_mask.py
│   │   ├── crop_imgs.py
│   │   ├── get_radiomic_features.py
│   │   ├── process_medical_images.py
│   │   └── create_viz.py
│   ├── test
│   │   ├── bash
│   │   │   ├── upscale_test_set.sh
│   │   │   └── upscale_imgs.sh
│   │   ├── upscale_test_set.py
│   │   ├── upscale_test_set_without_low_res.py
│   │   ├── upscale_imgs.py
│   │   ├── calculate_fid.py
│   │   ├── calculate_metrics.py
│   │   ├── recons_vqvae.py
│   │   ├── generation_and_recons.py
│   │   ├── util_transformations.py
│   │   └── custom_transforms.py
│   ├── train
│   │   ├── bash
│   │   │   ├── train_aekl.sh
│   │   │   └── train_ldm.sh
│   │   ├── train_aekl.py
│   │   ├── train_ldm.py
│   │   ├── train_ldm_without_low_res.py
│   │   ├── train_vae.py
│   │   ├── train_vqvae.py
│   │   ├── util_training.py
│   │   ├── util_training_vae.py
│   │   ├── util_training_vqvae.py
│   │   ├── util_transformations.py
│   │   └── custom_transforms.py
├── notebooks
│   ├── code_tcc_test.ipynb
│   └── data_processing.ipynb
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── create_image_docker.sh
```

## Explicação dos Principais Arquivos

### Configs

- **aekl_configs/**: Configurações do AutoencoderKL.
- **ldm_configs/**: Configurações dos modelos Latent Diffusion.
- **ldm_cond_configs/**: Configurações de modelos LDM condicionados (ex: com máscara de pulmão).
- **vae_configs/**: Configurações para modelos VAE.
- **vqvae_configs/**: Configurações para modelos VQ-VAE.

### Pré-processamento (`src/preprocessing_data`)

Scripts para preparar os dados antes do treinamento:
- **process_medical_images.py**: Gera arquivos `.tsv` com os caminhos das imagens para treino, validação e teste.
- **create_reference_img.py**: Cria imagem de referência a partir de DICOM.
- **create_images_label.py**: Classifica imagens por brilho.
- **create_multiclass_mask.py**: Gera máscaras binárias/multiclasse para pulmão.
- **creating_lung_mask.py**: Gera máscaras binárias para pulmão.
- **crop_imgs.py**: Realiza crop nas imagens DICOM.
- **convert_32.py**: Converte TIFF para float32 e normaliza.
- **get_radiomic_features.py**: Extrai features radiômicas.
- **create_viz.py**: Gera visualizações PNG das imagens.
- **best_100.py**: Calcula MSSIM e seleciona melhores imagens.
- **creating_csv_a_part.py**: Cria arquivos `.tsv` balanceados.
- **4d_lung_data_analise.py**: Análise de dados 4D do pulmão.

### Treinamento (`src/train`)

Scripts para treinar os modelos:
- **train_aekl.py**: Treina o AutoencoderKL.
- **train_ldm.py**: Treina o modelo Latent Diffusion.
- **train_ldm_without_low_res.py**: Treina LDM sem imagens de baixa resolução.
- **train_vae.py**: Treina Variational Autoencoder.
- **train_vqvae.py**: Treina VQ-VAE.
- **util_training.py / util_training_vae.py / util_training_vqvae.py**: Funções utilitárias para loops de treino, validação e métricas.
- **util_transformations.py**: Funções para criação de dataloaders e transformações.
- **custom_transforms.py**: Transformações customizadas para pipeline MONAI.

### Teste e Avaliação (`src/test`)

Scripts para avaliação dos modelos e geração de métricas:
- **upscale_test_set.py**: Realiza upscale no conjunto de teste e salva métricas.
- **upscale_test_set_without_low_res.py**: Teste sem imagens de baixa resolução.
- **upscale_imgs.py**: Upscale de imagens médicas.
- **calculate_fid.py / calculate_metrics.py**: Calcula FID, PSNR, MAE, MSSIM, SSIM.
- **recons_vqvae.py / generation_and_recons.py**: Reconstrução e geração de imagens.
- **util_transformations.py / custom_transforms.py**: Utilitários para dataloaders e transformações.
- **bash/**: Scripts bash para rodar os processos via Docker.

### Notebooks

- **code_tcc_test.ipynb** e **data_processing.ipynb**: Exemplos e experimentos interativos.

## Arquivos de Configuração

- **.dockerignore**: Arquivos ignorados na construção da imagem Docker.
- **.gitignore**: Arquivos ignorados pelo Git.
- **Dockerfile**: Construção da imagem Docker.
- **LICENSE**: Licença Apache 2.0.
- **requirements.txt**: Dependências do projeto.
- **create_image_docker.sh**: Script para build da imagem Docker.

## Primeiros passos

1. **Construir a imagem Docker**

   Certifique-se de ter o Docker instalado. Execute:

   ```bash
   docker build -t <image-name> .
   ```
   Ou utilize o script:

   ```bash
   chmod +x create_image_docker.sh
   sh -xe create_image_docker.sh
   ```

2. **Pré-processamento dos dados**

   Crie uma pasta chamada `data` com os dados necessários. Execute o script para gerar os arquivos `.tsv`:

   ```bash
   docker run -it \
       --ipc=host \
       -v ${PWD}:/project/ \
       <image-name> \
       python /project/src/preprocessing_data/process_medical_images.py \
           --output_dir /project/outputs/tsv_files
   ```

3. **Treinamento dos modelos**

   - **AutoencoderKL**
     ```bash
     chmod +x ./src/train/bash/train_aekl.sh
     sh -xe ./src/train/bash/train_aekl.sh
     ```
   - **Latent Diffusion Model**
     ```bash
     chmod +x ./src/train/bash/train_ldm.sh
     sh -xe ./src/train/bash/train_ldm.sh
     ```

4. **Realizar upscale nas imagens de teste**

   ```bash
   chmod +x ./src/test/bash/upscale_test_set.sh
   sh -xe ./src/test/bash/upscale_test_set.sh
   ```

5. **Avaliação e Métricas**

   Utilize os scripts em `src/test` para calcular métricas como FID, PSNR, SSIM, MSSIM, MAE e gerar visualizações.

## Observações

- Para alterar hiperparâmetros, edite os arquivos `.yaml` em `configs/`.
- Para modificar a quantidade de imagens em cada subset, edite diretamente os scripts de pré-processamento.
- Os scripts bash facilitam a execução dos principais fluxos via Docker.

---

Para dúvidas ou sugestões, consulte os scripts e docstrings nos arquivos fonte.
