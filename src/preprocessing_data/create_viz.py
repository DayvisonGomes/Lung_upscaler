"""
Gera visualizações PNG a partir de imagens TIFF, incluindo diferença entre imagens originais e geradas.
Processa diretórios e salva resultados em subpastas.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from natsort import natsorted
from PIL import Image

def convert_and_save_tiff_to_png(input_path, output_path):
    img = tifffile.imread(input_path)
    
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img *= 255.0
    img = img.astype(np.uint8)

    Image.fromarray(img).save(output_path)


def generate_difference_plot(orig_img, gen_img, save_path):
    diff = orig_img.astype(np.float32) - gen_img.astype(np.float32)
    plt.figure(figsize=(6, 6))
    plt.imshow(diff, cmap='bwr', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    plt.colorbar(label='Diferença (Original - Gerada)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def process_directory(root_path, step=2):
    print(f"Processando: {root_path}")
    if not os.path.isdir(root_path):
        print(f"Path inválido: {root_path}")
        return
    
    files = natsorted([f for f in os.listdir(root_path) if f.endswith(".tiff") and 'logs' not in f])
    base_output_path = os.path.join(root_path, "_converted_png")
    os.makedirs(base_output_path, exist_ok=True)

    for i in range(0, len(files) - 1, step):
        try:
            orig_path = os.path.join(root_path, files[i])
            gen_path = os.path.join(root_path, files[i + 1])

            img_name = files[i].replace(".tiff", "")
            orig_img = tifffile.imread(orig_path)
            gen_img = tifffile.imread(gen_path)

            # Salvar PNGs
            convert_and_save_tiff_to_png(orig_path, os.path.join(base_output_path, f"{img_name}_orig.png"))
            convert_and_save_tiff_to_png(gen_path, os.path.join(base_output_path, f"{img_name}_gen.png"))

            # Plot diferença
            diff_path = os.path.join(base_output_path, f"{img_name}_diff.png")
            generate_difference_plot(orig_img, gen_img, diff_path)
        except IndexError:
            print(f"Pulando par incompleto no final da lista: {files[i:]}")
            break

    print(f"Finalizado: {root_path}")

if __name__ == '__main__':
    base_path = r"D:\Users\dayv\lung_upscaler\mamografIA_upscaler\outputs\runs_final\runs_final_32"
    selected_dirs = [
        'upscale_test_new_lung_Agen',
        'upscale_test_new_lung_Asr',
        'upscale_test_new_lung_Bgen',
        'upscale_test_new_lung_Bsr',
        'upscale_test_new_lung_Cgen',
        'upscale_test_new_lung_Csr',
        'upscale_test_new_lung_Dgen',
        'upscale_test_new_lung_Dsr'
    ]

    for dir_name in selected_dirs:
        dir_path = os.path.join(base_path, dir_name)

        if any(key in dir_name for key in ['Bgen', 'Cgen', 'Dgen']):
            step = 3
        else:
            step = 2

        process_directory(dir_path, step=step)
