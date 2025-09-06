"""
Converte imagens TIFF para float32 e normaliza os valores de pixel entre 0 e 1.
Gera uma nova pasta com as imagens convertidas.
"""

import os
import tifffile
import numpy as np


def get_max_value_min_value(base_path, selected_dirs):
    max_value = 0
    min_value = 0
    for dir_name in selected_dirs:
        dir_path = os.path.join(base_path, dir_name)
        for file in os.listdir(dir_path):
            if file.endswith(".tiff"):
                file_path = os.path.join(dir_path, file)
                img = tifffile.imread(file_path)
                max_value = max(max_value, np.max(img))
                min_value = min(min_value, np.min(img))
    return max_value, min_value

if __name__ == '__main__':

    base_path = r"D:\Users\dayv\lung_upscaler\mamografIA_upscaler\outputs\runs_final"
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
    max_value, min_value = get_max_value_min_value(base_path, selected_dirs)
    print(max_value, min_value)
    new_path = os.path.join(base_path, "runs_final_32")
    os.makedirs(new_path, exist_ok=True)

    for dir_name in selected_dirs:
        dir_path = os.path.join(base_path, dir_name)
        os.makedirs(os.path.join(new_path, dir_name), exist_ok=True)

        files = os.listdir(dir_path)
        for file in files:
            if file.endswith(".tiff"):
                file_path = os.path.join(dir_path, file)
                img = tifffile.imread(file_path)
                img = img.astype("float32")
                img = (img - min_value) / (max_value - min_value)
                tifffile.imwrite(os.path.join(new_path, dir_name, file), img)
    