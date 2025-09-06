"""
Análise de dados 4D de pulmão, geração de tabelas de treino/validação/teste e exportação de imagens DICOM para TIFF.
Funções principais:
- create_table: Cria subconjuntos de dados para treino, validação e teste.
- open_one_img_save_tiff: Salva uma amostra de imagem DICOM como TIFF.
- open_dcm: Carrega e aplica rescale em imagens DICOM.
- get_mean_std_all_train: Calcula média e desvio padrão das imagens de treino.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from natsort import natsorted
import pydicom
import tifffile
import numpy as np

# & d:/Users/dayv/lung_upscaler/envs/day_env/python.exe d:/Users/dayv/mamografIA_upscaler/src/preprocessing_data/4d_lung_data_analise.py

def create_table(df, train_id, path_metadata, set_df):
       
    for id in train_id:
        df_id = df[df['Subject ID'] == id]

        for i, row in df_id.iterrows():
            files_path = os.path.join(os.path.dirname(path_metadata), row['File Location'])

            for i, file in enumerate(natsorted(os.listdir(files_path))):
                if i >= 19:
                    set_df = set_df._append({'Subject ID': id, 'image': os.path.join(files_path, file)}, ignore_index=True)

    return set_df

def open_one_img_save_tiff(df):
    sample = df.sample(n=1, random_state=50)
    img_path = sample['image'].values[0]
    
    img = pydicom.dcmread(img_path)
    img_array = img.pixel_array
    
    name = img_path.split('\\')[-1].split('.')[0]

    if 'RescaleSlope' in img:
        rescale_slope = img.RescaleSlope
    else:
        rescale_slope = 1
    
    if 'RescaleIntercept' in img:
        rescale_intercept = img.RescaleIntercept
        if np.sign(rescale_intercept) == -1:
            rescale_intercept = -rescale_intercept
    else:
        rescale_intercept = 0
    
    if np.min(img_array) < 0:
        y = img_array * rescale_slope + rescale_intercept
    else:
        y = img_array * rescale_slope

    tifffile.imwrite(os.path.join(os.path.dirname(path_metadata), f'{name}_original.tiff'), img_array.astype(np.float32))
    tifffile.imwrite(os.path.join(os.path.dirname(path_metadata), f'{name}_rescaled.tiff'), y.astype(np.float32))

def open_dcm(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)

    if 'RescaleSlope' in dcm:
        rescale_slope = dcm.RescaleSlope
    else:
        rescale_slope = 1
    
    if 'RescaleIntercept' in dcm:
        rescale_intercept = dcm.RescaleIntercept
        if np.sign(rescale_intercept) == -1:
            rescale_intercept = -rescale_intercept
    else:
        rescale_intercept = 0
    
    if np.min(img) < 0:
        y = img * rescale_slope + rescale_intercept
    else:
        y = img * rescale_slope

    return y

def get_mean_std_all_train(train):
    images_list = []
    train_= train[:15000].copy()
    for _, row in train_.iterrows():
        img = open_dcm(row['image'])

        images_list.append(img)  

    vec_all_images = np.concatenate([img.flatten() for img in images_list], axis=0)

    print(vec_all_images.shape)
    print(f'Mean: {vec_all_images.mean()}')
    print(f'Std: {vec_all_images.std()}')

if __name__ == '__main__':
    
    path_metadata = r'D:\Users\dayv\4d-lung-data\manifest-ObLxS9Wd1073675925233948759\metadata.csv'
    train = pd.read_csv(r'D:\Users\dayv\4d-lung-data\manifest-ObLxS9Wd1073675925233948759\train.tsv', sep='\t')
    print(train.head())
    #get_mean_std_all_train(train)
    
    df = pd.read_csv(path_metadata)

    cols = ['Number of Images', 'SOP Class Name', 'Modality', 'Manufacturer', 'Study Description', 'Collection']

    for col in cols:
        print(df[col].value_counts())
    
    df = df[(df['Manufacturer'] == 'ADAC') & (df['Modality'] == 'CT')]
    unique_id = df['Subject ID'].unique()

    train_id, test_id = train_test_split(unique_id, test_size=0.3, random_state=42)
    val_id, test_id = train_test_split(test_id, test_size=0.5, random_state=42)

    print(train_id.shape)
    print(val_id.shape)
    print(test_id.shape)

       
    os.makedirs(os.path.dirname(path_metadata), exist_ok=True)
    train_df = pd.DataFrame(columns=['Subject ID', 'image'])
    val_df = pd.DataFrame(columns=['Subject ID', 'image'])
    test_df = pd.DataFrame(columns=['Subject ID', 'image'])
 
    train_df = create_table(df, train_id, path_metadata, train_df).sample(frac=1, random_state=42)
    val_df = create_table(df, val_id, path_metadata, val_df).sample(frac=1, random_state=42)
    test_df = create_table(df, test_id, path_metadata, test_df).sample(frac=1, random_state=42)
    
    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)
    exit()   
    train_df.to_csv(os.path.join(os.path.dirname(path_metadata), 'train.tsv'), index=False, sep='\t')
    val_df.to_csv(os.path.join(os.path.dirname(path_metadata), 'val.tsv'), index=False, sep='\t')
    test_df.to_csv(os.path.join(os.path.dirname(path_metadata), 'test.tsv'), index=False, sep='\t')

    open_one_img_save_tiff(test_df)

            
        