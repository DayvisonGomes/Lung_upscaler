"""
Cria máscaras binárias e multiclasses para pulmão em imagens CT usando windowing e segmentação automática.
Inclui funções para pós-processamento das máscaras.
"""

import os
import pandas as pd
import pydicom
import numpy as np
import tifffile
from lungmask import LMInferer
import SimpleITK as sitk
import scipy.ndimage as ndimage


def window_ct(dcm, w, c, ymin, ymax):
    """Windows a CT slice.

    Args:
        dcm (pydicom.dataset.FileDataset): The DICOM dataset containing the CT slice.
        w: Window Width parameter.
        c: Window Center parameter.
        ymin: Minimum output value.
        ymax: Maximum output value.

    Returns:
        Windowed slice.
    """
    if 'RescaleIntercept' in dcm:
        rescale_intercept = dcm.RescaleIntercept
        if np.sign(rescale_intercept) == -1:
            rescale_intercept = -rescale_intercept
    else:
        rescale_intercept = 0
    
    if np.min(dcm.pixel_array) < 0:
        y = dcm.pixel_array * dcm.RescaleSlope + rescale_intercept
    else:
        y = dcm.pixel_array * dcm.RescaleSlope

    #y = np.zeros_like(x)
    #y[x <= (c - 0.5 - (w - 1) / 2)] = ymin
    #y[x > (c - 0.5 + (w - 1) / 2)] = ymax
    #mask = (x > (c - 0.5 - (w - 1) / 2)) & (x <= (c - 0.5 + (w - 1) / 2))
    #y[mask] = ((x[mask] - (c - 0.5)) / (w - 1) + 0.5) * (ymax - ymin) + ymin

    return y

def normalize_image(image, ymin, ymax):
    return (image - ymin) / (ymax - ymin)
    
def create_lung_mask(image, threshold=0.8):
    """Creates a binary mask to highlight lung structures.

    Args:
        image (np.array): The windowed CT image.
        threshold (float): Relative threshold for segmentation.

    Returns:
        np.array: Binary mask with lung structures highlighted.
    """
    mask = np.zeros_like(image)
    mask[image > threshold] = 1
    return mask
    
def create_segmentation_mask(image, inferer):

    input_image = sitk.ReadImage(image)
    segmentation = inferer.apply(input_image)

    segmentation = segmentation.squeeze()
    segmentation_np = np.array(segmentation)
    
    return segmentation_np

def fill_holes_in_mask(mask):
    filled_mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return filled_mask


def change_value(mask):
    mask = np.where(mask == 1, 0.5, mask)
    
    return mask

def change_mask_two_class(output_path):
    #output_path = r'D:\Users\dayv\mamografIA_upscaler\data\data_lung_multiclass_masks'
    filenames = os.listdir(output_path)
        
    for name in filenames:
        mask_path = os.path.join(output_path, name)
        mask = tifffile.imread(mask_path)
        valores_unicos = np.unique(mask).shape[0]
            
        if valores_unicos == 2:
            mask = change_value(mask)
            tifffile.imwrite(mask_path, mask)

if __name__ == '__main__':

    #base_path = r'D:\Users\dayv\mamografIA_upscaler\outputs\tsv_files_train_valid_pulmao'
    #output_path = r'/project/data/data_lung_multiclass_masks'
    base_path = r'D:\Users\dayv\4d-lung-data\manifest-ObLxS9Wd1073675925233948759'
    output_path = r'D:\Users\dayv\4d-lung-data\manifest-ObLxS9Wd1073675925233948759\data_lung_multiclass_masks'
    
    change_mask_two_class(output_path)
    exit()
    os.makedirs(output_path, exist_ok=True)
    inferer = LMInferer(modelname='R231CovidWeb')
    
    df_train = pd.read_csv(os.path.join(base_path, 'train.tsv'), sep='\t')
    df_test = pd.read_csv(os.path.join(base_path, 'test.tsv'), sep='\t')
    df_val = pd.read_csv(os.path.join(base_path, 'validation.tsv'), sep='\t')
    
    df_list = [df_train, df_test, df_val]

    k = 0
    for df in df_list:
        
        for i, row in df.iterrows():
            path = row['image']
            #path = path.replace('/project/ct_noel', 'D:/Users/bbruno/Data/ct_noel')
            
            subname = path.split('\\')[-2]
            name = subname + '-' + path.split('\\')[-1].split('.')[0]
            print(name)
            dcm = pydicom.dcmread(path)
            #window_center = float(dcm.WindowCenter)
            window_center = None
            #window_width = float(dcm.WindowWidth)
            window_width = None
            ymin = 0
            ymax = 2**16

            windowed_image = window_ct(dcm, window_width, window_center, ymin, ymax)
            normalized_image = normalize_image(windowed_image, ymin, ymax)
            
            closed_lung_mask = create_lung_mask(normalized_image, threshold=0.01)
            closed_lung_mask = fill_holes_in_mask(closed_lung_mask)

            segmentation_np = create_segmentation_mask(path, inferer)
            segmentation_np = np.where(segmentation_np >= 1, 1, 0)
            
            filtered_with_closed_lung_mask = np.where(closed_lung_mask == 1, normalized_image, 0)
            filtered_with_segmentation_np = np.where(segmentation_np == 1, normalized_image, 0)
            #image with less high pixels
            combined_filtered = np.maximum(filtered_with_closed_lung_mask, filtered_with_segmentation_np)

            lung_mask_binary = np.where(closed_lung_mask != 0, 1, 0)
            segmentation_np_mask = np.where(segmentation_np == 1, 2, 0)

            combined_mask = lung_mask_binary + segmentation_np_mask 
            combined_mask = np.where(combined_mask == 3, 2, combined_mask)
            combined_mask = combined_mask.astype(np.uint8)
            combined_mask = (combined_mask / combined_mask.max())
            
            output_file_path = f'{name}_mask.tiff'
            final_path = os.path.join(output_path, output_file_path)
            tifffile.imwrite(final_path, combined_mask.astype(np.float32))
            
            del windowed_image
            del normalized_image
            del combined_mask

            if i % 1000 == 0:
                print(f'{i} de um total de {df.shape[0]}')
        k += 1
        print(k)
            
            

            
