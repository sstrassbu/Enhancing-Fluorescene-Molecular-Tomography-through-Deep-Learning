import numpy as np
import SimpleITK as sitk
from PIL import Image
import cv2
import os 

def change_name_png(image_path, path_saving, filename, new_name):
    image = Image.open(image_path)
    image.save(new_name)
    return 0 

def change_name_mha(image_path, path_saving, filename, new_name):
    image_arr, spacing, origin  = load_itk(image_path)
    image = sitk.GetImageFromArray(image_arr)
    #sitk.WriteImage(image,new_name)
    return 0

def iterate_folder_change_name(filepath_input, path_saving):
    filepath = os.listdir(filepath_input)
    filepath.sort()
    for filename in filepath:
        new_name = path_saving + filename[:8]+'41'+filename[10:]
        if filename.endswith(".png"): 
            print(new_name)
            image_path = os.path.join(filepath_input, filename)
            print(image_path)
            change_name_png(image_path, path_saving, filename, new_name)
        if filename.endswith(".mha"):
            image_path = os.path.join(filepath_input, filename)
            change_name_mha(image_path, path_saving, filename, new_name)
    return 0

def png_to_mh(path, path_saving, filename):
    image = Image.open(path)
    array0 = np.array(image)
    # Add an additional dimension to match the desired shape (30, 110, 1)
    array0 = np.expand_dims(array0, axis=-1)    
    print("shape", array0.shape)
    array0 = np.reshape(array0, (30, 55, 1))  # Assuming the original shape was (30, 55)
    print("shape", array0.shape)
    array0 = array0.astype(np.float32)
    array0 = (array0 - np.min(array0)) / (np.max(array0) - np.min(array0))
    #array0 = cv2.normalize(array0, None, 1.0, 0.0, cv2.NORM_MINMAX, cv2.CV_32F)
    print("shape", array0.shape)
    array0 = np.transpose(array0, axes=(2, 0, 1))
    print("shape", array0.shape)

    mha = sitk.GetImageFromArray(array0)
    #sitk.WriteImage(mha, path_saving + filename[:9]+'9'+filename[11:-4] + ".mha")
    sitk.WriteImage(mha, path_saving + filename[:9]+'9'+filename[10:-4] + ".mhd")
    return 0

def iterate_folder(filepath_input, path_saving):
    filepath = os.listdir(filepath_input)
    filepath.sort()
    for filename in filepath:
        if filename.endswith(".png"): 
            image_path = os.path.join(filepath_input, filename)
            png_to_mh(image_path, path_saving, filename)
    return 0

path_to_png = "/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset038_addTwo/imagesTr"
path_to_save_mha = "/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset039_addTwo/mha_imTr/"
iterate_folder(path_to_png, path_to_save_mha)

path_to_png = "/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset038_addTwo/imagesTs"
path_to_save_mha = "/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset039_addTwo/mha_imTs/"
iterate_folder(path_to_png, path_to_save_mha)

path_to_names = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset041_addTwo_noise/labelsTr'
iterate_folder_change_name(path_to_names, path_to_names+'/')