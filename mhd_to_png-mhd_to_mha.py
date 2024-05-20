import subprocess
import SimpleITK as sitk
import numpy as np
import cv2
from PIL import Image
import os  

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin()))) 
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return image, spacing, origin

def mhd_to_mha(image_path, filename_output_mha):
    image_arr, spacing, origin  = load_itk(image_path)
    image = sitk.GetImageFromArray(image_arr)
    sitk.WriteImage(image,filename_output_mha)
    return 0

def mh_to_png(path, path_saving, filename):
    mha = sitk.ReadImage(path)
    array0 = sitk.GetArrayFromImage(mha)   
    #array = np.reshape(array0, (30, 110))
    array = np.reshape(array0, (30, 55))
    image = Image.fromarray(cv2.normalize(array, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U))
    image.save(path_saving+filename[:-4]+".png")
    return 0

def iterate_folder(filepath_input, path_saving):
    os.makedirs(filepath_input+'/png', exist_ok=True)
    filepath = os.listdir(filepath_input)
    filepath.sort()
    for filename in filepath:
        if filename.endswith(".mha"): 
            image_path = os.path.join(filepath_input, filename)
            if filepath_input == path_saving:
                path_saving = filepath_input+'/png/'
            mh_to_png(image_path, path_saving, filename)
    return 0

def mh_to_png_ph(path, path_saving, filename):
    mha = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(mha)   
    print("sh", array.shape)
    #array = np.reshape(array0, (30, 110))
    im = cv2.normalize(array, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i, j] == 232:
                im[i, j] = 100
    image = Image.fromarray(im)
    image.save(path_saving+filename[:-4]+".png")
    return 0

def iterate_folder_phant(filepath_input, path_saving):
    os.makedirs(filepath_input+'/png', exist_ok=True)
    filepath = os.listdir(filepath_input)
    filepath.sort()
    for filename in filepath:
        if filename.endswith(".mhd"): 
            image_path = os.path.join(filepath_input, filename)
            if filepath_input == path_saving:
                path_saving = filepath_input+'/png/'
            mh_to_png_ph(image_path, path_saving, filename)
    return 0

imgTr= '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset029_FMI2position_awgn/imagesTr'
imgTs = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset029_FMI2position_awgn/imagesTs'
labTr = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset029_FMI2position_awgn/labelsTr'

phantTs = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset029_FMI2position_awgn/phantom/test'
phantTr = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset029_FMI2position_awgn/phantom/train'

iterate_folder(imgTr, imgTr)
iterate_folder(imgTs, imgTs)
iterate_folder(labTr, labTr)

iterate_folder_phant(phantTs, phantTs)
iterate_folder_phant(phantTr, phantTr)
 