import subprocess
import SimpleITK as sitk
import numpy as np
import cv2
from PIL import Image
import os 

def execute_command(command):
    try:
        # Execute the command
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        # Print the output
        print("Command output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        # Print the error message if the command fails
        print("Command failed with error:", e)
        print("Error output:")
        print(e.stderr)

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
    #print("sh", array0.shape)
    array = np.reshape(array0, (30, 55))
    image = Image.fromarray(cv2.normalize(array, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U))
    image.save(path_saving+filename[:-9]+".png")
    return 0

def iterate_folder(filepath_input, path_saving):
    filepath = os.listdir(filepath_input)
    filepath.sort()
    for filename in filepath:
        if filename.endswith("0.mhd"): 
            image_path = os.path.join(filepath_input, filename)
            execute_command('mhd-AWGN ' +filename+' 5 5')
            mh_to_png(image_path, path_saving, filename)
    return 0

def iterate_folder1(filepath_input, path_saving):
    filepath = os.listdir(filepath_input)
    filepath.sort()
    for filename in filepath:
        if filename.endswith("AWGN.mhd"): 
            image_path = os.path.join(filepath_input, filename)
            mh_to_png(image_path, path_saving, filename)
            path_saving1 = image_path[:-9]+".mha"
            image_path1 = path_saving1[:-14]+'7'+path_saving1[-13:]
            mhd_to_mha(image_path, image_path1)
    return 0


test_data_src = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset024_FMInoise/imagesTs'
train_data_src = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset024_FMInoise/imagesTr'
test_data_dest = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset024_FMInoise/noise/test/data/'
train_data_dest = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset024_FMInoise/noise/train/data/'

train_data_src = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset037_FMInoise5_self/mhd_data/train/data'
train_data_dest = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset037_FMInoise5_self/mhd_data/train/data/noise/'
test_data_src = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset037_FMInoise5_self/mhd_data/test/data'
test_data_dest = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset037_FMInoise5_self/mhd_data/test/data/noise/'

#first execute this
iterate_folder(test_data_src, test_data_dest)
#then execute this
iterate_folder1(test_data_src, test_data_dest)

