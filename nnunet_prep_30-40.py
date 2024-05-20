from matplotlib import pyplot as plt 
import matplotlib as mpl
import matplotlib.pylab as plt
import SimpleITK as sitk
import numpy as np
import cv2
from PIL import Image
import os 
import json
from skimage.transform import radon, rescale, resize
import shutil

def split_data_in_directory(src_dir_home, dest_dir_data, dest_dir_label):
    if not os.path.exists(src_dir_home):
        print("Source directory does not exist.")
        return
    if os.path.exists(dest_dir_data):
        print("Dest directory does exist.")
        return
    if os.path.exists(dest_dir_label):
        print("Dest directory does exist.")
        return
    os.makedirs(dest_dir_data, exist_ok=True)
    os.makedirs(dest_dir_label, exist_ok=True)

    dirs = [d for d in os.listdir(src_dir_home) if os.path.isdir(os.path.join(src_dir_home, d))]
    dirs.sort()
    for i, directory in enumerate(dirs):
        if directory.startswith("lipros-30-sphere_"):
            src_directory = os.path.join(src_dir_home, directory)
            dest_directory = os.path.join(dest_dir_data, directory)
            shutil.copytree(src_directory, dest_directory)
        else: 
            src_directory = os.path.join(src_dir_home, directory)
            dest_directory = os.path.join(dest_dir_label, directory)
            shutil.copytree(src_directory, dest_directory)
    print(f"Copied {src_directory} to {dest_directory}")
    print("split_data_in_directory - done")

def copy_every_third_directory(src_dir_data, src_dir_label, dest_dir_train, dest_dir_train_label, dest_dir_test, dest_dir_test_label):
    # Check if the source directory exists
    if not os.path.exists(src_dir_data):
        print("Source directory data does not exist.")
        return
    if not os.path.exists(src_dir_label):
        print("Source directory label does not exist.")
        return
    if os.path.exists(dest_dir_train):
        print("Dest directory does exist.")
        return
        
    os.makedirs(dest_dir_train, exist_ok=True)
    os.makedirs(dest_dir_train_label, exist_ok=True)
    os.makedirs(dest_dir_test, exist_ok=True)
    os.makedirs(dest_dir_test_label, exist_ok=True)

    dirs = [d for d in os.listdir(src_dir_data) if os.path.isdir(os.path.join(src_dir_data, d))]
    dirs.sort()
    for i, directory in enumerate(dirs):
            if (i + 1) % 3 == 0: 
                src_directory = os.path.join(src_dir_data, directory)
                dest_directory = os.path.join(dest_dir_test, directory)
                shutil.copytree(src_directory, dest_directory)
            else: 
                src_directory = os.path.join(src_dir_data, directory)
                dest_directory = os.path.join(dest_dir_train, directory)
                shutil.copytree(src_directory, dest_directory)
    print(f"Copied {src_directory} to {dest_directory}")

    dirs = [d for d in os.listdir(src_dir_label) if os.path.isdir(os.path.join(src_dir_label, d))]
    dirs.sort()
    for i, directory in enumerate(dirs):            
            if (i + 1) % 3 == 0: 
                src_directory = os.path.join(src_dir_label, directory)
                dest_directory = os.path.join(dest_dir_test_label, directory)
                shutil.copytree(src_directory, dest_directory)
            else: 
                src_directory = os.path.join(src_dir_label, directory)
                dest_directory = os.path.join(dest_dir_train_label, directory)
                shutil.copytree(src_directory, dest_directory)
                        
    print(f"Copied {src_directory} to {dest_directory}")
    print("copy_every_third_directory - done")
 
def add_laser_layer(image_arr, filename_output_label_split, filename):
    arrays = []
    label = filename_output_label_split+filename
    for i in range(0,30): 
        arrays.append(image_arr)
    arrays_combined = np.array(arrays)
    image_combined = sitk.GetImageFromArray(arrays_combined)
    sitk.WriteImage(image_combined,label)
    image_arr, spacing, origin  = load_itk(label)
    image_arr = np.transpose(image_arr, axes=(0, 2, 1))
    return image_arr

def add_laser_layer_bin(image_arr, filename_output_label_split, filename):
    arrays = []
    label_split = filename_output_label_split+filename
    for i in range(0,30): 
        arrays.append(image_arr)
    arrays_combined = np.array(arrays)
    image_combined = sitk.GetImageFromArray(arrays_combined)
    sitk.WriteImage(image_combined,label_split)
    image_arr, spacing, origin  = load_itk(label_split)
    image_arr = np.transpose(image_arr, axes=(1, 0, 2))
    return image_arr

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin()))) 
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return image, spacing, origin

def mhd_to_mha(image_path, filename_output_mha_bin, filename_output_label_split, filename, filename_output_mha):
    image_arr, spacing, origin  = load_itk(image_path)
    #image_arr = resize(image_arr, (1,30, 55),anti_aliasing=False)
    resized_array_3d = np.zeros((1, 30, 55), dtype=np.uint8)
    for i in range(image_arr.shape[0]):
        resized_array_3d[i,:,:] = cv2.resize(image_arr[i,:,:].astype(np.uint8), (55, 30), interpolation=cv2.INTER_NEAREST)
    
    arr = np.reshape(resized_array_3d, (30, 55))
    arr = Image.fromarray(cv2.normalize(arr, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U))
    arr.save(filename_output_mha_bin[:-4]+".png")
    arr.save(filename_output_mha[:-4]+".png")

    image = sitk.GetImageFromArray(resized_array_3d)
    sitk.WriteImage(image,filename_output_mha)
    return image

def load_ALL_label(filepath_input, filename_output_mha, filename_output_mhd, type_mod, filename_output_label_split, filename_output_mha_nonbin, filename_output_mha_bin, filename_output_mha_self):
    print("saved as: ", filename_output_mha)
    print("load_ALL open")
    arrays = []
    filepath = os.listdir(filepath_input)
    filepath.sort()
    for filename in filepath:
        if filename.endswith("oCams.mhd"):
            image_path = os.path.join(filepath_input, filename)
            image = mhd_to_mha(image_path, filename_output_mha_nonbin, filename_output_label_split, filename, filename_output_mha_nonbin)
        if filename.endswith("binarized-10.mhd"):
            image_path = os.path.join(filepath_input, filename)
            image = mhd_to_mha(image_path, filename_output_mha_bin, filename_output_label_split, filename, filename_output_mha)
            sitk.WriteImage(image,filename_output_mha)
            sitk.WriteImage(image,filename_output_mhd)
    print("load_ALL - done")
    return 0
 
def phantom_to_mha_2d(filename_output_mha_phantom, filename_output_mhd_phantom, filepath_input):
    filepath = os.listdir(filepath_input)
    filepath.sort()
    for filename in filepath:
        if filename.startswith("phantom"):
            if filename.endswith(".mhd"): 
                image_path = os.path.join(filepath_input, filename)
                image_arr, spacing, origin  = load_itk(image_path)
                print("im.sh", image_arr.shape)
                summarized_array = np.sum(image_arr, axis=0)
                print("im.sh final", summarized_array.shape)
                im = cv2.normalize(summarized_array, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                for i in range(im.shape[0]):
                    for j in range(im.shape[1]):
                        if im[i, j] == 232:
                            im[i, j] = 100
                im = Image.fromarray(im)
                im.save(filename_output_mha_phantom[:-4]+".png")
                image = sitk.GetImageFromArray(summarized_array)
                sitk.WriteImage(image,filename_output_mha_phantom)
                sitk.WriteImage(image,filename_output_mhd_phantom)
                #image = mhd_to_mha(image_path, filename_output_mha_phantom, filename_output_mha_phantom, filename)
    return 0

def load_ALL_data(filepath_input, filename_output_mha, filename_output_mhd, type_mod):
    print("saved as: ", filename_output_mha)
    print("load_ALL open")
    arrays = []
    filepath = os.listdir(filepath_input)
    filepath.sort()
    for filename in filepath:
        if filename.endswith(type_mod+".mhd"):
            if filename.endswith(type_mod+".mhd"): 
                image_path = os.path.join(filepath_input, filename)
                image_arr, spacing, origin  = load_itk(image_path)
                #print("im.sh", image_arr.shape)
                summarized_array = np.sum(image_arr, axis=0)
                #print("im.sh final", summarized_array.shape)
                arrays.append(summarized_array)
    arrays_combined = np.array(arrays)
    if not type_mod.startswith('BLI'):
        array_final = np.transpose(arrays_combined, axes=(2, 0, 1))
    array_final = resize(array_final, (1,30, 55),anti_aliasing=True)
    arr = np.reshape(array_final, (30, 55))
    arr = Image.fromarray(cv2.normalize(arr, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U))
    arr.save(filename_output_mha[:-4]+".png")
    image_combined = sitk.GetImageFromArray(array_final)
    sitk.WriteImage(image_combined,filename_output_mha)
    sitk.WriteImage(image_combined,filename_output_mhd)
    print("load_ALL - done")
    return 0
 
def create_label_string():
    result_dict = {}
    result_dict["background"] = 0
    for i in range(1, 256):
        result_dict[str(i)] = i
    result_json_string = json.dumps(result_dict)
    with open("output1.json", "w") as f:
        f.write(result_json_string)
    return result_json_string
    
def update_json(name, train_ind, test_ind):
    label_string = create_label_string()
    with open("dataset.json", "r") as jsonFile:
        data = json.load(jsonFile)
    data["numTraining"] = train_ind
    data["numTest"] = test_ind
    with open("dataset.json", "w") as jsonFile:
        json.dump(data, jsonFile)
    dst = os.path.join(name, "/dataset.json")
    shutil.copy2("dataset.json", name)
    print("update_json - done")
    return 0

def for_loop(path_saving, name1, dest_dir_train, dest_dir_train_label, dest_dir_test, dest_dir_test_label):
    #type_mod_data = "-∀l"
    type_mod_data = "-∀l-AWGN"
    type_mod_label = "BLI-oCam-"
    i = 0
    #___________train____________
    print("for_loop TRAIN - starts")
    filenames = os.listdir(dest_dir_train)
    filelabels = os.listdir(dest_dir_train_label)
    filenames.sort()
    filelabels.sort()
    for filename, filelabel in zip(filenames, filelabels):
        image_path_in_itr = os.path.join(dest_dir_train, filename)
        image_path_in_ltr = os.path.join(dest_dir_train_label, filelabel)
        # only defining the nnUNet-specific-names of the files
        if i<10: 
            filename_output_itr = path_saving+"/imagesTr/"+name1+"_00"+str(i)+"_0000.mha"
            filename_output_ltr = path_saving+"/labelsTr/"+name1+"_00"+str(i)+".mha"
            filename_output_mhd_itr = path_saving+"/mhd_data/train/data/"+name1+"_00"+str(i)+"_0000.mhd"
            filename_output_mhd_ltr = path_saving+"/mhd_data/train/label/"+name1+"_00"+str(i)+".mhd"
            filename_output_mha_nonbin = path_saving+"/labels_nonbinaryTr/"+name1+"_00"+str(i)+".mha"
            filename_output_mha_bin = path_saving+"/labels_binaryTr/"+name1+"_00"+str(i)+".mha"
            filename_output_mha_self = path_saving+"/labels_selfTr/"+name1+"_00"+str(i)+".mha"
            filename_output_mha_phantom = path_saving+"/phantom/train/"+name1+"_00"+str(i)+".mha"
            filename_output_mhd_phantom = path_saving+"/phantom/train/"+name1+"_00"+str(i)+".mhd"
        if 10 <= i < 100:
            filename_output_itr = path_saving+"/imagesTr/"+name1+"_0"+str(i)+"_0000.mha"
            filename_output_ltr = path_saving+"/labelsTr/"+name1+"_0"+str(i)+".mha"
            filename_output_mhd_itr = path_saving+"/mhd_data/train/data/"+name1+"_0"+str(i)+"_0000.mhd"
            filename_output_mhd_ltr = path_saving+"/mhd_data/train/label/"+name1+"_0"+str(i)+".mhd"
            filename_output_mha_nonbin = path_saving+"/labels_nonbinaryTr/"+name1+"_0"+str(i)+".mha"
            filename_output_mha_bin = path_saving+"/labels_binaryTr/"+name1+"_0"+str(i)+".mha"
            filename_output_mha_self = path_saving+"/labels_selfTr/"+name1+"_0"+str(i)+".mha"
            filename_output_mha_phantom = path_saving+"/phantom/train/"+name1+"_0"+str(i)+".mha"
            filename_output_mhd_phantom = path_saving+"/phantom/train/"+name1+"_0"+str(i)+".mhd"
        filename_output_label_split = path_saving+"/mhd_data/split/"
        
        print("image_Train starts")
        print("Filename:", filename)
        load_ALL_data(image_path_in_itr, filename_output_itr, filename_output_mhd_itr, type_mod_data)
        phantom_to_mha_2d(filename_output_mha_phantom, filename_output_mhd_phantom, image_path_in_itr)
        print("label_Train starts")  
        print("Filelabel:", filelabel)      
        load_ALL_label(image_path_in_ltr, filename_output_ltr, filename_output_mha_self, type_mod_label, filename_output_label_split, filename_output_mha_nonbin, filename_output_mha_bin, filename_output_mha_self)
        i += 1
        
    k = i
    #__________test__________
    print("for_loop TEST - starts")
    filenames = os.listdir(dest_dir_test)
    filelabels = os.listdir(dest_dir_test_label)
    filenames.sort()
    filelabels.sort()
    for filename, filelabel in zip(filenames, filelabels):
        image_path_in_its = os.path.join(dest_dir_test, filename)
        image_path_in_lts = os.path.join(dest_dir_test_label, filelabel)
        # only defining the nnUNet-specific-names of the files
        if k<10: 
            filename_output_its = path_saving+"/imagesTs/"+name1+"_00"+str(k)+"_0000.mha" 
            filename_output_lts = path_saving+"/test_label/"+name1+"_00"+str(k)+".mha" 
            filename_output_mhd_its = path_saving+"/mhd_data/test/data/"+name1+"_00"+str(k)+"_0000.mhd"
            filename_output_mhd_lts = path_saving+"/mhd_data/test/label/"+name1+"_00"+str(k)+".mhd"
            filename_output_mha_nonbin = path_saving+"/labels_nonbinaryTr/"+name1+"_00"+str(k)+".mha"
            filename_output_mha_bin = path_saving+"/labels_binaryTr/"+name1+"_00"+str(k)+".mha"
            filename_output_mha_self = path_saving+"/labels_selfTr/"+name1+"_00"+str(k)+".mha"
            filename_output_mha_phantom = path_saving+"/phantom/test/"+name1+"_00"+str(k)+".mha"
            filename_output_mhd_phantom = path_saving+"/phantom/test/"+name1+"_00"+str(k)+".mhd"

        if 10 <= k < 100:
            filename_output_its = path_saving+"/imagesTs/"+name1+"_0"+str(k)+"_0000.mha"
            filename_output_lts = path_saving+"/test_label/"+name1+"_0"+str(k)+".mha"
            filename_output_mhd_its = path_saving+"/mhd_data/test/data/"+name1+"_0"+str(k)+"_0000.mhd"
            filename_output_mhd_lts = path_saving+"/mhd_data/test/label/"+name1+"_0"+str(k)+".mhd"
            filename_output_mha_nonbin = path_saving+"/labels_nonbinaryTr/"+name1+"_0"+str(k)+".mha"
            filename_output_mha_bin = path_saving+"/labels_binaryTr/"+name1+"_0"+str(k)+".mha"
            filename_output_mha_phantom = path_saving+"/phantom/test/"+name1+"_0"+str(k)+".mha"
            filename_output_mhd_phantom = path_saving+"/phantom/test/"+name1+"_0"+str(k)+".mhd"
            filename_output_mha_self = path_saving+"/labels_selfTr/"+name1+"_0"+str(k)+".mha"
        k += 1   
        filename_output_label_split = path_saving+"/mhd_data/split/"
        print("load_ALL for image_Test starts")
        print("Filename:", filename)
        load_ALL_data(image_path_in_its, filename_output_its, filename_output_mhd_its, type_mod_data)
        phantom_to_mha_2d(filename_output_mha_phantom, filename_output_mhd_phantom, image_path_in_its)
        print("load_ALL for label_Test starts")        
        print("Filelabel:", filelabel)
        load_ALL_label(image_path_in_lts, filename_output_lts, filename_output_mha_self, type_mod_label, filename_output_label_split, filename_output_mha_nonbin, filename_output_mha_bin, filename_output_mha_self)
        #load_ALL_data(image_path_in_lts, filename_output_lts, filename_output_mha_self, type_mod_label)


    update_json(path_saving,i,k-i)  
    print("for_loop - done")
    return 0

def make_dir(name_file):
    os.makedirs(name_file, exist_ok=True)
    os.makedirs(name_file+"/imagesTr", exist_ok=True)
    os.makedirs(name_file+"/labelsTr", exist_ok=True)
    os.makedirs(name_file+"/imagesTs", exist_ok=True)
    os.makedirs(name_file+"/test_label", exist_ok=True)
    os.makedirs(name_file+"/mhd_data/train/data", exist_ok=True)
    os.makedirs(name_file+"/mhd_data/train/label", exist_ok=True)
    os.makedirs(name_file+"/mhd_data/split", exist_ok=True)
    os.makedirs(name_file+"/mhd_data/test/data", exist_ok=True)
    os.makedirs(name_file+"/mhd_data/test/label", exist_ok=True)
    os.makedirs(name_file+"/labels_nonbinaryTr", exist_ok=True)
    os.makedirs(name_file+"/labels_binaryTr", exist_ok=True)
    os.makedirs(name_file+"/labels_selfTr", exist_ok=True)
    os.makedirs(name_file+"/phantom/train/", exist_ok=True)
    os.makedirs(name_file+"/phantom/test/", exist_ok=True)
    
    print("directories for nnUNet2 were made")
    return 0

#_____________________EDIT_____________________

#_________organize simulation-output data according to label, training and testing data______________
directory_home = '/home/svea/lipros/lipros_training/training_set_06/'
source_directory_simulation = directory_home
source_directory_label = source_directory_simulation+'2024-03-25-16-08-22-lipros-30-sphere'
source_directory_data = source_directory_simulation+'svea-2024-03-22-16-06-06-lipros-30-sphere'
#source_directory_data = source_directory_simulation+'test_old'
destination_directory_train = directory_home+'test_oldtrain'
#destination_directory_train = directory_home+'train'
destination_directory_train_label = directory_home+'train_label'
destination_directory_test = directory_home+'test'
destination_directory_test_label = directory_home+'testing_label'

#________prepare data as input for nnUNet2________
path_saving = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset040_FMIposition55'
name_saving = "Dataset040"

make_dir(path_saving)
for_loop(path_saving, name_saving, destination_directory_train, destination_directory_train_label, destination_directory_test, destination_directory_test_label )
print("job done") 
