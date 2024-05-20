from PIL import Image, ImageChops
import numpy as np
from PIL import Image
import os
 
def redhot(image, label):
    image = Image.open(image)
    image = image.convert("RGB")
    array = np.array(image)
    print(array[25])
    #array[np.all(array == [255,255,255], axis=2)] = [183, 18, 31]
    array[np.all(array == [255,255,255], axis=2)] = [247, 98, 98]
    result_image = Image.fromarray(array)
    result_image.save(label)

def subtract(image1, image2, label):
    image1 = Image.open(image1)
    image2 = Image.open(image2)
    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")
    array1 = np.array(image1)
    array2 = np.array(image2)
    #print("ar1",array1[-1])
    #print("ar2",array2[-1])
    if array1.shape != array2.shape:
        raise ValueError("Images must have the same shape")
    # Subtract array2 from array1
    diff_array = np.abs(array2 - array1)
    #print("shapediff",diff_array.shape)
    #print("diff",diff_array[-15])
    # Create a new image array for visualization
    mask = np.all(diff_array == [0, 0, 0], axis=-1)
    diff_array[mask] = [255, 255, 255]
    mask1 = np.all(diff_array == [1, 1, 1], axis=-1)
    diff_array[mask1] = [34,139,34]
    mask4 = np.all(diff_array == [128, 128, 128], axis=-1)
    diff_array[mask4] = [34,139,34]
    mask2 = np.all(diff_array == [127, 127, 127], axis=-1)
    diff_array[mask2] = [255, 51, 51]
    mask3 = np.all(diff_array == [129,129,129], axis=-1)
    diff_array[mask3] = [0, 128, 255]
    print("diff5",diff_array[14])
    result_image = Image.fromarray(diff_array)
    result_image.save(label)

directory_path1 = "doubleallsubtract/38/"
#directory_path1 = "doubleallsubtract/41/"
directory_path2 = "labelsubtract/"
images_folder1 = [f for f in os.listdir(directory_path1) if f.endswith('.png') and not f.endswith("subtract.png")]
images_folder2 = [f for f in os.listdir(directory_path2) if f.endswith('.png')]
images_folder1.sort()
images_folder2.sort()
for filename1,filename2 in zip(images_folder1,images_folder2):
    #label = directory_path1+filename1[:-4]+"subtract.png"
    label = "doubleallsubtract/subtract/"+filename1[:-4]+"subtract.png"
    image1 = os.path.join(directory_path1, filename1)
    image2 = os.path.join(directory_path2, filename2)
    subtract(image1, image2, label)
    print(filename1,filename2)
