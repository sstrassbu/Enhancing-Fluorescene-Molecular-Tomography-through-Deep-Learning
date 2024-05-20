from PIL import Image, ImageChops
import numpy as np
from PIL import Image
import os


def artefacts(image1, label):
    image1 = Image.open(image1)
    image1 = image1.convert("RGB")
    array1 = np.array(image1)
    positions = np.where(np.all(array1 == [0, 0, 0], axis=-1))
    for pos in zip(*positions):
        #print(f"Pixel with value [0, 0, 0] found at position: {pos[0]}, {pos[1]}")
        if pos[1] != 0 and pos[1] != array1.shape[0] - 1:
            #array1[pos[0], pos[1]] = [230, 230, 230]
            print("here", pos)

    mask = np.all(array1 == [0, 0, 0], axis=-1)
    mask[:,0:10] = False  # Exclude first row
    mask[:,-10:-1] = False  # Exclude last row
    mask[:,-1] = False
    array1[mask] = [240, 240, 240]
    mask = np.all(array1 == [0, 0, 0], axis=-1)
    mask[:,0] = False  # Exclude first row
    mask[:,10:-10] = False  # Exclude last row
    mask[:,-1] = False  # Exclude last row
    array1[mask] = [120,120,120]
    #print("ar2",array1[-16])
    result_image = Image.fromarray(array1)
    result_image.save(label)

directory_path1 = "doubleall/CT/"
images_folder1 = [f for f in os.listdir(directory_path1) if f.endswith('.png')]
#images_folder1 = [f for f in os.listdir(directory_path1) if f.endswith('.png') and not f.endswith("subtract.png")]
images_folder1.sort()
print(images_folder1)
for filename1 in images_folder1:
    #label = directory_path1+filename1[:-4]+"subtract.png"
    label = "artefacts/"+filename1
    image1 = os.path.join(directory_path1, filename1)
    artefacts(image1, label)
    print(filename1)

directory_path1 = "singleall/CT/"
images_folder1 = [f for f in os.listdir(directory_path1) if f.endswith('.png')]
#images_folder1 = [f for f in os.listdir(directory_path1) if f.endswith('.png') and not f.endswith("subtract.png")]
images_folder1.sort()
print(images_folder1)
for filename1 in images_folder1:
    #label = directory_path1+filename1[:-4]+"subtract.png"
    label = "artefacts/"+filename1
    image1 = os.path.join(directory_path1, filename1)
    artefacts(image1, label)
    print(filename1)



