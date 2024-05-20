import os
import random
from PIL import Image
import numpy as np

# Function to get a list of PNG files in a directory
def get_png_files_in_directory(directory):
    files = []
    filepath = os.listdir(directory)
    filepath.sort()
    for file in filepath:
        if file.endswith(".png"):
            files.append(os.path.join(directory, file))
    return files

# Function to choose two random files from a list
def choose_random_files(files):
    return random.sample(files, 2)

# Function to read PNG images and convert them to arrays
def png_to_array(image_path):
    image = Image.open(image_path)
    return np.array(image)

# Function to add arrays together
def add_arrays(arrays):
    result = arrays[0]/2
    for array in arrays[1:]:
        result = np.add(result, array/2)
    return result

# Function to save array as PNG image
def array_to_png(array, output_path):
    image = Image.fromarray(array.astype('uint8'))
    image.save(output_path)

# Example directories
directory1 = "/home/svea/Documents/nnUNetv2/nnUNet_raw/adding_two/imagesTr_Add"
directory2 = "/home/svea/Documents/nnUNetv2/nnUNet_raw/adding_two/labelsTr_Add"
directory3 = "/home/svea/Documents/nnUNetv2/nnUNet_raw/adding_two/phantom_Add"

# Get list of PNG files in each directory
files1 = get_png_files_in_directory(directory1)
files2 = get_png_files_in_directory(directory2)
files3 = get_png_files_in_directory(directory3)

# Check if both directories have the same number of images
if len(files1) != len(files2) != len(file3):
    raise ValueError("Both directories must have the same number of images.")

# Run the loop 20 times
for i in range(20):
    random_files1 = []
    random_files2 = []
    random_files3 = []

    # Choose two random files from the first directory
    random_files1 = choose_random_files(files1)
    print(type(random_files1))
    
    # Corresponding files from the second directory
    index = files1.index(random_files1[0])
    index2 = files1.index(random_files1[1])

    #files1.remove(random_files1[0])
    random_files2 = [files2[index], files2[index2]]
    #files2.remove(random_files2[0])
    random_files3 = [files3[index], files3[index2]]
    #files3.remove(random_files3[0])

    print(f"Iteration {i+1}:")
    print("Randomly chosen files from directory 1:")
    print(random_files1)
    print("Randomly chosen files from directory 2:")
    print(random_files2)
    print("Randomly chosen files from directory 3:")
    print(random_files3)
    print()  # Add an empty line for readability

    # Read PNG images as arrays
    arrays1 = [png_to_array(image_path) for image_path in random_files1]
    arrays2 = [png_to_array(image_path) for image_path in random_files2]
    arrays3 = [png_to_array(image_path) for image_path in random_files3]

    # Add arrays together
    sum_array1 = add_arrays(arrays1)
    sum_array2 = add_arrays(arrays2)
    sum_array3 = add_arrays(arrays3)

    directory_aim1 = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset039_add_two/imagesTr'
    directory_aim2 = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset039_add_two/labelsTr'
    directory_aim3 = '/home/svea/Documents/nnUNetv2/nnUNet_raw/Dataset039_add_two/phantoms'
    # Save the result as a PNG image
    if i < 10:
        output_path1 = f"{directory_aim1}/Dataset038_00{70+i}_0000.png"
        output_path2 = f"{directory_aim2}/Dataset038_00{70+i}.png"
        output_path3 = f"{directory_aim3}/Dataset038_00{70+i}.png"
    if i < 20:
        output_path1 = f"{directory_aim1}/Dataset038_0{80+(i-10)}_0000.png"
        output_path2 = f"{directory_aim2}/Dataset038_0{80+(i-10)}.png"
        output_path3 = f"{directory_aim3}/Dataset038_0{80+(i-10)}.png"
    
    array_to_png(sum_array1, output_path1)
    array_to_png(sum_array2, output_path2)
    array_to_png(sum_array3, output_path3)
    print(f"Sum image saved as {output_path1}")
    print()  
