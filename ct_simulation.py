import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import PIL as pillow, numpy
import os

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, resize
from skimage.transform import iradon
from skimage.transform.radon_transform import _get_fourier_filter
from skimage import io  

def do(path, filename):
    fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2,3, figsize=(9, 8))
    image = io.imread(path)
    dimensions = image.shape
    # Check if the array is three-dimensional
    if len(dimensions) == 3:
        # Convert the image to grayscale - in case shape is (x, y, 3)
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    image4 = Image.fromarray(image)
    #image4.save('output_addtwo/'+filename[:-4]+'init.png')
    ax1.set_title(filename[:4]+"\ninitial phantom")
    ax1.imshow(image4, cmap=plt.cm.Greys_r)
    print("image shape: ", dimensions)
    print("image type: ", type(image))
    print("image shape after: ", image.shape)
    image1 = rescale(image, scale=0.75, mode='reflect', channel_axis=None)# 0.75 for 160 
    theta = np.linspace(0., 360., max(image1.shape), endpoint=False)
    sinogram_init = radon(image1, theta=theta)
    sino= np.transpose(sinogram_init, axes=(1, 0))
    sino1 = resize(sino, (30, 55),anti_aliasing=True)
    sino1 = cv2.normalize(sino1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    image1 = Image.fromarray(sino1)
    image1.save('output_addtwo/'+filename[:-4]+'_0.png')
    ax3.set_title("initial sinogram (transposed)\ninput for reconstruction")
    ax3.imshow(image1, cmap=plt.cm.Greys_r)
    image_orig = image


    # roation 1 
    rotated_array = np.rot90(image_orig, k=1)
    image3 = Image.fromarray(rotated_array)
    #image3.save('output_addtwo/'+filename[:-4]+'turned.png')
    ax2.set_title("turned phantom")
    ax2.imshow(image3, cmap=plt.cm.Greys_r)
    image = rescale(rotated_array, scale=0.75, mode='reflect', channel_axis=None)#oder 0.75 for 160 ; 214 auf 360
    print("image shape RESCALED: ", image.shape)
    theta = np.linspace(0., 360., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    sino= np.transpose(sinogram, axes=(1, 0))
    sino1 = resize(sino, (30, 55),anti_aliasing=True)
    sino1 = cv2.normalize(sino1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    print("sinogram shape: ", sino1.shape)
    print("sino type: ", type(sino1))
    image1 = Image.fromarray(sino1)
    image1.save('output_addtwo/'+filename[:-4]+'_1.png')
    ax4.set_title("turned sinogram (transposed)\ninput for reconstruction")
    ax4.imshow(image1, cmap=plt.cm.Greys_r)

    # roation -1 
    rotated_array = np.rot90(image_orig, k=-1)
    image3 = Image.fromarray(rotated_array)
    #image3.save('output_addtwo/'+filename[:-4]+'turned.png')
    ax2.set_title("turned phantom")
    ax2.imshow(image3, cmap=plt.cm.Greys_r)
    image = rescale(rotated_array, scale=0.75, mode='reflect', channel_axis=None)#oder 0.75 for 160 ; 214 auf 360
    print("image shape RESCALED: ", image.shape)
    theta = np.linspace(0., 360., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    sino= np.transpose(sinogram, axes=(1, 0))
    sino1 = resize(sino, (30, 55),anti_aliasing=True)
    sino1 = cv2.normalize(sino1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    print("sinogram shape: ", sino1.shape)
    print("sino type: ", type(sino1))
    image1 = Image.fromarray(sino1)
    image1.save('output_addtwo/'+filename[:-4]+'_2.png')

    # roation -2 
    rotated_array = np.rot90(image_orig, k=-2)
    image3 = Image.fromarray(rotated_array)
    #image3.save('output_addtwo/'+filename[:-4]+'turned.png')
    ax2.set_title("turned phantom")
    ax2.imshow(image3, cmap=plt.cm.Greys_r)
    image = rescale(rotated_array, scale=0.75, mode='reflect', channel_axis=None)#oder 0.75 for 160 ; 214 auf 360
    print("image shape RESCALED: ", image.shape)
    theta = np.linspace(0., 360., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    sino= np.transpose(sinogram, axes=(1, 0))
    sino1 = resize(sino, (30, 55),anti_aliasing=True)
    sino1 = cv2.normalize(sino1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    print("sinogram shape: ", sino1.shape)
    print("sino type: ", type(sino1))
    image1 = Image.fromarray(sino1)
    image1.save('output_addtwo/'+filename[:-4]+'_3.png')
    return 0


# Specify the directory path
directory_path = "input/"

# Iterate through the files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".png"):
        # Get the full file path
        full_path = os.path.join(directory_path, filename)
        # Call the function with the file path
        do(full_path, filename)


