import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import PIL as pillow, numpy
import os
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, resize
from skimage.transform import iradon
from skimage.transform.radon_transform import _get_fourier_filter
from skimage import io  

def doCT(path, filename, dir_path, mode,mode1,vgl):
    image = io.imread(path)
    dimensions = image.shape
    if len(dimensions) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    print("image shape: ", dimensions)
    print("image shape after: ", image.shape)
    sinogram = image
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8, 4.5))
    ax1.set_title("input sinogram")
    ax1.imshow(sinogram, cmap=plt.cm.Greys_r)
    if mode == "transpose":
        image = np.transpose(image, axes=(1, 0))
        print("image shape TRANSPOSED: ")
    vgl = io.imread(vgl)
    #__________Reconstruction - Filtered back projection__________
    print("sinogram: ", type(sinogram))
    print("im: ", type(image))
    zero = np.zeros(image.shape[1])
    if mode1 == "not_ct":
        print("image vstsackdone")
        image = np.vstack((zero, image, zero))
        image = np.vstack((zero, image, zero))
        image = np.vstack((zero, image, zero))
        image = np.vstack((zero, image, zero))
        image = np.vstack((zero, image, zero))
        image = np.vstack((zero, image, zero))
        image = resize(image, (55, 30),anti_aliasing=True)

    print("image shape RESCALED: ", image.shape)
    theta = np.linspace(0., 360., image.shape[1], endpoint=False)
    print(image.dtype)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    print("theta: ", theta.shape)
    ax2.set_title("sinogram (transposed\ninput for reconstruction")
    ax2.imshow(image, cmap=plt.cm.Greys_r)
    ax4.set_title(f"original phantom {vgl.shape}")
    ax4.imshow(vgl, cmap=plt.cm.Greys_r)
    fig.tight_layout()
    # filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
    reconstruction_fbp = iradon(image, theta=theta, filter_name='ramp')
    ax3.set_title("Reconstruction\nFiltered back projection")
    reconstruction_fbp = cv2.normalize(reconstruction_fbp, None, 0, 255, cv2.NORM_MINMAX)
    ax3.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    fig.tight_layout()
    rec55 = reconstruction_fbp
    figrec, (ax1r) = plt.subplots(1,1, figsize=(8, 4.5))
    ax1r.imshow(rec55, cmap=plt.cm.Greys_r)
    figrec.tight_layout()
    plt.savefig(dir_path+filename[:-4]+'recon.png')
    #plt.show()
    plt.close()
    return 0


def doBLI(path, filename, dir_path, mode,mode1,vgl):
    image = io.imread(path)
    dimensions = image.shape
    if len(dimensions) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    print("image shape: ", dimensions)
    print("image shape after: ", image.shape)
    sinogram = image
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8, 4.5))
    ax1.set_title("input sinogram")
    ax1.imshow(sinogram, cmap=plt.cm.Greys_r)
    if mode == "transpose":
        image = np.transpose(image, axes=(1, 0))
        print("image shape TRANSPOSED: ")
    vgl = io.imread(vgl)
    #__________Reconstruction - Filtered back projection__________
    print("sinogram: ", type(sinogram))
    print("im: ", type(image))
    zero = np.zeros(image.shape[1])
    if mode1 == "not_ct":
        print("image vstsackdone")
        image = np.vstack((zero, image, zero))
        image = np.vstack((zero, image, zero))
        image = np.vstack((zero, image, zero))
        image = np.vstack((zero, image, zero))
        image = np.vstack((zero, image, zero))
        image = np.vstack((zero, image, zero))
        image = resize(image, (55, 30),anti_aliasing=True)
    print("image shape RESCALED: ", image.shape)
    theta = np.linspace(0., 360., image.shape[1], endpoint=False)
    print(image.dtype)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    print("theta: ", theta.shape)
    ax2.set_title("sinogram (transposed\ninput for reconstruction")
    ax2.imshow(image, cmap=plt.cm.Greys_r)
    ax4.set_title(f"original phantom {vgl.shape}")
    ax4.imshow(vgl, cmap=plt.cm.Greys_r)
    fig.tight_layout()
    reconstruction_fbp = iradon(image, theta=theta, filter_name='ramp')
    ax3.set_title("Reconstruction\nFiltered back projection")
    reconstruction_fbp = cv2.normalize(reconstruction_fbp, None, 0, 255, cv2.NORM_MINMAX)
    ax3.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    fig.tight_layout()
    rec55 = reconstruction_fbp

    reconstruction_fbp = iradon(image, theta=theta, filter_name='hann', interpolation='linear', preserve_range=True)
    reconstruction_fbp = cv2.normalize(reconstruction_fbp, None, 0, 255, cv2.NORM_MINMAX)
    print('pre',reconstruction_fbp[0,0])
    boarder = reconstruction_fbp[0,0]

    #reconstruction_fbp[reconstruction_fbp > 100] = 250
    reconstruction_fbp[reconstruction_fbp > 150] = 250 #for DS37
    for i in range(0, reconstruction_fbp.shape[0]):
        k = 0
        for j in range(0, reconstruction_fbp.shape[1]):
            if k == 0: 
                if not reconstruction_fbp[i, j] == boarder:
                    reconstruction_fbp[i, :j] = 0
                    k = 1
    reconstruction_fbp = np.flip(reconstruction_fbp, axis=1)
    for i in range(0, reconstruction_fbp.shape[0]):
        k = 0
        for j in range(0, reconstruction_fbp.shape[1]):
            if k == 0:
                if not reconstruction_fbp[i, j] == boarder:
                    reconstruction_fbp[i, :j] = 0
                    k = 1
    reconstruction_fbp = np.flip(reconstruction_fbp, axis=1)
    print(reconstruction_fbp[0])
    #reconstruction_fbp[reconstruction_fbp > 190] = 250
    reconstruction_fbp[ (reconstruction_fbp > 0) & (reconstruction_fbp <= 100) ] = 100
    #reconstruction_fbp[ (reconstruction_fbp > 0) & (reconstruction_fbp <= 150) ] = 100 #for DS37
    ax3.set_title("Reconstruction\nFiltered back projection")
    ax3.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    fig.tight_layout()
    #plt.savefig(dir_path+filename[:-4]+'ramp_recon_filter.png')
    
    fig2rec, (ax2r) = plt.subplots(1,1, figsize=(8, 4.5))
    #reconstruction_fbp = np.rot90(reconstruction_fbp, k=2)
    ax2r.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    fig2rec.tight_layout()
    plt.savefig(dir_path+filename[:-4]+'filterrecon.png')
    figrec, (ax1r) = plt.subplots(1,1, figsize=(8, 4.5))
    ax1r.imshow(rec55, cmap=plt.cm.Greys_r)
    figrec.tight_layout()
    plt.savefig(dir_path+filename[:-4]+'recon.png')
    #plt.show()
    plt.close()
    return 0

# Specify the directory path
directory_path = "singleall/BLI/37/"
mode = "transpose"#when input is straight a sinogram 
mode1 = "nnot_ct"#when data is NOT ct data (not 026, 023)
vgl = 'practice/Dataset032_045.png'
for filename in os.listdir(directory_path):
    if filename.endswith(".png") and not filename.endswith("recon.png"):
        full_path = os.path.join(directory_path, filename)
        #doBLI(full_path, filename, directory_path, mode , mode1, vgl) #for DS37

directory_path = "doubleall/BLI/"
vgl = 'practice/Dataset032_045.png'
for filename in os.listdir(directory_path):
    if filename.endswith(".png") and not filename.endswith("recon.png"):
        full_path = os.path.join(directory_path, filename)
        #doBLI(full_path, filename, directory_path, mode , mode1, vgl)

#directory_path = "singleall/CT/"
directory_path = "artefacts/artefacts/"
vgl = 'practice/Dataset032_045.png'
for filename in os.listdir(directory_path):
    if filename.endswith(".png") and not filename.endswith("recon.png"):
        full_path = os.path.join(directory_path, filename)
        doBLI(full_path, filename, directory_path, mode , mode1, vgl)

directory_path = "doubleall/CT/"
vgl = 'practice/Dataset032_045.png'
for filename in os.listdir(directory_path):
    if filename.endswith(".png") and not filename.endswith("recon.png"):
        full_path = os.path.join(directory_path, filename)
        #doCT(full_path, filename, directory_path, mode , mode1, vgl)

directory_path = "practice/ctvgl/"
vgl = 'practice/Dataset032_045.png'
for filename in os.listdir(directory_path):
    if filename.endswith(".png") and not filename.endswith("recon.png"):
        full_path = os.path.join(directory_path, filename)
        #doBLI(full_path, filename, directory_path, mode , mode1, vgl)
