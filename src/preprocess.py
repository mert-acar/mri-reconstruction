import os
import math
import numpy as np
from tqdm import tqdm
import scipy.linalg as la
from os.path import join as pjn
from scipy.io import savemat
from read_ocmr import read_ocmr as read
from numpy.fft import fftshift, ifftshift, ifftn, fftn

# Data shape: [kx ky kz coil phase set slice rep avg]

def normalize(img):
    mag = np.abs(img)
    mag /= mag.max()
    pha = np.angle(img)
    r = mag * np.cos(pha)
    i = mag * np.sin(pha)
    return r + (1j * i)

def k2image(kData):
    img = fftshift(ifftn(ifftshift(kData, axes=[0,1]), s=None, axes=[0,1]), axes=[0,1])
    img *= np.sqrt(np.prod(np.take(img.shape, [0,1])))
    return img

def coil_combination(im_coil, w=5):
    image = im_coil.copy()
    [Nx, Ny, Nc] = image.shape
    S = np.zeros((Nx, Ny, Nc), dtype=image.dtype)
    M = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            kernel = image[
                        np.maximum(i-w, 0):np.minimum(i+w, Nx) + 1,
                        np.maximum(j-w, 0):np.minimum(j+w, Ny) + 1, :
                    ].transpose(1,0,2).reshape((-1, Nc))
            kernel = np.matrix(kernel)
            temp = np.conjugate(kernel.H.dot(kernel))
            V, D = la.eig(temp)
            idx = np.argmax(V)
            V = abs(V[idx])
            aD = D[:, idx]
            S[i, j, :] = aD * np.exp(-1j * np.angle(aD[0]))
            M[i, j] = np.sqrt(V)
    mask = (M > 0.1 * np.max(np.abs(M))).astype(np.int)
    for i in range(S.shape[-1]):
        S[:, :, i] = np.multiply(S[:, :, i], mask)
    combined = np.sum(np.multiply(im_coil, np.conjugate(S)),2)
    combined = np.divide(combined, np.sum(np.multiply(S,np.conjugate(S)),2) + 1e-8)
    return combined

def filtered(kData):
    kData = np.mean(kData, axis=len(kData.shape) - 1)
    for i in range(3):
        kData = kData[tuple([slice(None) if (i != len(kData.shape) - 1) else math.floor(kData.shape[i] / 2) for i in range(len(kData.shape)) ])]
    return kData

def pad(array):
    offsets = [(256 - array.shape[dim]) / 2 for dim in range(array.ndim)]
    # Create an array of zeros with the reference shape
    result = np.zeros([int(array.shape[0] + (2 * offsets[0])), int(array.shape[1] + (2 * offsets[1])), array.shape[2]], dtype=array.dtype)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [
        slice(int(offsets[0]), int(offsets[0] + array.shape[0])),
        slice(int(offsets[1]), int(offsets[1] + array.shape[1])),
        slice(None)
    ]
    # Insert the array in the result at the specified offsets
    result[tuple(insertHere)] = array
    return result

if __name__ == '__main__':
    path = pjn('../../ocmr/data', 'fully_sampled')
    for f in tqdm(os.listdir(path)):
        fname = f.split('.')
        if fname[-1] != 'h5':
            continue
        count = 0
        kData, param = read(pjn(path, f))
        kData = filtered(kData)

        #Inverse 2D Fourier Transform
        im_coil = k2image(kData)

        #Remove RO Oversampling
        R0 = im_coil.shape[0]
        im_coil = im_coil[math.floor(R0 / 4) : math.floor(3 * R0 / 4)]
        for z in range(im_coil.shape[2]):
            im = im_coil[:,:,z,:,:]
            for t in range(im.shape[-1]):
                img = im[:,:,:,t]
                img = pad(img)
                combined_im = coil_combination(img)
                combined_im = normalize(combined_im)
                savemat(pjn('../data','preprocessed', fname[0] + '_' + str(z) + '_' + str(t) + '.mat'), {'xn': combined_im})
