import os
import torch
import numpy as np
from math import ceil
from helpers import *
from scipy.io import loadmat
from numpy.lib.stride_tricks import as_strided


class OCMRDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, fold='train', fraction=0.85, shuffle=None, acceleration_factor=4.0, sample_n=10, acq_noise=0, centred=False, norm='ortho'):
        self.data_path = data_path
        self.acc = acceleration_factor
        self.sample_n = sample_n
        self.noise = acq_noise
        self.centred = centred
        self.norm = norm
        self.files = os.listdir(self.data_path)
        if shuffle:
            np.random.seed(shuffle)
            np.random.shuffle(self.files) 
        if fold == 'train':
            self.files = self.files[:int(len(self.files) * fraction)] 
        else:
            self.files = self.files[int(len(self.files) * fraction):]
    

    def __len__(self):
        return len(self.files)


    def undersample(self, x, mask):
        '''
        Undersample x. FFT2 will be applied to the last 2 axis
        Parameters
        ----------
        x: array_like
            data
        mask: array_like
            undersampling mask in fourier domain
        noise_power: float
            simulates acquisition noise, complex AWG noise.
            must be percentage of the peak signal
        Returns
        -------
        xu: array_like
            undersampled image in image domain. Note that it is complex valued
        x_fu: array_like
            undersampled data in k-space
        '''
        assert x.shape == mask.shape
        # zero mean complex Gaussian noise
        noise_power = self.noise
        nz = np.sqrt(.5)*(np.random.normal(0, 1, x.shape) + 1j * np.random.normal(0, 1, x.shape))
        nz = nz * np.sqrt(noise_power)

        if self.norm == 'ortho':
            # multiplicative factor
            nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
        else:
            nz = nz * np.prod(mask.shape[-2:])

        if self.centred:
            x_f = fft2c(x, norm=self.norm)
            x_fu = mask * (x_f + nz)
            x_u = ifft2c(x_fu, norm=self.norm)
            return x_u, x_fu
        else:
            x_f = fft2(x, norm=self.norm)
            x_fu = mask * (x_f + nz)
            x_u = ifft2(x_fu, norm=self.norm)
            return x_u, x_fu

    def cartesian_mask(self, shape):
        N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
        pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
        lmda = Nx/(2.*self.acc)
        n_lines = int(Nx / self.acc)

        # add uniform distribution
        pdf_x += lmda * 1./Nx

        if self.sample_n:
            pdf_x[Nx//2-self.sample_n//2:Nx//2+self.sample_n//2] = 0
            pdf_x /= np.sum(pdf_x)
            n_lines -= self.sample_n

        mask = np.zeros((N, Nx))
        for i in range(N):
            idx = np.random.choice(Nx, n_lines, False, pdf_x)
            mask[i, idx] = 1

        if self.sample_n:
            mask[:, Nx//2-self.sample_n//2:Nx//2+self.sample_n//2] = 1

        size = mask.itemsize
        mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

        mask = mask.reshape(shape)

        if not self.centred:
            mask = ifftshift(mask, axes=(-1, -2))

        return mask



    def __getitem__(self, idx):
        # Load the data
        data = loadmat(os.path.join(self.data_path, self.files[idx]))['xn']
        data = np.expand_dims(data, 0)
        mask = self.cartesian_mask(data.shape)
        data_und, k_und = self.undersample(data, mask)
        data_gnd = format_data(data)
        data_und = format_data(data_und)
        k_und = format_data(k_und)
        mask = format_data(mask, mask=True)
        
        return {
            'image': data_und,
            'k': k_und.transpose(1,2,0),
            'mask': mask.transpose(1,2,0),
            'full': data_gnd
        }

if __name__ == '__main__':
    import yaml
    args = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    dataset = OCMRDataset(fold='train', **args['dataset'])
    sample = dataset[0]
    print('Sample image shape:', sample['image'].shape)
