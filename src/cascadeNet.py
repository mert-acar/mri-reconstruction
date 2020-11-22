import torch
import numpy as np
import torch.nn as nn
from dataset import OCMRDataset

# input shape: [batch_size, 2, Nx, Ny] --> 2 channel input for real and imaginary channels

class ResnetBlock(nn.Module):
    def __init__(self, in_channels=2, num_layers=5, num_filters=64, kernel_size=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.layers = [
            nn.Conv2d(in_channels, num_filters, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        ]
        
        for i in range(1,num_layers - 1):
            self.layers.append(nn.Conv2d(num_filters, num_filters, kernel_size, stride, padding))
            self.layers.append(nn.ReLU(inplace=True))
        
        # Reconstruction layer
        self.layers.append(nn.Conv2d(num_filters, in_channels, kernel_size, stride, padding))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        residual = x['image']
        out = self.layers(x['image'])
        out += residual 
        return {
            'image': out,
            'k': x['k'],
            'mask': x['mask'],
            'full': x['full']
        }


class DataConsistency(nn.Module):
    def __init__(self, noise_level=None):
        super(DataConsistency, self).__init__()
        self.noise = noise_level

    def forward(self, x):
        image = x['image'].permute(0, 2, 3, 1) # prepare for torch.fft

        temp = torch.fft(image, signal_ndim=2, normalized=True)
        
        if self.noise:
            temp = (temp + self.noise * x['k']) / (1 + self.noise)
        else:
            temp = (1 - x['mask']) * temp + x['k']

        temp = torch.ifft(temp, signal_ndim=2, normalized=True)
        temp = temp.permute(0, 3, 1, 2).float() 
        
        return {
            'image': temp,
            'k': x['k'],
            'mask': x['mask'],
            'full': x['full']
        }


class CascadeNetwork(nn.Module):
    def __init__(self, num_cascades=5, num_layers=5, num_filters=64, kernel_size=3, stride=1, padding=1, noise=None):
        super(CascadeNetwork, self).__init__()
        self.block = []
        for i in range(num_cascades):
            self.block.append(ResnetBlock(2, num_layers, num_filters, kernel_size, stride, padding))
            self.block.append(DataConsistency(noise))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    import yaml
    args = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

    network_args = args['network']
    net = CascadeNetwork(**network_args)

    data_args = args['dataset']
    dataset = OCMRDataset(**data_args)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8)
    anan = iter(dataloader)

    sample = next(anan)
    sample['image'] = sample['image'].float().cuda()

    out = net(sample)
