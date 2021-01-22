from os.path import join as pjn
import matplotlib.pyplot as plt
import numpy as np

expPath = '../logs/sanity_check2/'

"""
# -- Graphs and shit
with open(pjn(expPath, 'log_train.txt'), 'r') as f:
    log_data = f.readlines()

headers = [header.strip() for header in log_data[0].split('    ')][1:]
data = [l.strip().split() for l in log_data[2:]]
data = np.array([list(map(float, l)) for l in data])

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(data[:,0], data[:,1], label='train loss')
ax1.plot(data[:,0], data[:,2], label='test loss')
ax1.grid(True)
ax1.legend()
ax1.set_title('Train and Test Losses')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss (1e-3)')
ax1.set_ylim((0.2, 1))

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(data[:,0], data[:,4], label='Test PSNR')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('PSNR (dB)')
ax2.grid(True)
ax2.legend()

plt.show()
"""

# -- Error maps, SSIM calculations, PSNR... we got'em all.
import yaml
import torch
from tqdm import tqdm
from helpers import *
from dataset import OCMRDataset
from models import CascadeNetwork

with open(pjn(expPath, 'ExperimentSummary.yaml'), 'r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(pjn(expPath, 'checkpoint'), map_location=device)['model']
model = CascadeNetwork(**args['network'])
model.load_state_dict(state_dict)
model.to(device)

dataloader = torch.utils.data.DataLoader(
    OCMRDataset(fold='test', **args['dataset']),
    batch_size=1
)

criterion = torch.nn.MSELoss()

mse = 0
ssim_score = 0
psnr = 0
for i, sample in tqdm(enumerate(dataloader)):
    for key in sample:
        sample[key] = sample[key].float().to(device) 
    output = model(sample)
    loss = criterion(output['image'], sample['full'])
    grnd = back_format(sample['full']).cpu().detach().numpy()
    pred = back_format(output['image']).cpu().detach().numpy()
    psnr += complex_psnr(grnd, pred)
    mse += loss.item()
    ssim_score += ssim(grnd, pred)
    err = (np.abs(grnd - pred) ** 2).squeeze()
    plt.imsave(str(i) + '.png', err, cmap='jet')
