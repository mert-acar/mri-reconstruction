import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers import *

def evaluate(model, dataloader, draw=False, **kwargs):
    mse = 0
    ssim_score = 0
    psnr = 0
    num_samples = len(dataloader)
    criterion = torch.nn.MSELoss()
    i = 0
    for batch in tqdm(dataloader, 'Evaluating', total=num_samples):
        i += 1
        for key in batch:
            batch[key] = batch[key].float()
            if kwargs['cuda']:
                batch[key] = batch[key].cuda()

        bsize = batch['image'].shape[0]
        output = model(batch)
        loss = criterion(output['image'], batch['full'])
        mse += loss.item() * 1e3
        pred = back_format(output['image'].detach().numpy())
        grnd = back_format(batch['full'].detach().numpy())
        psnr += complex_psnr(grnd, pred)
        ssim_score += ssim(pred, grnd)
        if draw:
            pred = pred[0]
            grnd = grnd[0]
            err = np.abs(pred - grnd) ** 2
            err -= err.min()
            err /= err.max()
            err *= 255
            err = err.astype(np.uint8)
            err = cv2.applyColorMap(err, 2)
            plt.imsave(os.path.join(kwargs['output_path'], str(i) + '.png'), err)

    mse /= num_samples
    ssim_score /= num_samples
    psnr /= num_samples
    print('Average MSE (10^-3):', mse)
    print('Average SSIM:', np.round(ssim_score * 100, 2))
    print('Average PSNR:', np.round(psnr, 2))

if __name__ == '__main__':
    import yaml
    from models import CascadeNetwork
    from dataset import OCMRDataset
    args = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

    state_dict = torch.load('../logs/sanity_check2/checkpoint', map_location='cuda' if args['train']['cuda'] else 'cpu')
    state_dict = state_dict['model']
    model = CascadeNetwork(**args['network']) 
    model.load_state_dict(state_dict)
    dataloader = torch.utils.data.DataLoader(
        OCMRDataset(fold='test', **args['dataset']),
        batch_size=2
    )
    evaluate(model, dataloader, draw=True, **args['train'])
