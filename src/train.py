import os
import yaml
import torch
import warnings
import numpy as np
from time import time
from tqdm import tqdm
from evaluate import evaluate
from tabulate import tabulate
from dataset import OCMRDataset
from models import CascadeNetwork
from shutil import rmtree, copyfile
from helpers import back_format, complex_psnr, ssim

warnings.filterwarnings('ignore')

def checkpoint(model, optimizer, path):
    print('SAVING MODEL...')
    dict2save = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(dict2save, os.path.join(path, 'checkpoint'))
    

def train_model(model, dataloaders, optimizer, criterion, args):
    before = time()
    best_loss = 9999999
    best_epoch = -1
    last_train_loss = -1
    loss_table = []
    for epoch in range(1, args['num_epochs'] + 1):
        print('-' * 20)
        print('Epoch {} / {}'.format(epoch, args['num_epochs']))
        for phase in ['train', 'test']:
            running_error = 0 
            base_psnr = 0
            test_psnr = 0
            ssim_score = 0
            for batch in tqdm(dataloaders[phase], phase, total=(len(dataloaders[phase]))):
                for key in batch.keys():
                    batch[key] = batch[key].float()
                    if args['cuda']:
                        batch[key] = batch[key].cuda()
                bsize = batch['image'].shape[0]
                output = model(batch)
                optimizer.zero_grad()
                loss = criterion(output['image'], batch['full'])
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                else:
                    for im_i, und_i, pred_i in zip(back_format(batch['full'].cpu().detach().numpy()), 
                                                   back_format(batch['image'].cpu().detach().numpy()),
                                                   back_format(output['image'].cpu().detach().numpy())):
                        base_psnr += complex_psnr(im_i, und_i)
                        test_psnr += complex_psnr(im_i, pred_i)
                        ssim_score += ssim(im_i, pred_i)
                running_error += loss.item() * bsize * 1000
            epoch_loss = running_error / len(dataloaders[phase].dataset)
            if phase == 'train':
                last_train_loss = epoch_loss
                print('Training Loss:\t\t{:.9f}'.format(last_train_loss))
            else:
                base_psnr /= len(dataloaders[phase].dataset) 
                test_psnr /= len(dataloaders[phase].dataset) 
                ssim_score /= len(dataloaders[phase].dataset)
                loss_table.append([epoch, last_train_loss, epoch_loss, base_psnr, test_psnr, ssim_score])
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_epoch = epoch
                    checkpoint(model, optimizer, args['output_path'])
                print('Testing Loss:\t\t{:.9f}'.format(epoch_loss))
                print('Base PSNR:\t\t{:.9f}'.format(base_psnr))
                print('Test PSNR:\t\t{:.9f}'.format(test_psnr))
                print('SSIM:\t\t\t{:.3f}'.format(ssim_score))
        if(args['early_stop'] != 0) and ((epoch - best_epoch) >= args['early_stop']):
            print('No improvements in', args['early_stop'], 'epochs, break...')
            break
    t = tabulate(loss_table, headers=['Epoch', 'Train Loss', 'Test Loss', 'Base PSNR', 'Test PSNR', 'SSIM'])
    with open(os.path.join(args['output_path'], 'log_train.txt'), 'w') as f:
        f.write(t)
    elapsed = time() - before
    print("Training complete in {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))
    state_dict = torch.load(os.path.join(args['output_path'], 'checkpoint'))['model']
    model.load_state_dict(state_dict)
    evaluate(model, dataloaders['test'], **args)

def train(args): 
    train_args = args['train']
    
    try:
        rmtree(train_args['output_path'])
    except Exception:
        pass
    os.mkdir(train_args['output_path'])
    copyfile('config.yaml', os.path.join(train_args['output_path'], 'ExperimentSummary.yaml')) 

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            OCMRDataset(fold='train', **args['dataset']),
            batch_size=train_args['batch_size'],
            shuffle=True,
        ),
        'test': torch.utils.data.DataLoader(
            OCMRDataset(fold='test', **args['dataset']),
            batch_size=train_args['batch_size'],
            shuffle=False,
        ),
    }

    model = CascadeNetwork(**args['network'])
    if train_args['cuda']:
        model = torch.nn.DataParallel(model)
        model.cuda()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_args['learning_rate'],
        betas=(train_args['b_1'], train_args['b_2']),
        weight_decay=train_args['l2']
    )

    criterion = torch.nn.MSELoss()
    print('Training starting with', len(dataloaders['train'].dataset),
          'training and', len(dataloaders['test'].dataset), 'testing data...')
    train_model(model, dataloaders, optimizer, criterion, train_args)


if __name__ == '__main__':
    args = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader) 
    train(args)
