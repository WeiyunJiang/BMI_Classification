#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:13:11 2021

@author: weiyunjiang
"""
import numpy as np
import torch
import torch.utils.data.distributed
import random
import os
import utils

from tqdm import tqdm
from models import VGG_16, Alex_Net, Efficient_Net, SE_Net
from dataio import Breast_Dataset
from args import breast_arg
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

def test(model, test_data_loader, args):
    with torch.no_grad():
        total_acc = []
        total_pred = []
        total_label = []
        model.eval()
        for step, batch in tqdm(enumerate(test_data_loader)):  
            image, label = batch['image'], batch['label']
            image = image.to(device)
            label = label.to(device)
            
            pred = model(image.float())
            pred = pred.squeeze(-1)
            label = label.type(torch.FloatTensor)
            
            label = label.to(device)

            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            total_pred.append(pred)
            total_label.append(label)
            correct_results_sum = (pred == label).sum().float()
        
            acc = correct_results_sum/pred.shape[0]
            total_acc.append(acc.clone().detach().cpu().numpy())
        tn, fp, fn, tp = confusion_matrix(total_pred, total_label).ravel()
        tqdm.write("acc: %.4f" 
                   % (np.mean(total_acc)))
        
     
        
        
        
        
        
if __name__ == '__main__': 
    args = breast_arg()
    # Set random seed
    seed = 40
    print(f'Using random seed {seed}')
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    root_path = os.path.join(args.logging_root, args.exp_name)
    
    
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(device)    
    num_features = 5
    if args.model == 'vgg':
        model = VGG_16(num_features, args.cat_feat)
    elif args.model == 'alexnet':
        model = Alex_Net(num_features, args.cat_feat)
    elif args.model == 'effnet':
        model = Efficient_Net(num_features, args.cat_feat)
    elif args.model == 'senet':
        model = SE_Net(num_features, args.cat_feat)
    else:
        raise NotImplementedError('Not implemented for name={args.name}')
        
    model.to(device) 
    total_n_params = utils.count_parameters(model)
    print(f'Total number of parameters of {args.model}: {total_n_params}')
    
    test_dataset = Breast_Dataset(split='test', data_aug=False)

    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    checkpoints_dir = os.path.join(root_path, 'checkpoints')
    PATH = os.path.join(checkpoints_dir, 'model_best_val.pth')
    train_state_dict = torch.load(PATH)

    model.load_state_dict(train_state_dict)
    model.to(device)
    test(model, test_data_loader, args)