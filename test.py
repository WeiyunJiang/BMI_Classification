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
from dataio import Breast_Dataset, Modality_Dataset
from args import breast_arg
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, classification_report

import matplotlib.pyplot as plt

def test(model, test_data_loader, args):
    '''

    Parameters
    ----------
    model : str
        name of the model {vgg, effnet, senet}.
    test_data_loader : dataloader
        dataloader for testing dataset.
    args : arguments
        misc arguments.

    Returns
    -------
    Prints the TP, TN, FP, FN and save the roc curve

    '''
    with torch.no_grad():
        total_acc = []
        total_pred = []
        total_raw_pred = []
        total_label = []
        model.eval()
        for step, batch in tqdm(enumerate(test_data_loader)):  
            image, label = batch['image'], batch['label']
            image = image.to(device)
            label = label.to(device)
            
            pred = model(image.float())
            pred = pred.squeeze(-1)
            label = label.type(torch.LongTensor)
            
            label = label.to(device)
            total_raw_pred.append(pred.clone().detach().cpu().numpy())
            # pred[pred > 0.5] = 1
            # pred[pred <= 0.5] = 0
            total_pred.append(pred.clone().detach().cpu().numpy())
            total_label.append(label.clone().detach().cpu().numpy())
            correct_results_sum = (torch.argmax(pred, dim=-1) == label).sum().float()
        
            acc = correct_results_sum/pred.shape[0]
            total_acc.append(acc.clone().detach().cpu().numpy())

        # compute the roc curve
        fpr, tpr, thresholds = roc_curve(total_label, total_raw_pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot()  
        plt.show()
        plt.savefig(args.exp_name +  "roc.png") 
        tn, fp, fn, tp = confusion_matrix(total_label, total_pred).ravel()
        # compute the accuracy
        mean_acc = np.mean(total_acc)
        print(classification_report(total_label, total_pred))
        tqdm.write(f"acc: {mean_acc}, tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}" )
        
     
        
        
        
        
        
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
    if args.cat_feat: num_features = 2
    else: num_features = 0
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
    
    test_dataset = Modality_Dataset(split='test', data_aug=False)

    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    checkpoints_dir = os.path.join(root_path, 'checkpoints')
    PATH = os.path.join(checkpoints_dir, 'model_best_val.pth')
    train_state_dict = torch.load(PATH)

    model.load_state_dict(train_state_dict)
    model.to(device)
    test(model, test_data_loader, args)
