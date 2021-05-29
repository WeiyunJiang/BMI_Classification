import os
import h5py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip
import random
from models import VGG_16, Alex_Net
from utils import get_dict_csv 
import SimpleITK as sitk
import imageio 
from data import get_file_names
import csv

def preprocess_mod(filenames, split):
    for filename in filenames:
        filename = './' + filename
        itkimage = sitk.ReadImage(filename)
        mri_scan_all = sitk.GetArrayFromImage(itkimage)
        for slice_idx in range(60, 100):
            mri_scan = mri_scan_all[slice_idx]
            mri_scan_uint8 = int16_to_uint8(mri_scan)
            mri_scan_uint8 = np.uint8(mri_scan_uint8)
            name = filename.split('/')[4] + '_slice_idx_' + str(slice_idx) + '_' + filename.split('/')[5] 
            pth = './modality/' + str(split) + '/' + name + '.png'
            imageio.imwrite(pth, mri_scan_uint8) 
        
def int16_to_uint8(img):
    max_new = 255
    min_new = 0
    img_new = (img - img.min()) * ((max_new - min_new) / (img.max() - img.min())) + min_new
    return img_new

if __name__ == '__main__':
    seed = 40
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # train_files = ['./archive/train/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_Flair.54512.mha',
    #               './archive/train/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_T1.54513.mha']
    test_files, train_files = get_file_names()
    with open('train_mod.csv', 'w') as f:
        for file in train_files:
            label = file.split('/')[-1].split('.')[-3].split('_')[-1]
            if label != '0T' and label != 'OT':
                f.write("%s %s\n"%(file, label))
    with open('test_mod.csv', 'w') as f:
        for file in test_files:
            label = file.split('/')[-1].split('.')[-3].split('_')[-1]
            f.write("%s %s\n"%(file, label))
    
    with open('./train_mod.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    # preprocess_mod(train_files, split='train')
    # preprocess_mod(test_files, split='test')