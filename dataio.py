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

class Breast_Dataset(Dataset):
    def __init__(self, split='train', data_aug=False, resolution=(128, 128), downsample=False):
        self.downsample = downsample
        self.resolution = resolution
        # train split
        if split == 'train':
            dict_csv = get_dict_csv('./data_train.csv')
            self.filenames = list(dict_csv.keys())
            self.labels = [x[0] for x in dict_csv.values()]
            self.feature1 = [x[6] for x in dict_csv.values()] # Eccentricity
            self.feature2 = [x[10] for x in dict_csv.values()]# mean intensity
            # data augmentation
            if data_aug is True:
                self.transform_image = Compose([
                    RandomVerticalFlip(),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform_image = Compose([
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        # validation split
        elif split == 'val': 
            dict_csv = get_dict_csv('./data_valid.csv')
            self.filenames = list(dict_csv.keys())
            self.labels = [x[0] for x in dict_csv.values()]
            self.feature1 = [x[6] for x in dict_csv.values()] # Eccentricity
            self.feature2 = [x[10] for x in dict_csv.values()]# mean intensity
            self.transform_image = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif split == 'test': # test split
            dict_csv = get_dict_csv('./data_test.csv')
            self.filenames = list(dict_csv.keys())
            self.labels = [x[0] for x in dict_csv.values()]
            self.feature1 = [x[6] for x in dict_csv.values()] # Eccentricity
            self.feature2 = [x[10] for x in dict_csv.values()]# mean intensity
            
            self.transform_image = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise NotImplementedError('Not implemented for name={split}')
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[idx]
        with h5py.File("../" + filename, 'r') as f:
            data = f['data'][()]
            img = data[:, :, 0]
            scaled_img = rescale_img_np(img, tmax=255.0, tmin=0.0)
            PIL_image = Image.fromarray(np.uint8(scaled_img)).convert('RGB')    
        if self.downsample: # downsample the image to reduce storage
            PIL_image = PIL_image.resize(self.resolution)   
        image = self.transform_image(PIL_image.copy())
        if label == 'MALIGNANT': # define label
            label = float(1.0)
        else:
            label = float(0.0)
        feature1 = self.feature1[idx] # define feature
        feature2 = self.feature2[idx]
        feature_cat = np.array([feature1, feature2])
        sample = {'image': image, 'label': label, 'feature':feature_cat}
        return sample
    
def rescale_img_np(x, mode='scale', tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        
        xmax = np.max(x)
        xmin = np.min(x)
        
        if xmin == xmax:
            return 0.5 * np.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = np.clamp(x, 0, 1)
    return x

def rescale_img(x, mode='scale', tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        
        xmax = torch.max(x)
        xmin = torch.min(x)
        
        if xmin == xmax:
            return 0.5 * np.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x

if __name__ == '__main__':
    # filename = '../data/00001/LEFT_MLO.h5'
    # with h5py.File(filename, 'r') as f:
    #     data = f['data'][()]
    #     img = data[:, :, 0]
    #     scaled_img = rescale_img(img, tmax=255.0, tmin=0.0)
    #     PIL_image = Image.fromarray(np.uint8(scaled_img)).convert('RGB')
    seed = 40
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    train_dataset = Breast_Dataset(split='train', data_aug=True)
    val_dataset = Breast_Dataset(split='val', data_aug=False)
    test_dataset = Breast_Dataset(split='test', data_aug=False)
    # train_dataset[0]
    
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for i, batch in enumerate(test_data_loader):
        print(i)
        print(batch['image'].shape) # 64, 3, 256, 256
        print(batch['label'].shape) # 64, 1
        print(batch['feature'].shape) # 64, 2
        image_rescaled = rescale_img(batch['image'][0])
        image_rescaled = image_rescaled.numpy().transpose(1,2,0)
        plt.imshow(image_rescaled)
        
        image = batch['image']
        model = VGG_16() 
        out = model(image)
        print(out.shape) 
    pass
    
    
    