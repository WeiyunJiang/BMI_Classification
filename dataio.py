import os
import h5py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor
import random
from models import VGG_16

class Breast_Dataset(Dataset):
    def __init__(self, resolution=(128, 128), downsample=True):
        self.downsample = downsample
        self.resolution = resolution
        with open("./data.csv", 'r') as f:
            self.filenames = [line.split()[0].split(',')[0] for line in f]
        with open("./data.csv", 'r') as f:
            self.labels = [line.split()[0].split(',')[1] for line in f]
        self.transform_image = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = Compose([
            ToTensor(),
            ])
    
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
        
        if self.downsample:
            image = PIL_image.resize(self.resolution)   
        
        image = np.asarray(image, dtype=np.uint8) 
        image = self.transform(image.copy())
        if label == 'MALIGNANT':
            label = float(1.0)
        else:
            label = float(0.0)
        
        sample = {'image': image, 'label': label}
        
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
    
    breast_dataset = Breast_Dataset()
    breast_dataset[0]
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(breast_dataset,
                                                                             [450, 65, 130])
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for i, batch in enumerate(train_data_loader):
        print(i)
        print(batch['image'].shape) # 64, 3, 128, 128
        print(batch['label'].shape) # 64, 1
        
        image_rescaled = rescale_img(batch['image'][0])
        image_rescaled = image_rescaled.numpy().transpose(1,2,0)
        plt.imshow(image_rescaled)
        
        image = batch['image']
        model = VGG_16() 
        out = model(image)
        print(out.shape) 
    pass
    
    
    