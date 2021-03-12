#!/usr/bin/env python
# coding: utf-8

# In[49]:


import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import random
import numpy as np
from PIL import Image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


class CycleDataset(data.Dataset):
    
    def is_image_file(self, fname):
        return any(fname.endswith(extension) for extension in IMG_EXTENSIONS)
    def generate_index(self, path):
        images = []
        assert os.path.isdir(path)
        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images
    
    def __init__(self, config, mode):
        self.dir_A = os.path.join(config['dataset']['data_location'], mode + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(config['dataset']['data_location'], mode + 'B')  # create a path '/path/to/data/trainB'
        
        self.A_paths = sorted(self.generate_index(self.dir_A))
        self.B_paths = sorted(self.generate_index(self.dir_B))
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[random.randint(0, self.B_size -1)]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        return {'A': A_img, 'B': B_img, 'A_path': A_path, 'B_path': B_path}
    def __len__(self):
        return max(self.A_size, self.B_size)


# In[50]:


if __name__ == "__main__" :
    data = CycleDataset(config, "train")
    print(data[0])


# In[ ]:





# In[ ]:




