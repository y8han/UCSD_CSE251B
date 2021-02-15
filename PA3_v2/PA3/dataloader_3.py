from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import pandas as pd
from collections import namedtuple
import random

n_class    = 27

# a label and all meta information
Label = namedtuple( 'Label' , [
    'name'        , 
    'level3Id'    , 
    'color'       , 
    ] )

labels = [
    #       name                     level3Id  color
    Label(  'road'                 ,    0  , (128, 64,128)  ),
    Label(  'drivable fallback'    ,    1  , ( 81,  0, 81)  ),
    Label(  'sidewalk'             ,    2  , (244, 35,232)  ),
    Label(  'non-drivable fallback',    3  , (152,251,152)  ),
    Label(  'person/animal'        ,    4  , (220, 20, 60)  ),
    Label(  'rider'                ,    5  , (255,  0,  0)  ),
    Label(  'motorcycle'           ,    6  , (  0,  0,230)  ),
    Label(  'bicycle'              ,   7  , (119, 11, 32)  ),
    Label(  'autorickshaw'         ,   8  , (255, 204, 54) ),
    Label(  'car'                  ,   9  , (  0,  0,142)  ),
    Label(  'truck'                ,  10 ,  (  0,  0, 70)  ),
    Label(  'bus'                  ,  11 ,  (  0, 60,100)  ),
    Label(  'vehicle fallback'     ,  12 ,  (136, 143, 153)),  
    Label(  'curb'                 ,   13 ,  (220, 190, 40)),
    Label(  'wall'                 ,  14 ,  (102,102,156)  ),
    Label(  'fence'                ,  15 ,  (190,153,153)  ),
    Label(  'guard rail'           ,  16 ,  (180,165,180)  ),
    Label(  'billboard'            ,   17 ,  (174, 64, 67) ),
    Label(  'traffic sign'         ,  18 ,  (220,220,  0)  ),
    Label(  'traffic light'        ,  19 ,  (250,170, 30)  ),
    Label(  'pole'                 ,  20 ,  (153,153,153)  ),
    Label(  'obs-str-bar-fallback' , 21 ,  (169, 187, 214) ),  
    Label(  'building'             ,  22 ,  ( 70, 70, 70)  ),
    Label(  'bridge/tunnel'        ,  23 ,  (150,100,100)  ),
    Label(  'vegetation'           ,  24 ,  (107,142, 35)  ),
    Label(  'sky'                  ,  25 ,  ( 70,130,180)  ),
    Label(  'unlabeled'            ,  26 ,  (  0,  0,  0)  ),
]

class Resize(object):
    def __init__(self, output_size): # the output size should be a tuple of exactly (1080, 1920)
        self.output_size = output_size
        self.imageResize = transforms.Resize(self.output_size, Image.BILINEAR)
        self.labelResize = transforms.Resize(self.output_size, Image.NEAREST)
    def __call__(self, sample):
        image, label = sample
        img = self.imageResize(image)
        lab = self.labelResize(label)
        return img, lab
class MirrorFlip(object):
    def __init__(self, probability = 0.5):
        self.hFlip = transforms.functional.hflip
        self.prob = probability
    def __call__(self, sample):
        image, label = sample
        if random.random() > self.prob:
            img = self.hFlip(image)
            lab = self.hFlip(label)
        else:
            img = image
            lab = label
        return img, lab
class Rotate(object):
    def __init__(self, degree = 10):
        self.degree = degree
        self.rotation = transforms.functional.rotate
    def __call__(self, sample):
        image, label = sample
        imageDegree = self.degree * random.random() - self.degree/2
        img = self.rotation(image, angle=imageDegree, resample=Image.BILINEAR)
        lab = self.rotation(label, angle=imageDegree, resample=Image.NEAREST)
        return img, lab

class RondomCrop(object):
    def __init__(self, shape):
        self.shape = shape
    def __call__(self, sample):
        image, label = sample
        a, b, c, d = transforms.RandomCrop.get_params(image, output_size=self.shape)
        img = transforms.functional.crop(image, a, b, c, d)
        lab = transforms.functional.crop(label, a, b, c, d)
        return img,lab
    
class Blur(object):
    def __init__(self, kernel_size = 3, sigma=(0.1, 1)):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.blurr = transforms.GaussianBlur(self.kernel_size, self.sigma)
    def __call__(self, sample):
        image, lab = sample
        self.blurr(image)
        return image, lab
    
class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()
    def __call__(self, sample):
        image, label = sample
        image = self.transform(image)
        label = torch.from_numpy(np.asarray(label).copy()).long()
        return image, label

class CenterCrop(object):
    def __init__(self):
        pass
    def __call__(self):
        pass
class Normalize(object):
    def __init__(self):
        self.transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    def __call__(self, sample):
        image, label = sample
        image = self.transform(image)
        return image, label

class IddDataset(Dataset):
    def __init__(self, csv_file, n_class=n_class, transforms_=None, w = 0, h = 0):
        self.data      = pd.read_csv(csv_file)
        self.n_class   = n_class
        self.mode = csv_file
        
        # Add any transformations here
        
        # The following transformation normalizes each channel using the mean and std provided
        self.transforms = transforms.Compose([
                                            Resize((w, h)),
                                              #RondomCrop((1024, 1320)),
                                              #Blur(),
                                              #Rotate(),
                                              #MirrorFlip(),
                                              ToTensor(),
                                              #Normalize(),
                                             ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_name = self.data.iloc[idx, 0]
        img = Image.open(img_name).convert('RGB')
        label_name = self.data.iloc[idx, 1]
        label = Image.open(label_name)

        img, label = self.transforms((img, label)) # Normalization
        #label = torch.from_numpy(label.copy()).long() # convert to tensor

        # create one-hot encoding
        label = torch.squeeze(label)
        h, w = label.shape
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
           target[c][c==label] = 1
        
        return img, target, label




