#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchvision import models
import torch.utils.data as data
import itertools
#from constants import *
from file_utils import read_file_in_dir
#from generator import define_G
#from discriminator import define_D
import networks
import random


# In[2]:


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


# In[3]:



class CycleGAN(nn.Module):
    
    def __init__(self, config_data):
        super().__init__()
        self.lr = config_data['cycleGAN']['lr']
        self.criterionGAN = nn.MSELoss()
        self.criterionCycle = torch.nn.L1Loss()
        if torch.cuda.is_available():
            self.criterionGAN = self.criterionGAN.to("cuda")
            self.criterionCycle = self.criterionCycle.to("cuda")
        
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.lambda__ = 10
        
        
        self.model_G_A = networks.define_G(3, 3, 64, 'unet_128', norm='instance',gpu_ids=[0])
        #define_G(3, 3, 64, 'unet_128', 'instance', use_dropout=True)
        self.model_G_B =networks.define_G(3, 3, 64, 'unet_128', norm='instance', gpu_ids=[0])
        
        self.model_D_A = networks.define_D(3, 64, 'basic', 3, gpu_ids=[0])
        self.model_D_B = networks.define_D(3, 64, 'basic', 3, gpu_ids=[0])
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)
        
        # changes needed
        #torch.nn.init.xavier_uniform(self.model_G_A.weight.data)
        #torch.nn.init.xavier_uniform(self.model_G_B.weight)
        #torch.nn.init.xavier_uniform(self.model_D_A.weight)
        #torch.nn.init.xavier_uniform(self.model_D_B.weight)
        self.optimizerG = torch.optim.Adam(itertools.chain(self.model_G_A.parameters(), self.model_G_B.parameters()), lr=self.lr, betas = (0.5, 0.99))
        self.optimizerD = torch.optim.Adam(itertools.chain(self.model_D_A.parameters(), self.model_D_B.parameters()), lr=self.lr, betas = (0.5, 0.99))
        self.optimizer = []
        self.optimizer.append(self.optimizerG)
        self.optimizer.append(self.optimizerD)
    
    def forward(self, input_image):
        '''G_A is generating A from B and G_B is generating B from A'''
        self.real_A = input_image['A']
        self.real_B = input_image['B']
        
        if torch.cuda.is_available():
            self.real_A = self.real_A.to("cuda")
            self.real_B = self.real_B.to("cuda")
        
        #G_A(B)
        self.fake_A = self.model_G_A(self.real_B)
        
        #G_B(A)
        self.fake_B = self.model_G_B(self.real_A)
        
        #G_A(G_B(A))
        self.recreate_A = self.model_G_A(self.fake_B)
        
        #G_B(G_A(B))
        self.recreate_B = self.model_G_B(self.fake_A)
    def basic_D_backward(self, model_D, real, fake):
        pred_real = model_D(real)
        loss_D_real = self.criterionGAN(pred_real, self.real_label.expand_as(pred_real).to('cuda'))
        pred_fake = model_D(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, self.fake_label.expand_as(pred_fake).to('cuda'))
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    def D_A_backward(self):
        '''D_A is discriminate A'''
        fake_A_POOL = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.basic_D_backward(self.model_D_A, self.real_A, fake_A_POOL)
    def D_B_backward(self):
        fake_B_POOL = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.basic_D_backward(self.model_D_B, self.real_B, fake_B_POOL)
    def backward_G(self):
        check_G_A = self.model_D_A(self.fake_A)
        
        self.loss_G_A = self.criterionGAN(check_G_A,self.real_label.expand_as(check_G_A).to('cuda'))
        check_G_B = self.model_D_B(self.fake_B)
        self.loss_G_B = self.criterionGAN(check_G_B,self.real_label.expand_as(check_G_B).to('cuda'))
        self.loss_cycle_A = self.criterionCycle(self.recreate_A, self.real_A) * self.lambda__
        self.loss_cycle_B = self.criterionCycle(self.recreate_B, self.real_B) * self.lambda__
        self.loss_G = self.loss_cycle_A + self.loss_cycle_B + self.loss_G_A + self.loss_G_B
        self.loss_G.backward()
    def update(self, input_image):
        self.forward(input_image)
        self.set_model_grad([self.model_D_A, self.model_D_B], False)
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()
        self.set_model_grad([self.model_D_A, self.model_D_B], True)
        self.optimizerD.zero_grad()
        self.D_A_backward()
        self.D_B_backward()
        self.optimizerD.step()
        return self.loss_D_A, self.loss_D_B, self.loss_G, self.loss_G_A, self.loss_G_B, self.loss_cycle_A, self.loss_cycle_B
        
    def set_model_grad(self, nets, requires):
        for net in nets:
            if net is not None:
                for para in net.parameters():
                    para.require_grad = requires
        
        
def get_model(config_data):
    return CycleGAN(config_data)


# In[4]:


#config_data = read_file_in_dir('./','parameter.json')


# In[5]:


#a = get_model(config_data)


# In[4]:


#print(a)


# In[ ]:




