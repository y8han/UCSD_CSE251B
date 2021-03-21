#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchvision import models
import torch.utils.data as data
import itertools
from file_utils import read_file_in_dir
from generator import define_G
from discriminator import define_D
import random
import numpy as np

# In[2]:
# This Image Pool helper method is copied directly fom the origial CycleGAN paper
# ImagePool is not a critical method therefore we choose to use it directly to reduce the possibility of bug
# and to focus on the main part of the project

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
        self.criterionIdt = nn.L1Loss()
        if torch.cuda.is_available():
            self.criterionGAN = self.criterionGAN.to("cuda")
            self.criterionCycle = self.criterionCycle.to("cuda")
            # used to make sure the color is on the same domain.
            self.criterionIdt = self.criterionIdt.to("cuda")

        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.lambda__ = 10

        self.idt = 0.2
        self.model_G_A = define_G(3, 3, 64, 'unet_128', norm='instance')
        #define_G(3, 3, 64, 'unet_128', 'instance', use_dropout=True)
        self.model_G_B =define_G(3, 3, 64, 'unet_128', norm='instance')

        self.model_D_A = define_D(3, 64, 3)
        self.model_D_B = define_D(3, 64, 3)
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        # changes needed
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

        self.idt_A = self.model_G_A(self.real_B)
        self.loss_identity_A = self.criterionIdt(self.idt_A, self.real_B) * self.idt
        self.idt_B = self.model_G_B(self.real_A)
        self.loss_identity_B = self.criterionIdt(self.idt_B, self.real_A) * self.idt

        self.loss_G_B = self.criterionGAN(check_G_B,self.real_label.expand_as(check_G_B).to('cuda'))
        self.loss_cycle_A = self.criterionCycle(self.recreate_A, self.real_A) * self.lambda__
        self.loss_cycle_B = self.criterionCycle(self.recreate_B, self.real_B) * self.lambda__
        self.loss_G = self.loss_cycle_A + self.loss_cycle_B + self.loss_G_A + self.loss_G_B + self.loss_identity_A + self.loss_identity_B
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
        return np.array([self.loss_D_A.data.cpu().float(), self.loss_D_B.data.cpu().float(), self.loss_G_A.data.cpu().float(),
                         self.loss_G_B.data.cpu().float(), self.loss_cycle_A.data.cpu().float(), self.loss_cycle_B.data.cpu().float()])

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
