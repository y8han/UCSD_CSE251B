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
from generator import define_G
from discriminator import define_D

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
        
        
        self.model_G_A = define_G(3, 3, 64, 'unet_128', 'instance', use_dropout=True)
        self.model_G_B = define_G(3, 3, 64, 'unet_128', 'instance', use_dropout=True)
    
        self.model_D_A = define_D(3, 64, n_layers_D=3, use_sigmoid=True)
        self.model_D_B = define_D(3, 64, n_layers_D=3, use_sigmoid=True)
        
        # changes needed
        #torch.nn.init.xavier_uniform(self.model_G_A.weight.data)
        #torch.nn.init.xavier_uniform(self.model_G_B.weight)
        #torch.nn.init.xavier_uniform(self.model_D_A.weight)
        #torch.nn.init.xavier_uniform(self.model_D_B.weight)
        self.optimizerG = torch.optim.Adam(itertools.chain(self.model_G_A.parameters(), self.model_G_B.parameters()), lr=self.lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.model_D_A.parameters(), self.model_D_B.parameters()), lr=self.lr)
        
    
    def forward(self, input_image):
        '''G_A is generating A from B and G_B is generating B from A'''
        self.real_A = input_image['A']
        self.real_B = input_image['B']
        
        if torch.cuda.is_available():
            self.real_A = self.real_A.to("cuda")
            self.real_A = self.real_B.to("cuda")
        
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
        loss_D_real = self.criterionGAN(pred_real, self.real_label.expand_as(pred_real))
        pred_fake = model_D(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, self.fake_label.expand_as(pred_fake))
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    def D_A_backward(self, real, fake):
        '''D_A is discriminate A'''
        self.loss_D_A = self.basic_D_backward(self.model_D_A, self.real_A, self.fake_A)
    def D_B_backward(self, real, fake):
        self.loss_D_B = self.basic_D_backward(self.model_D_B, self.real_B, self.fake_B)
    def backward_G(self):
        check_G_A = self.model_D_A(self.fake_A)
        self.loss_G_A = self.criterionGAN(check_G_A,self.real_label.expand_as(check_G_A))
        check_G_B = self.model_D_B(self.fake_B)
        self.loss_G_B = self.criterionGAN(check_G_B,self.real_label.expand_as(check_G_B))
        self.loss_cycle_A = self.criterionCycle(self.recreate_A, self.real_A) * 0.5
        self.loss_cycle_B = self.criterionCycle(self.recreate_B, self.real_B) * 0.5
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()
    def update(self, input_image):
        self.forward(input_image)
        self.set_model_grad([self.model_D_A, sel.model_D_B], False)
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()
        self.set_model_grad([self.model_D_A, sel.model_D_B], True)
        self.optimizerD.zero_grad()
        self.D_A_backward()
        self.D_B_backward()
        self.optimizerD.step()
        return self.loss_D_A, self.loss_D_B, self.loss_G
        
    def set_model_grad(self, nets, requires):
        for net in nets:
            if net is not None:
                for para in net.parameters():
                    para.require_grad = requires
        
        
def get_model(config_data):
    return CycleGAN(config_data)


# In[2]:


#config_data = read_file_in_dir('./','parameter.json')


# In[3]:


#a = get_model(config_data)


# In[4]:


#print(a)


# In[ ]:




