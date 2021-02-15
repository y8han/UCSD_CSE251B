#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
                                    )
        # No pooling 
        
        self.layer2 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
                                    )
        
        # No pooling
        
        self.layer3 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
                                    )
        # No pooling
        
        self.layer4 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
                                    )
        
        # No pooling
        
        self.layer5 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
                                    )
        
        # No pooling
        
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2) 
        
        # Standard 2x upsample
        
        self.layer6 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
                                    )
        
        self.deconv2 = nn.ConvTranspose2d(128, 64 , 2, stride=2)
        
        self.layer7 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
                                    )
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        
        self.layer8 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
                                    )
        
        self.deconv4 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        
        self.layer9 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    )
        self.layer10 = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        c1 = self.layer1(x) # of same size h, w, 16
        c2 = self.layer2(c1) # h//2, w//2, 32
        c3 = self.layer3(c2) # h//4, w//4, 64
        c4 = self.layer4(c3) # h//8, w//8, 128
        c5 = self.layer5(c4) # h//16, w//16, 256
        u6 = self.deconv1(c5) # h//8, w//8, 128
        c6 = self.layer6(torch.cat([u6, c4], dim=1)) # h//8, w//8,128
        u7 = self.deconv2(c6) # h//4, w//4, 64
        c7 = self.layer7(torch.cat([u7, c3], dim=1)) # h//4, w//4, 64
        u8 = self.deconv3(c7) # h//2, w//2, 32
        c8 = self.layer8(torch.cat([u8, c2], dim=1)) #h//2, w//2, 32
        u9 = self.deconv4(c8) #h, w, 16
        c9 = self.layer9(torch.cat([u9, c1], dim=1)) #h, w, 16
        final = self.layer10(c9)
        # verify the output shape
        return final

    


# In[ ]:




