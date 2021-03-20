#!/usr/bin/env python
# coding: utf-8

# In[20]:


import torch
import torch.nn as nn
import numpy as np


# In[21]:


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1 or classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[25]:


def define_D(input_nc, ndf,
             n_layers_D=3, use_sigmoid=True):
    # make sure the number of layers should be 3
    use_gpu = torch.cuda.is_available()
    netD = NLayerDiscriminator(input_nc, ndf, n_layers_D)
    if use_gpu:
        netD = netD.to('cuda')
    netD.apply(weights_init)
    return netD

# In[26]:


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()

        kw = 4  # fixed to be 4 -> 70 * 70 patchGan
        # require the following parameters (do not change)
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        #         nf_mult = 1
        #         nf_mult_prev = 1

        for i in range(n_layers):
            sequence += [
                nn.Conv2d(ndf * (2 ** i), ndf * (2 ** (i + 1)), kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm2d(ndf * (2 ** (i + 1))),
                nn.LeakyReLU(0.2, True)
            ]
        # In order to specify the PatchGan to be 70 * 70, we need to have five layers. (more or less layers are wrong)
        #         for n in range(1, n_layers):
        #             nf_mult_prev = nf_mult
        #             nf_mult = min(2**n, 8)
        #             sequence += [
        #                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
        #                                 kernel_size=kw, stride=2, padding=padw),
        #                 # TODO: use InstanceNorm
        #                 nn.BatchNorm2d(ndf * nf_mult),
        #                 nn.LeakyReLU(0.2, True)
        #             ]
        sequence += [
            nn.Conv2d(ndf * (2 ** (n_layers)), ndf * (2 ** (n_layers)),
                      kernel_size=kw, stride=2, padding=padw),
            # TODO: use InstanceNorm
            nn.BatchNorm2d(ndf * (2 ** (n_layers))),
            nn.LeakyReLU(0.2, True)
        ]
        #         nf_mult_prev = nf_mult
        #         nf_mult = min(2**n_layers, 8)
        #         sequence += [
        #             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
        #                             kernel_size=kw, stride=1, padding=padw),
        #             # TODO: useInstanceNorm
        #             nn.BatchNorm2d(ndf * nf_mult),
        #             nn.LeakyReLU(0.2, True)
        #         ]

        sequence += [nn.Conv2d(ndf * (2 ** (n_layers)), 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# In[ ]:




