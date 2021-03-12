#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv, os
from torch.utils.data import DataLoader
from cycle_dataset import CycleDataset

def get_datasets(config_data):
    train_dataloader = DataLoader(CycleDataset(config_data, 'train'), batch_size=config_data['experiment']['batch_size'], num_workers=config_data['experiment']['num_worker'])
    test_dataloader = DataLoader(CycleDataset(config_data, 'test'), batch_size=config_data['experiment']['batch_size'], num_workers=config_data['experiment']['num_worker'])
    return train_dataloader, test_dataloader


# In[ ]:




