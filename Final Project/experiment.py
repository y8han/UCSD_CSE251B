#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from file_utils import *
from tqdm import tqdm
from datetime import datetime
from dataset_factory import get_datasets
from model_factory import get_model


# In[ ]:


class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)
        self.__name = config_data['experiment_name']
        self.__data_path = config_data['dataset']['data_location']
        self.__experiment_dir = os.path.join('./experiment_data', self.__name)
        
        # Load Datasets
        self.__train_loader, self.__test_loader = get_datasets(config_data)
        
        # Setup Experiment
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        
        # Init Model
        self.__model = get_model(config_data)
        self.__load_experiment()
        
    def __load_experiment(self):
        os.makedirs('./experiment_data', exist_ok=True)
        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__model.__optimizer.load_state_dict(state_dict['optimizer'])
        else:
            os.makedirs(self.__experiment_dir)

    def run(self):
        start_epoch = self.__current_epoch
        for epoch in tqdm(range(start_epoch, self.__epochs)):  # loop over the dataset multiple times
            start_time = datetime.now()
            print(start_time)
            self.__current_epoch = epoch
            self.__model.idt = 0.5
            train_loss = self.__train()
            self.__record_stats(train_loss)
            if (epoch%20 == 0):
                self.__save_model(epoch)
                
        self.plot_stats()

    def __train(self):
        self.__model.train()
        training_loss = np.zeros(6)
        size = 0
        for i, data in enumerate(self.__train_loader):
            loss = self.__model.update(data)
            training_loss += loss
            size+=1
        return training_loss/size

    def test(self):
        self.__model.eval()
        test_loss = 0
        bleu1 = 0
        bleu4 = 0

        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                raise NotImplementedError()

        result_str = "Test Performance: Loss: {}, Perplexity: {}, Bleu1: {}, Bleu4: {}".format(test_loss,
                                                                                           bleu1,
                                                                                           bleu4)
        self.__log(result_str)

        return test_loss, bleu1, bleu4
    
    def __save_model(self, epoch):
        root_model_path = os.path.join(self.__experiment_dir, str(epoch) + '_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss):
        self.__training_losses.append(train_loss.tolist())
        self.plot_stats()
        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        [a,b,c,d,e,f] = plt.plot(x_axis, self.__training_losses)
        plt.legend([a,b,c,d,e,f], ['Discriminator A', 'Discriminator B', 'Generator A', 'Generator B', 
                                                       'Cycle A', 'Cycle B'], loc='best')
        plt.xlabel("Epochs")
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()


# In[2]:


if __name__ == "__main__":
    a = Experiment("parameter")


# In[ ]:


print(a)


# In[ ]:


a.run()


# In[ ]:




