import matplotlib.pyplot as plt
import os, random
import numpy as np
from PIL import Image

# plot the curve of Cost on Training & Validation set against epoch
def plotFunc(train_list, val_list, ActivationFunction, param = 'Loss', lr = 0.005, gamma = 0.9, lambad = 0.00, n1 = 0, n2 = 0, do_save_fig = False): # param = 'Loss' or 'Accuracy'
    fig = plt.figure()
    save_root = "./figures/" + ActivationFunction + "_" + param + "_" + str(lr) + "_" + str(gamma) + "_" + str(lambad) + "_" + str(n1) + "_" + str(n2) + "_curves.png"
    TrainLabel = 'Training Set ' + param
    ValLabel = 'Validation Set ' + param
    plt.plot(train_list, 'b', label=TrainLabel)
    plt.plot(val_list, 'r', label=ValLabel)
    plt.xlabel('M epochs')
    plt.ylabel(param)
    plt.title('Training and Validation ' + param + ' across training epochs')
    plt.grid('color')
    plt.legend(['Training Set', 'Validation Set'])
    if do_save_fig:
        plt.savefig(save_root)
