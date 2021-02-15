import torch
from dataloader_3 import n_class
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, random

Tagret = [0 , 2, 9, 17, 25]

def iou_compu(pred, target):
    ious = []
    target_ious = []
    pred_oneshot = torch.zeros(target.shape).to('cuda')
    pred_oneshot_ = (-1) * torch.ones(target.shape).to('cuda')
    tmp = pred.argmax(axis = 0)

    for c in range(n_class):
        pred_oneshot[c][c == tmp] = 1
        pred_oneshot_[c][c == tmp] = 1

    for cls in range(n_class - 1):
        tmp = target[cls,:,:]
        FP_TP_FN_TP = torch.sum(tmp) + torch.sum(pred_oneshot[cls,:,:])
        FP_TP_FN_TP = int(FP_TP_FN_TP.item())
        TP = torch.eq(tmp, pred_oneshot_[cls,:,:]) == True
        TP = TP[TP].shape[0]
        intersection = TP # intersection calculation
        union = FP_TP_FN_TP - TP #Union calculation
        if union == 0:
            # ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
            continue
        else:
            ious.append(intersection / union)
            # Append the calculated IoU to the list ious
            if cls in Tagret:
                target_ious.append(intersection / union)
    ious = sum(ious) / len(ious)
    target_ious = sum(target_ious) / len(target_ious)
    return ious, target_ious


def pixel_acc(pred, target, use_gpu):
    if use_gpu:
        Remove_class = (n_class - 1) * torch.ones(target.shape).to('cuda')  #In order to remove class 26 when computing accuracy
    else:
        Remove_class = (n_class - 1) * torch.ones(target.shape)
    mask = torch.gt(Remove_class, target)  #remove the class 26
    pred = pred.argmax(axis = 1)
    total_pixel = list(pred[mask].shape)[0]
    accu = torch.eq(pred[mask], target[mask]) == True
    num_accu = list(accu[accu].shape)[0] / total_pixel  # pixel accuracy
    return num_accu

# plot the curve of Cost on Training & Validation set against epoch
def plotLoss(train_list, val_list, param = 'Loss', do_save_fig = False):
    fig = plt.figure()
    save_root = "./figures/" + param + "_curves.png"
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

def plotPixelaccracy(val_list, param='P_accu', do_save_fig = False):
    fig = plt.figure()
    save_root = "./figures/" + param + "_curves.png"
    ValLabel = 'Validation Set ' + param
    plt.plot(val_list, 'r', label=ValLabel)
    plt.xlabel('M epochs')
    plt.ylabel(param)
    plt.title('Validation ' + param + ' across training epochs')
    plt.grid('color')
    plt.legend(['Validation Set'])
    if do_save_fig:
        plt.savefig(save_root)

def ploIoU(IoU_list, Target_IoU_list, param = 'IoU', do_save_fig = False):
    fig = plt.figure()
    save_root = "./figures/" + param + "_curves.png"
    IoU = 'Validation Set ' + param
    Target_IoU = 'Validation Set Target ' + param
    plt.plot(IoU_list, 'b', label = IoU)
    plt.plot(Target_IoU_list, 'r', label = Target_IoU)
    plt.xlabel('M epochs')
    plt.ylabel(param)
    plt.title('Validation ' + param + ' across training epochs')
    plt.grid('color')
    plt.legend(['IoU', 'Target IoU'])
    if do_save_fig:
        plt.savefig(save_root)