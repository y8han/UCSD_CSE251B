import torch
from dataloader_3 import n_class
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, random

def compute_aver(target_ious):
    if len(target_ious) > 0:
        return sum(target_ious) / len(target_ious)
    else:
        return -1

def iou_compu(pred, target):
    ious = []
    target_ious_0 = []
    target_ious_2 = []
    target_ious_9 = []
    target_ious_17 = []
    target_ious_25 = []
    pred_oneshot = torch.zeros(target.shape).to('cuda')
    pred_oneshot_ = (-1) * torch.ones(target.shape).to('cuda')
    tmp = pred.argmax(axis = 0)

    for c in range(n_class):
        pred_oneshot[c][c == tmp] = 1
        pred_oneshot_[c][c == tmp] = 1

    for cls in range(n_class - 1):
        tmp1 = target[cls,:,:]
        FP_TP_FN_TP = torch.sum(tmp1) + torch.sum(pred_oneshot[cls,:,:])
        FP_TP_FN_TP = int(FP_TP_FN_TP.item())
        TP = torch.eq(tmp1, pred_oneshot_[cls,:,:]) == True
        TP = TP[TP].shape[0]
        intersection = TP # intersection calculation
        union = FP_TP_FN_TP - TP #Union calculation
        if union == 0:
            # ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
            continue
        else:
            ious.append(intersection / union)
            # Append the calculated IoU to the list ious
            if cls == 0:
                target_ious_0.append(intersection / union)
            elif cls == 2:
                target_ious_2.append(intersection / union)
            elif cls == 9:
                target_ious_9.append(intersection / union)
            elif cls == 17:
                target_ious_17.append(intersection / union)
            elif cls == 25:
                target_ious_25.append(intersection / union)

    ious = sum(ious) / len(ious)
    target_ious_0 = compute_aver(target_ious_0)
    target_ious_2 = compute_aver(target_ious_2)
    target_ious_9 = compute_aver(target_ious_9)
    target_ious_17 = compute_aver(target_ious_17)
    target_ious_25 = compute_aver(target_ious_25)
    return ious, target_ious_0, target_ious_2, target_ious_9, target_ious_17, target_ious_25


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

def ploIoU(IoU_list, TargetIou_list_0, TargetIou_list_2, TargetIou_list_9, TargetIou_list_17, TargetIou_list_25, param = 'IoU', do_save_fig = False):
    fig = plt.figure()
    save_root = "./figures/" + param + "_curves.png"
    IoU = 'Validation Set ' + param
    Target_IoU = 'Validation Set Target ' + param
    plt.plot(IoU_list, 'b', label = IoU)
    plt.plot(TargetIou_list_0, 'r', label = Target_IoU)
    plt.plot(TargetIou_list_2, 'k', label=Target_IoU)
    plt.plot(TargetIou_list_9, 'g', label=Target_IoU)
    plt.plot(TargetIou_list_17, 'y', label=Target_IoU)
    plt.plot(TargetIou_list_25, 'c', label=Target_IoU)
    plt.xlabel('M epochs')
    plt.ylabel(param)
    plt.title('Validation ' + param + ' across training epochs')
    plt.grid('color')
    plt.legend(['IoU', 'road', 'sidewalk', 'car', 'billboard', 'sky'])
    if do_save_fig:
        plt.savefig(save_root)