import torch
from dataloader_3 import n_class
from sklearn.metrics import confusion_matrix
import numpy as np

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