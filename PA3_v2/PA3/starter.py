from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import os

# TODO: Some missing values are represented by '__'. You need to fill these up.
Batch_size = 16
Width = 256
Height = 480
train_dataset = IddDataset(csv_file='train.csv', w = Width, h = Height)
batch_train = DataLoader(train_dataset, batch_size = Batch_size, num_workers = 4, shuffle = True)
val_dataset = IddDataset(csv_file='val.csv', w = Width, h = Height)
batch_val = DataLoader(val_dataset, batch_size = Batch_size, num_workers = 4, shuffle = True)
test_dataset = IddDataset(csv_file='test.csv', w = Width, h =Height)
batch_test = DataLoader(test_dataset, batch_size = Batch_size, num_workers = 4, shuffle = True)

# train_loader = DataLoader(dataset=train_dataset, batch_size= __, num_workers= __, shuffle=True)
# val_loader = DataLoader(dataset=val_dataset, batch_size= __, num_workers= __, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size= __, num_workers= __, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # Kernal parameters are learnable
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

def judge(alist):
	if any(alist[i+1] <= alist[i] for i in range(0,len(alist)-1)):
		return False
	else:
		return True

epochs = 40   # Trainging Epoch
earlyStop_thres = 3  #error on validation continueu to go up for earlyStop_thres epoches (early stop)
criterion = torch.nn.CrossEntropyLoss(ignore_index = n_class)  # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
# TODO: ignore index out of boundry (0-26, but 27,28 may appear)
# TODO: Update Weight of each class

fcn_model = FCN(n_class = n_class)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr = 0.01)

use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()

        
def train(init_accu, use_gpu, InitioU, Init_tagretioU):
    Accuracy_list = []
    Accuracy_list.append(init_accu)
    Iou_list = []
    Iou_list.append(InitioU)
    TargetIou_list = []
    TargetIou_list.append(Init_tagretioU)
    ValLoss = []
    TrainLoss = []
    fcn_model.train()
    Early_stop = []
    for epoch in range(epochs):
        ts = time.time()
        average_loss = torch.zeros(1)
        index = 0
        if use_gpu:
            average_loss = average_loss.to('cuda')
        for iter, (X, tar, Y) in enumerate(batch_train):
            # each iter bag contains a batch size of images and labels
            # X is the input  size: N,H,W
            # tar is the target size: N,n_class,H,W
            # Y is label (compute cross entropy loss)
            optimizer.zero_grad()
            if use_gpu:
                inputs = X.to('cuda') # Move your inputs onto the gpu
                labels = Y.to('cuda') # Move your labels onto the gpu
            else:
                inputs, labels = X, Y # Unpack variables into inputs and labels
            outputs = fcn_model.forward(inputs)
            loss = criterion(outputs, labels)
            average_loss += loss
            index += 1
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print("Train Set: epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        TrainLoss.append(float(average_loss.cpu()) / index)
        print("Training: Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        accu, val_loss, Iou, targetIou = val(epoch + 1, use_gpu)
        Accuracy_list.append(accu)
        ValLoss.append(val_loss)
        Iou_list.append(Iou)
        TargetIou_list.append(targetIou)

        if len(Early_stop) < earlyStop_thres:  #The first earlyStop_thres steps
            Early_stop.append(val_loss)
        else: #compare the current valid loss with Early_stop[earlyStop_thres - 1] and Early_stop[earlyStop_thres - 1] with Early_stop[earlyStop_thres - 2], .... etc
            Early_stop.append(val_loss)
            flag = judge(Early_stop)
            if flag:
                print("Early stop!")
                break
            else:
                Early_stop.pop(0)

        if val_loss <= min(ValLoss):
            print("New model is saved!")
            torch.save(fcn_model, 'best_model')

        print("Train Loss_list:", TrainLoss)
        print("Valid Accuracy_list:", Accuracy_list)
        print("Valid Loss_list:", ValLoss)
        print("Valid IoU_list:", Iou_list)
        print("Valid TargetIoU_list:", TargetIou_list)
        fcn_model.train()

    #visualize & save figures
    plotLoss(TrainLoss, ValLoss, param = "Loss", do_save_fig = True)
    plotPixelaccracy(Accuracy_list, param = "P_accu", do_save_fig = True)
    ploIoU(Iou_list, TargetIou_list, param = "IoU", do_save_fig = True)


def val(epoch, use_gpu):
    fcn_model.eval() # Don't forget to put in eval mode !
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    ts = time.time()
    Accuracy = []
    Loss = []
    IoU = []
    TargetIoU = []
    # print("Compute Accuracy: epoch {}".format(epoch))
    for iter, (X, tar, Y) in enumerate(batch_val):
        if use_gpu:
            inputs = X.to('cuda')  # Move your inputs onto the gpu
            labels = Y.to('cuda')  # Move your labels onto the gpu
        else:
            inputs, labels = X, Y  # Unpack variables into inputs and labels
        outputs = fcn_model.forward(inputs)
        loss = criterion(outputs, labels)
        num_accu = pixel_acc(outputs, labels, use_gpu)
        aver_iou = []
        aver_target_iou = []
        for t_ in range(outputs.shape[0]):
            if use_gpu:
                iou, target_ious = iou_compu(outputs[t_], tar.to('cuda')[t_])
                aver_iou.append(iou)
                aver_target_iou.append(target_ious)
            else:
                iou, target_ious = iou_compu(outputs[t_], tar[t_])
                aver_iou.append(iou)
                aver_target_iou.append(target_ious)
        aver_iou = sum(aver_iou) / len(aver_iou)
        aver_target_iou = sum(aver_target_iou) / len(aver_target_iou)
        Loss.append(float(loss.cpu().detach()))
        Accuracy.append(num_accu)
        IoU.append(aver_iou)
        TargetIoU.append(aver_target_iou)
        if iter % 10 == 0:
            print("epoch{}, iter{}, accuracy".format(epoch, iter))
    Aver_accu = sum(Accuracy) / len(Accuracy)
    Aver_loss = sum(Loss) / len(Loss)
    IoU = sum(IoU) / len(IoU)
    TargetIoU = sum(TargetIoU) / len(TargetIoU)
    print("Validation: Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
    # print("Validation Set: Pixel accuracy(Loss) at epoch {} is {}({})".format(epoch, Aver_accu, Aver_loss))
    return Aver_accu, Aver_loss, IoU, TargetIoU

# No need to plot the curves on test dataset? (not mentioned in pdf)
# def test():
# 	fcn_model.eval()
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
if __name__ == "__main__":
    figure_save = './figures/'
    if not os.path.exists(figure_save):
        os.makedirs(figure_save)
    accu, loss, Iou, targetIou = val(0, use_gpu)  # show the accuracy before training
    train(accu, use_gpu, Iou, targetIou)
