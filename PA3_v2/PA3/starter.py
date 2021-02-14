from torchvision import utils
from basic_fcn import *
from dataloader_3 import *
# from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time


# TODO: Some missing values are represented by '__'. You need to fill these up.
Batch_size = 4
train_dataset = IddDataset(csv_file='train.csv')
batch_train = DataLoader(train_dataset, batch_size = Batch_size, num_workers = 2)
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')


# train_loader = DataLoader(dataset=train_dataset, batch_size= __, num_workers= __, shuffle=True)
# val_loader = DataLoader(dataset=val_dataset, batch_size= __, num_workers= __, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size= __, num_workers= __, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # Kernal parameters are learnable
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

epochs = 50   # Trainging Epoch
criterion = torch.nn.CrossEntropyLoss(ignore_index = n_class)  # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
# TODO: ignore index out of boundry (0-26, but 27,28 may appear)

fcn_model = FCN(n_class = n_class)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr=0.9)

use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()

        
def train():
    fcn_model.train()
    for epoch in range(epochs):
        ts = time.time()
        # For test (one image)
        # for i in range(100):
        #     optimizer.zero_grad()
        #     tmp_a = train_dataset[i][0].shape[0]
        #     tmp_b = train_dataset[i][0].shape[1]
        #     tmp_c = train_dataset[i][0].shape[2]
        #     X = train_dataset[i][0].reshape(1, tmp_a, tmp_b, tmp_c)
        #     X = X
        #     outputs = fcn_model.forward(X)
        #     print(outputs.shape)
        #     test = outputs[0]
        #     print(train_dataset[i][2].shape)
        #     print(test.shape)
        #     print(test.reshape(1024,n_class,1920).shape)
        #     print(torch.max(train_dataset[i][2]))
        #     print(torch.min(train_dataset[i][2]))
        #     loss = criterion(test.reshape(1024,n_class,1920), train_dataset[i][2])
        #     print(loss)
        #     loss.backward()
        #     optimizer.step()
        # For test (one image)
        for iter, (X, Y) in enumerate(batch_train):
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
            # print(inputs.shape)
            outputs = fcn_model.forward(inputs)
            # Output size: N,n_class.H,W
            # print(outputs.shape)
            # print(labels.shape)
            # print(labels[0])
            # print(torch.max(labels[0]))
            # print(torch.max(labels[1]))
            # print(torch.max(labels[2]))
            # print(torch.max(labels[3]))
            loss = criterion(outputs, labels)
            # print(loss)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, 'best_model')

        val(epoch)
        fcn_model.train()
    


def val(epoch):
    fcn_model.eval() # Don't forget to put in eval mode !
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    
def test():
	fcn_model.eval()
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
if __name__ == "__main__":
    # val(0)  # show the accuracy before training
    train()