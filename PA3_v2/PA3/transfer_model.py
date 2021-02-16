from torchvision import models
import torch.nn as nn


# def transfer_learning(n_class):
#     model = models.vgg16(pretrained=True) # Use the VGG-16.
#     # Although it didn’t record the lowest error, it is found to work
#     # well for the task and was quicker to train than other models.
#
#     # Freeze model weights
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # Parameters of newly constructed modules have requires_grad=True by default
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, n_class) # the size of each output sample = n_class = 27
#
#
#     # Move to GPU
#     model = model.to('cuda')
#     # Distribute across 2 GPUs
#     model = nn.DataParallel(model)
#
#     criterion = nn.CrossEntropyLoss(ignore_index = n_class)
#     optimizer = optim.Adam()

class TransferModel(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)  # activation function: ReLu
        self.model = models.resnet18(pretrained=True)  # Use the VGG-16.
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        for param in self.model.parameters():
            param.requires_grad = False
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        self.softmax = nn.Softmax()

    def forward(self, x):  # x -> input (one image or batch of images?)
        # Encoder
        x5 = self.model(x)
        x6 = self.deconv1(x5)
        x6 = self.bn1(self.relu(x6))
        x7 = self.deconv2(x6)
        x7 = self.bn2(self.relu(x7))
        x8 = self.deconv3(x7)
        x8 = self.bn3(self.relu(x8))
        x9 = self.deconv4(x8)
        x9 = self.bn4(self.relu(x9))
        x10 = self.deconv5(x9)
        x10 = self.bn5(self.relu(x10))

        score = self.classifier(x10)
        score = self.softmax(score)

        return score  # size=(N, self.n_class, x.H/1, x.W/1)   N -> batch size