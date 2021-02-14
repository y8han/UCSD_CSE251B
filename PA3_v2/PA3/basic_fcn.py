import torch.nn as nn

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)  # activation function: ReLu
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        #ConvTranspose2d -> small size to larger one (upsample)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        self.softmax = nn.Softmax()

    def forward(self, x):  #x -> input (one image or batch of images?)
        # Encoder
        x1 = self.conv1(x)
        x1 = self.bnd1(self.relu((x1)))
        x2 = self.conv2(x1)
        x2 = self.bnd2(self.relu((x2)))
        x3 = self.conv3(x2)
        x3 = self.bnd3(self.relu((x3)))
        x4 = self.conv4(x3)
        x4 = self.bnd4(self.relu((x4)))
        x5 = self.conv5(x4)
        x5 = self.bnd5(self.relu((x5)))
        # Decoder
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