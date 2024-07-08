import numpy as np
import torch
import torch.nn as nn
from scf import *

class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()

        self.pool1 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels)
        self.depthconv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channels*2)

        self.conv2 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, stride=1, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels*2, in_channels*4, kernel_size=3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(in_channels*4)

        self.conv4 = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3, stride=1, padding=0)

        self.conv5 = nn.Conv2d(in_channels*4, in_channels*8, kernel_size=3, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(in_channels*8)

        self.conv6 = nn.Conv2d(in_channels*8, in_channels*8, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.Conv2d(in_channels*8, in_channels*8, kernel_size=2, stride=2, groups=in_channels*8)

        self.relu = nn.SiLU()

    def forward(self, x):

        x = self.pool1(x)        #256
        x = self.depthconv(x)
        x = self.pool1(x)
        x = self.depthconv(x)  #128

        x = self.conv1(x)  #64
        x = self.bn1(x)
        x = self.relu(x)

        for i in range(1):
            x = self.conv2(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.conv3(x) #32
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv5(x) #16
        x = self.bn3(x)
        x = self.relu(x)

        for i in range(1):
            x = self.conv6(x)
            x = self.bn3(x)
            x = self.relu(x)
        
        x = self.pool2(x) #8

        return x


class classifier(nn.Module):
    def __init__(self, in_channels, out_classes, dropout_prob = 0.2):
        super(classifier, self).__init__()

        self.out_classes = out_classes

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels, 16)
        self.fc2 = nn.Linear(16, out_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    

class Model(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_prob = 0.5):
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.cnn = CNN(in_channels)

        self.head = classifier(in_channels*8, self.num_classes, dropout_prob=dropout_prob)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.head(x)

        # if self.num_classes == 2:
        #     x = self.sigmoid(x)
        
        # else:
        #     x = self.softmax(x)

        return x