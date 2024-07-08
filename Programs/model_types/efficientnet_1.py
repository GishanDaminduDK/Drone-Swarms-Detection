import torch
import torch.nn as nn
from torchvision import models

class Model(nn.Module):

    def __init__(self, in_channels, model_freeze=True):
        super(Model, self).__init__()

        base_model = models.efficientnet_b0(weights='DEFAULT')
        if model_freeze:
            for param in base_model.parameters():
                param.requires_grad_(False)
        
        self.model = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=2, stride=2, groups=4),
            nn.Conv2d(4, 4, kernel_size=2, stride=2, groups=4),
            nn.Conv2d(4, 3, kernel_size=1),
            base_model.features,
            base_model.avgpool,
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, 32),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
    

    def unfreeze_params(self):
        for param in self.model.parameters():
            param.requires_grad_(True)