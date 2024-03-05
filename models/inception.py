import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from classifier import Head

class INCEPTION(nn.Module):
    def __init__(self):
        super(INCEPTION, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        # for i, b in enumerate(self.model.blocks):
        #     print(i)
        #     print(b)
        self.blocks = []
        self.blocks.append(self.model.Conv2d_1a_3x3)
        self.blocks.append(self.model.Conv2d_2a_3x3)
        self.blocks.append(self.model.Conv2d_2b_3x3)
        self.blocks.append(self.model.maxpool1)
        self.blocks.append(self.model.Conv2d_3b_1x1)
        self.blocks.append(self.model.Conv2d_4a_3x3)
        self.blocks.append(self.model.maxpool2)
        self.blocks.append(self.model.Mixed_5b)
        self.blocks.append(self.model.Mixed_5c)
        self.blocks.append(self.model.Mixed_5d)
        self.blocks.append(self.model.Mixed_6a)
        self.blocks.append(self.model.Mixed_6b)
        self.blocks.append(self.model.Mixed_6c)
        self.blocks.append(self.model.Mixed_6d)
        self.blocks.append(self.model.Mixed_6e)
        # self.blocks.append(self.model.Mixed_7a)
        # self.blocks.append(self.model.Mixed_7b)
        # self.blocks.append(self.model.Mixed_7c)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
