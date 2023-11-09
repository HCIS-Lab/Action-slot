import torch
import torch.nn as nn

class R50(nn.Module):
    def __init__(self):
        super(R50, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        # for i, b in enumerate(self.model.blocks):
        #     print(i)
        #     print(b)
        self.blocks = []
        self.blocks.append(self.model.conv1)        
        self.blocks.append(self.model.bn1)
        self.blocks.append(self.model.relu)
        self.blocks.append(self.model.maxpool)

        self.blocks.append(self.model.layer1)
        self.blocks.append(self.model.layer2)
        self.blocks.append(self.model.layer3)
        self.blocks.append(self.model.layer4)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x