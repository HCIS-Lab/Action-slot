import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from MaskFormer.demo.demo import get_maskformer
from retrieval_head import Head

class CNNGRU(nn.Module):
    def __init__(self, num_ego_class, num_actor_class, road, use_backbone=False):
        super(CNNGRU, self).__init__()
        self.use_backbone = use_backbone
        if self.use_backbone:
            self.in_channel = 3
        else:
            self.in_channel = 2048 
        self.conv1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                    )
        self.bn1 = nn.BatchNorm2d(512)
        self.gru = nn.GRU(512, 256, 1, batch_first=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.bn2 = nn.BatchNorm2d(256)

        self.head = Head(256, num_ego_class, num_actor_class)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hidden = None
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            x = torch.permute(x, (1,0,2,3,4)) #[b, v, 2048, h, w]
            x = torch.reshape(x, (batch_size*seq_len, self.in_channel, height, width)) #[b, 2048, h, w]
        if self.use_backbone:
            x = self.backbone.backbone(x)['res5']
        x = self.conv1(x) #[b*l, 512, h, w]
        height, width = x.shape[2], x.shape[3]
        x = self.pool(x)
        x = torch.reshape(x, (batch_size, seq_len, 512))
        # x = self.dropout(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        ego, actor = self.head(x)
        return ego, actor
