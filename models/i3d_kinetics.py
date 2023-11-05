import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from classifier import Head
from pytorchvideo.models.hub import i3d_r50


class I3D_KINETICS(nn.Module):
    def __init__(self, num_ego_class, num_actor_class):
        super(I3D_KINETICS, self).__init__()
        self.num_ego_class = num_ego_class
        self.model = i3d_r50(True)
        self.head = Head(2048, num_ego_class, num_actor_class)
        self.model.blocks[-1] = nn.Sequential(
                                nn.Dropout(p=0.5, inplace=False),
                                # nn.Linear(in_features=2304, out_features=400, bias=True),
                                self.head,
                                )
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self, x):
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            # l, b, c, h, w
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
        num_block = len(self.model.blocks)
        # print(x.shape)
        for i in range(num_block-1):
            x = self.model.blocks[i](x)
        # b, 2048, 4, 8, 24        
        x = self.pool(x)
        x = torch.reshape(x, (batch_size, 2048))
        if self.num_ego_class != 0:
            ego, x = self.model.blocks[-1](x)
            return ego, x
        else:
            x = self.model.blocks[-1](x)
            return x
