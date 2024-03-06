import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from classifier import Head

class SlowFast(nn.Module):
    def __init__(self, num_ego_class, num_actor_class):
        super(SlowFast, self).__init__()
        self.num_ego_class = num_ego_class
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)


        self.head = Head(2304, num_ego_class, num_actor_class)
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
        slow_x = []
        for i in range(0, seq_len, 4):
        	slow_x.append(x[i])
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            slow_x = torch.stack(slow_x, dim=0)
            # l, b, c, h, w
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
            slow_x = torch.permute(slow_x, (1,2,0,3,4))
        num_block = len(self.model.blocks)

        x = [slow_x, x]
        for i in range(num_block-1):
            x = self.model.blocks[i](x)

        x = self.pool(x)
        x = torch.reshape(x, (batch_size, -1))

        if self.num_ego_class != 0:
            ego, x = self.model.blocks[-1](x)
            return ego, x
        else:
            x = self.model.blocks[-1](x)
            return x
