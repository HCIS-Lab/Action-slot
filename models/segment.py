import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from retrieval_head import Head
from pytorchvideo.models.hub import i3d_r50
import numpy as np
# from models.ConvGRU import *
from math import ceil 

def build_1D_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

def build_3d_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], resolution[2], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

class SoftPositionEmbed1D(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(2, hidden_size, bias=True)
        self.register_buffer("grid", build_1D_grid(resolution))
    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

class SoftPositionEmbed3D(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(6, hidden_size, bias=True)
        self.register_buffer("grid", build_3d_grid(resolution))
    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

class SEGMENT(nn.Module):
    def __init__(self, args, num_ego_class, num_actor_class):
        super(SEGMENT, self).__init__()
        self.hidden_dim = 128
        self.hidden_dim2 = 128
        self.resnet = i3d_r50(True)
        # self.resnet = self.resnet.blocks[:2]
        if args.backbone == 'i3d-2':
            self.resnet = self.resnet.blocks[:-2]
            self.resolution = (16, 48)
            self.in_c = 1024
        elif args.backbone == 'i3d-1':
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 2048
            self.resolution = (8, 24)
        elif args.backbone == 'x3d-1':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.projection = nn.Sequential(
                nn.Conv3d(192, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
                nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.AvgPool3d(kernel_size=(16, 3, 3), stride=1, padding=0),
                nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
                nn.ReLU())
            self.resnet.blocks[-1] = self.projection
            self.resnet = self.resnet.blocks
            self.in_c = 2048
            self.resolution = (6, 22)
        elif args.backbone == 'x3d-2':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 192
            self.resolution = (8, 24)
            self.resolution3d = (16, 8, 24)
        elif args.backbone == 'x3d-3':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.resnet = self.resnet.blocks[:-2]
            self.in_c = 96
            self.resolution = (16, 48)
        elif args.backbone == 'x3d-4':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.resnet = self.resnet.blocks[:-3]
            self.in_c = 48
            self.resolution = (32, 96)

        self.head = Head(self.hidden_dim2, num_ego_class, num_actor_class, self.in_c)
        # self.model.blocks[-1] = nn.Sequential(
        #                         nn.Dropout(p=0.5, inplace=False),
        #                         # nn.Linear(in_features=2304, out_features=400, bias=True),
        #                         self.head,
        #                         )
        # 64 192

        self.conv3d = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(self.in_c),
                nn.Conv3d(self.in_c, self.hidden_dim, (1, 1, 1), stride=1))

        self.fc = nn.Linear(self.hidden_dim2, self.hidden_dim2)
        self.drop = nn.Dropout(p=0.5)         
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

        self.len_seg = 4
        # self.self_attention = nn.MultiheadAttention(self.hidden_dim2, 1, batch_first=True)
        # self.pe = SoftPositionEmbed3D(self.hidden_dim2, [self.len_seg, self.resolution[0], self.resolution[1]])
        # self.ln = nn.Sequential(
        #         nn.ReLU(),
        #         nn.LayerNorm(self.hidden_dim2))
        # self.ffn = nn.Sequential(
        #         nn.Linear(self.hidden_dim2, self.hidden_dim2),
        #         nn.LayerNorm(self.hidden_dim2),
        #         nn.ReLU())
        # self.temp_pe = SoftPositionEmbed1D(self.hidden_dim2, [7])
        # self.temp_attention = nn.MultiheadAttention(self.hidden_dim2, 1, batch_first=True)
        # self.temp_ln = nn.Sequential(
        #         nn.ReLU(),
        #         nn.LayerNorm(self.hidden_dim2))
        # self.temp_ffn = nn.Sequential(
        #         nn.Linear(self.hidden_dim2, self.hidden_dim2),
        #         nn.LayerNorm(self.hidden_dim2),
        #         nn.ReLU())
        
    def forward(self, x):
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            # l, b, c, h, w
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
        # [bs, c, n, w, h]
        for i in range(len(self.resnet)):
            # x = self.resnet.blocks[i](x)
            x = self.resnet[i](x)

        ego_x = self.pool(x)
        ego_x = torch.reshape(ego_x, (batch_size, self.in_c))

        new_seq_len = x.shape[2]
        new_h, new_w = x.shape[3], x.shape[4]

        # # conv3d1x1
        # # [b, c, n , w, h]
        x = self.conv3d(x)
        segs = []
        for f in range(0, 13, 2):
            seg = x[:, :, f:f+self.len_seg, :, :]
            # seg_1 = x[:, :, f:f+self.len_seg, :, :]
            # seg_1 = torch.permute(seg_1, (0, 2, 3, 4, 1))

            # seg_1 = self.pe(seg_1)
            # seg_1 = torch.reshape(seg_1, (batch_size, -1, self.hidden_dim2))
            # seg, _ = self.self_attention(seg_1, seg_1, seg_1)
            # seg = self.ln(seg + seg_1)
            # seg = self.ffn(seg) + seg
            # seg = torch.reshape(seg, (batch_size, -1, new_h, new_w, self.hidden_dim2))
            # seg = torch.permute(seg, (0, 4, 1, 2, 3))

            seg = self.pool(seg)
            seg =torch.reshape(seg, (batch_size, -1))
            seg = self.fc(seg)
            segs.append(seg)
        segs = torch.stack(segs, dim=0)
        # l, b, c
        segs = torch.permute(segs, (1, 0, 2))

        # segs = self.temp_pe(segs)
        # segs, _ = self.temp_attention(segs, segs, segs)
        # segs = self.ln(segs) + segs
        # segs = self.ffn(segs) + segs

        segs = torch.sum(segs, 1)

        segs = self.drop(segs)
        ego_x = self.drop(ego_x)
        ego_x, segs = self.head(segs, ego_x)
        # return ego_x, x, attn_masks.view(batch_size,new_seq_len, self.num_slots, attn_masks.shape[2], attn_masks.shape[3])
        return ego_x, segs, 1
