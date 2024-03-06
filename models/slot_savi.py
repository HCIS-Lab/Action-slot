import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from classifier import Head, Allocated_Head
from pytorchvideo.models.hub import i3d_r50
import inception

import numpy as np
# from models.ConvGRU import *
from math import ceil 

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, num_actor_class=20, eps=1e-8, input_dim=64, resolution=[16, 8, 24]):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_actor_class = num_actor_class
        self.eps = eps
        self.scale = dim ** -0.5
        self.resolution = resolution
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim)).cuda()
        self.slots_sigma = torch.randn(1, 1, dim).cuda()
        self.slots_sigma = nn.Parameter(self.slots_sigma.absolute())

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        
        self.pool = nn.AdaptiveAvgPool1d(1)

    def get_attention(self, slots, inputs):
        b, h, w, d = inputs.shape
        slots_prev = slots
        inputs = torch.reshape(inputs, (b, -1, d))
        b, n, d = inputs.shape
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn_ori = dots.softmax(dim=1) + self.eps
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)
        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )
        slots = slots.reshape(b, -1, d)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def forward(self, inputs, num_slots = None):
        b, nf, h, w, d = inputs.shape
        dtype = inputs.dtype
        slots_out = []
        attns = []

        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_sigma.expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn(mu.shape, dtype = dtype).cuda()
        cur_slots = slots
        for f in range(nf):
            for i in range(3):
                cur_slots, cur_attn = self.get_attention(cur_slots, inputs[:,f,:,:])
            slots_out.append(cur_slots)
            attns.append(cur_attn)
            slots = cur_slots
        slots_out = torch.stack([slot for slot in slots_out])
        slots_out = slots_out.permute(1,0,2,3)
        slots_out = slots_out[:, -1, :self.num_actor_class, :]
        attns = torch.stack([attn for attn in attns])
        attns = attns.permute(1,0,2,3)
        return slots_out, attns


def build_3d_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], resolution[2], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


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

class SLOT_SAVI(nn.Module):
    def __init__(self, args, num_ego_class, num_actor_class, num_slots=21, box=False):
        super(SLOT_SAVI, self).__init__()
        self.args = args
        self.num_ego_class = num_ego_class
        self.hidden_dim = args.channel
        self.hidden_dim2 = args.channel
        self.slot_dim, self.temp_dim = args.channel, args.channel
        self.ego_c = 128
        self.num_slots = num_slots
        self.resnet = i3d_r50(True)

        if args.backbone == 'inception':
            self.resnet = inception.INCEPTION()
            self.in_c = 2048
            if args.dataset == 'taco':
                self.resolution = (8, 24)
                self.resolution3d = (args.seq_len, 5, 5)
            elif args.dataset == 'oats':
                self.resolution = (5, 5)
                self.resolution3d = (args.seq_len, 5, 5)

        elif args.backbone == 'i3d':
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 2048
            self.resolution = (8, 24)
            self.resolution3d = (4, 8, 24)

        elif args.backbone == 'x3d':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 192
            if args.dataset == 'oats':
                self.resolution = (7, 7)
                self.resolution3d = (16, 7, 7)
            else:
                self.resolution = (8, 24)
                self.resolution3d = (16, 8, 24)

        if args.allocated_slot:
            self.head = Allocated_Head(self.slot_dim, num_ego_class, num_actor_class, self.ego_c)
        else:
            self.head = Head(self.slot_dim, num_ego_class, num_actor_class+1, self.ego_c)

        if self.num_ego_class != 0:
            self.conv3d_ego = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm3d(self.in_c),
                    nn.Conv3d(self.in_c, self.ego_c, (1, 1, 1), stride=1),
                    )

        self.conv3d = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(self.in_c),
                nn.Conv3d(self.in_c, self.hidden_dim2, (1, 1, 1), stride=1),
                nn.ReLU(),)

        if args.bg_slot:
            self.slot_attention = SlotAttention(
                num_slots=num_slots+1,
                dim=self.slot_dim,
                eps = 1e-8,
                input_dim=self.hidden_dim2,
                resolution=self.resolution3d,
                num_actor_class = num_actor_class
                ) 
        else:
            self.slot_attention = SlotAttention(
                num_slots=num_slots,
                dim=self.slot_dim,
                eps = 1e-8,
                input_dim=self.hidden_dim2,
                resolution=self.resolution3d,
                num_actor_class = num_actor_class
                ) 

        self.FC1 = nn.Linear(self.hidden_dim2, self.hidden_dim2)
        self.FC2 = nn.Linear(self.hidden_dim2, self.hidden_dim2)
        self.LN = nn.LayerNorm(self.hidden_dim2)

        self.drop = nn.Dropout(p=0.5)         
        self.pe = SoftPositionEmbed3D(self.hidden_dim2, [self.resolution3d[0], self.resolution3d[1], self.resolution3d[2]])
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self, x, box=False):
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[T, b, C, h, w]

        if self.args.backbone == 'inception':
            x = torch.reshape(x, (seq_len*batch_size, 3, height, width))
            x = self.resnet(x)
            _, c, h, w  = x.shape
            x = torch.reshape(x, (self.args.seq_len, batch_size, c, h, w))
            x = x.permute(1, 2, 0, 3, 4)
        else:
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
            for i in range(len(self.resnet)):
                x = self.resnet[i](x)


        if self.num_ego_class != 0:
            ego_x = self.conv3d_ego(x)
            ego_x = self.pool(ego_x)
            ego_x = torch.reshape(ego_x, (batch_size, self.ego_c))

        new_seq_len = x.shape[2]
        new_h, new_w = x.shape[3], x.shape[4]

        # # [b, c, n , w, h]
        x = self.conv3d(x)
        


        x = torch.permute(x, (0, 2, 3, 4, 1))
        # [bs, n, w, h, c]
        x = torch.reshape(x, (batch_size, new_seq_len, new_h, new_w, -1))

        x = self.pe(x)
        x = torch.reshape(x, (batch_size, new_seq_len*new_h*new_w, -1))
        x = self.LN(x)
        x = self.FC1(x)
        x = F.relu(x)
        x = self.FC2(x)
        x = torch.reshape(x, (batch_size, new_seq_len, new_h, new_w, -1))

        x, attn_masks = self.slot_attention(x)

        b, l, n, thw = attn_masks.shape
        attn_masks = attn_masks.reshape(b*seq_len, n, -1)
        attn_masks = attn_masks.view(attn_masks.shape[0],attn_masks.shape[1],self.resolution[0], self.resolution[1])
        attn_masks = attn_masks.unsqueeze(-1)

        attn_masks = attn_masks.reshape(b*seq_len, n, -1)
        attn_masks = attn_masks.view(attn_masks.shape[0],attn_masks.shape[1],self.resolution[0], self.resolution[1])
        attn_masks = attn_masks.unsqueeze(-1)


        # x = torch.sum(x, 1)

        if self.num_ego_class != 0:
            ego_x = self.drop(ego_x)
            ego_x, x = self.head(x, ego_x)
            return ego_x, x, attn_masks.view(b, seq_len, n, self.resolution[0], self.resolution[1])
        else:
            x = self.head(x)
            return x, attn_masks.view(b, seq_len, n, self.resolution[0], self.resolution[1])

