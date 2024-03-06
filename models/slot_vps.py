import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from classifier import Head, Allocated_Head
from pytorchvideo.models.hub import i3d_r50
import inception
import r50

import numpy as np
from math import ceil 

class SlotAttention(nn.Module):
    def __init__(self, dim, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def get_attention(self, slots, inputs):
        slots_prev = slots
        b, n, d = inputs.shape
        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn_ori = dots.softmax(dim=1) + self.eps
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)
        updates = torch.einsum('bjd,bij->bid', v, attn)

        updates = updates.reshape(b, -1, d)
        return updates, attn_ori


    def forward(self, in_slots, inputs):
        b, nf, n, d = inputs.shape
        slots_out = []
        attns = []
        for f in range(nf):
            cur_slots, cur_attn = self.get_attention(in_slots[:,f,:,:],inputs[:,f,:,:])
            slots_out.append(cur_slots)
            attns.append(cur_attn)
        slots_out = torch.stack([slot for slot in slots_out])
        slots_out = slots_out.permute(1,0,2,3) #b, t, n, d 

        attn_out = torch.stack([attn for attn in attns])
        attn_out = attn_out.permute(1,0,2,3)
        return slots_out, attn_out

class TemporalSlotAttention(nn.Module):
    def __init__(self, dim, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)

    def get_attention(self, slots):
        b, t, n, d = slots.shape
        slots = slots.reshape(b, -1, d)
        slots_prev = slots
        norm = self.norm_input(slots)        
        k, v = self.to_k(norm), self.to_v(norm)
        q = self.to_q(norm)

        k = k.reshape(b, t, n, d).permute(0, 2, 1, 3).reshape(b, n, -1)
        v = v.reshape(b, t, n, d).permute(0, 2, 1, 3).reshape(b, n, -1)
        q = q.reshape(b, t, n, d).permute(0, 2, 1, 3).reshape(b, n, -1)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        # print(dots.shape)
        attn_ori = dots.softmax(dim=1) + self.eps
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)
        updates = torch.einsum('bjd,bij->bid', v, attn)

        updates = updates.reshape(b, n, t, d).permute(0, 2, 1, 3)
        updates = updates.reshape(b, t, n, d)
        # print(attn_ori.shape)
        # attn_ori = attn_ori.reshape(b, n, t, d).permute(0, 2, 1, 3)

        return updates



    def forward(self, in_slots):
        cur_slots = self.get_attention(in_slots)
        return cur_slots

class PR(nn.Module):
    def __init__(self, dim):
        super(PR, self).__init__()
        self.self_att = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.ffn = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim) 
        self.ln2 = nn.LayerNorm(dim) 
        self.ln3 = nn.LayerNorm(dim) 
        self.r = SlotAttention(dim)

    def forward(self, pre_slots, x):
        b, t, n, d = pre_slots.shape
        pre_slots = pre_slots.reshape(-1, n, d)
        pre_slots = F.relu(pre_slots)
        mid_slots, _ = self.self_att(pre_slots, pre_slots, pre_slots)
        mid_slots = self.ln1(mid_slots + pre_slots)
        mid_slots = F.relu(mid_slots)
        mid_slots = mid_slots.reshape(b, t, n, d)
        slots, attn = self.r(mid_slots, x)
        slots = slots + mid_slots
        slots = torch.reshape(slots, (-1, n, d))
        slots = self.ln2(slots)
        slots = F.relu(slots)
        slots = self.ln3(self.ffn(slots) + slots)
        slots = torch.reshape(slots, (b, t, n, d))
        return slots, attn

class VR(nn.Module):
    def __init__(self, dim):
        super(VR, self).__init__()
        self.r = TemporalSlotAttention(dim)
        self.ffn = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim) 
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, slots):
        # b, t, n, d = slots.shape
        slots = F.relu(slots)
        slots = self.ln1(self.r(slots) + slots)
        slots = F.relu(slots)
        slots = self.ln2(self.ffn(slots) + slots)
        return slots


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


class SLOT_VPS(nn.Module):
    def __init__(self, args, num_ego_class, num_actor_class, num_slots=21):
        super(SLOT_VPS, self).__init__()
        self.args = args
        self.num_ego_class = num_ego_class
        self.hidden_dim = args.channel
        self.hidden_dim2 = args.channel
        self.slot_dim, self.temp_dim = args.channel, args.channel
        self.ego_c = 128
        self.num_slots = num_slots
        self.resnet = i3d_r50(True)

        if args.dataset == 'nuscenes' and args.pretrain == 'oats':
            self.num_slots = 35
        if args.dataset == 'oats' and args.pretrain == 'taco':
            self.num_slots = 64

        if args.backbone == 'r50':
            self.resnet = r50.R50()
            self.in_c = 2048
            if args.dataset == 'taco':
                self.resolution = (8, 24)
                self.resolution3d = (args.seq_len, 5, 5)
            elif args.dataset == 'oats':
                self.resolution = (7, 7)
                self.resolution3d = (args.seq_len, 7, 7)

        elif args.backbone == 'i3d':
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 2048
            self.resolution = (8, 24)
            self.resolution3d = (4, 8, 24)

        elif args.backbone == 'x3d':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 192

            if args.dataset == 'oats' or args.pretrain == 'oats':
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

        self.encoder_pos = SoftPositionEmbed3D(self.hidden_dim2, [self.resolution3d[0], self.resolution3d[1], self.resolution3d[2]])
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim2)

        self.LN = nn.LayerNorm(self.hidden_dim2) 

        self.pr = PR(self.hidden_dim2)
        self.vr = VR(self.hidden_dim2)
        self.pr2 = PR(self.hidden_dim2)
        self.vr2 = VR(self.hidden_dim2)
        self.pr3 = PR(self.hidden_dim2)
        self.vr3 = VR(self.hidden_dim2)
        self.pr4 = PR(self.hidden_dim2)
        self.vr4 = VR(self.hidden_dim2)
        self.pr5 = PR(self.hidden_dim2)
        self.vr5 = VR(self.hidden_dim2)
        self.pr6 = PR(self.hidden_dim2)
        self.vr6 = VR(self.hidden_dim2)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.hidden_dim2)).cuda()
        self.slots_sigma = torch.randn(1, 1, self.hidden_dim2).cuda()
        self.slots_sigma = nn.Parameter(self.slots_sigma.absolute())
        mu = self.slots_mu.expand(1, self.num_slots, -1)
        sigma = self.slots_sigma.expand(1, self.num_slots, -1)
        slots = torch.normal(mu, sigma)
        slots = slots.contiguous()
        self.register_buffer("slots", slots)

        self.drop = nn.Dropout(p=0.5)         
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

    def extend_slots(self):
        mu = self.slots_mu.expand(1, 29, -1)
        sigma = self.slots_sigma.expand(1, 29, -1)
        slots = torch.normal(mu, sigma)
        slots = slots.contiguous()

        slots = torch.cat((self.slots[:, :-1, :], slots[:, :, :], torch.reshape(self.slots[:, -1, :], (1, 1, -1))), 1)
        self.register_buffer("slots", slots)

    def extract_slots_for_oats(self):

        oats_slot_idx = [
            13, 12, 50, 6, 3,
            55, 1, 0, 5, 10,
            8, 51, 9, 53, 2,
            4, 48, 59, 52, 61,
            63, 49, 60, 7, 30, 
            11, 57, 22, 62, 58,
            18, 54, 29, 17, 25
            ]
        slots = torch.cat(( torch.reshape(self.slots[:, idx, :], (1, 1, -1)) for idx in oats_slot_idx), 1)
        self.register_buffer("slots", slots)
        
    def forward(self, x):
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]

        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]

        if self.args.backbone == 'inception':
            x = torch.reshape(x, (seq_len*batch_size, 3, height, width))
            x = self.resnet(x)
            _, c, h, w  = x.shape
            x = torch.reshape(x, (self.args.seq_len, batch_size, c, h, w))
            x = x.permute(1, 2, 0, 3, 4)
        else:
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
            # [bs, c, n, w, h]
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

        x = self.encoder_pos(x)
        x = torch.reshape(x, (batch_size, -1, self.hidden_dim2))
        # [bs*n, h*w, c]
        x = self.LN(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  
        # [bs, n, h*w, c]
        x = x.view(batch_size, new_seq_len, -1, self.hidden_dim2)

        slots = self.slots.expand(batch_size, new_seq_len, -1, -1)

        slots, _ = self.pr(slots, x)
        slots = self.vr(slots)
        slots, _ = self.pr2(slots, x)
        slots = self.vr2(slots)
        slots, _ = self.pr3(slots, x)
        slots = self.vr3(slots)
        slots, _ = self.pr4(slots, x)
        slots = self.vr4(slots)
        slots, _ = self.pr5(slots, x)
        slots = self.vr5(slots)
        slots, attn_masks = self.pr6(slots, x)
        slots = self.vr6(slots)
        slots = torch.sum(slots, dim=1)
        slots = self.drop(slots)

        b, l, n, thw = attn_masks.shape
        attn_masks = attn_masks.reshape(b*seq_len, n, -1)
        attn_masks = attn_masks.view(attn_masks.shape[0],attn_masks.shape[1],self.resolution[0], self.resolution[1])
        attn_masks = attn_masks.unsqueeze(-1)

        attn_masks = attn_masks.reshape(b*seq_len, n, -1)
        attn_masks = attn_masks.view(attn_masks.shape[0],attn_masks.shape[1],self.resolution[0], self.resolution[1])
        attn_masks = attn_masks.unsqueeze(-1)

        if self.num_ego_class != 0:
            ego_x = self.drop(ego_x)
            ego_x, slots = self.head(slots, ego_x)
            return ego_x, slots, attn_masks.view(b, seq_len, n, self.resolution[0], self.resolution[1])
        else:
            slots = self.head(slots)
            return slots, attn_masks.view(b, seq_len, n, self.resolution[0], self.resolution[1])


