import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from retrieval_head import Head, Instance_Head
from pytorchvideo.models.hub import i3d_r50
import numpy as np
# from models.ConvGRU import *
from math import ceil 

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, num_actor_class=20, eps=1e-8, input_dim=64, resolution=[8, 24]):
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


        self.FC1 = nn.Linear(dim, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        # self.temp = nn.Sequential(
        #         nn.LayerNorm(input_dim),
        #         nn.ReLU(),
        #         nn.Linear(input_dim, dim),
        #         nn.ReLU()
        #         )

        mu = self.slots_mu.expand(1, self.num_slots, -1)
        sigma = self.slots_sigma.expand(1, self.num_slots, -1)
        slots = torch.normal(mu, sigma)
        self.len_seg = 4
        # self.PE = SoftPositionEmbed(dim, [self.len_seg, 8*24])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pe = SoftPositionEmbed3D(dim, [self.len_seg, resolution[0], resolution[1]])
        # self.sa = nn.MultiheadAttention(dim, 1, batch_first=True)
        # self.ln = nn.Sequential(
        #         nn.LayerNorm(dim),
        #         nn.ReLU(),)
        # self.ffn = nn.Sequential(
        #         nn.Linear(dim, dim),
        #         nn.LayerNorm(dim),
        #         nn.ReLU()
        #             )

        # self.p_pe = SoftPositionEmbed(dim, [resolution[0], resolution[1]])
        # self.p_ln = nn.Sequential(
        #         nn.LayerNorm(dim),
        #         nn.ReLU(),)
        # self.p_sa = nn.MultiheadAttention(dim, 1, batch_first=True)

        # self.slot_pe = SoftPositionEmbed(dim, [resolution[0], resolution[1]])


        slots = slots.contiguous()
        self.register_buffer("slots", slots)

    def get_attention(self, slots, inputs):
        b, l, h, w, d = inputs.shape
        inputs = self.pe(inputs)
        inputs = torch.reshape(inputs, (b, -1, d))
        seg, _ = self.sa(inputs, inputs, inputs)
        inputs = self.ln(inputs + seg)
        inputs = self.ffn(inputs) + inputs
        inputs = torch.reshape(inputs, (b, l, h*w, d))
        inputs = torch.permute(inputs, (0, 2, 3, 1))
        # b, h*w, d, l
        inputs = torch.reshape(inputs, (b, -1, l))
        # b, h*w*d, l
        inputs = F.relu(inputs)
        inputs = self.pool(inputs)
        # b, h*w*d, 1

        # inputs = torch.reshape(inputs, (b, h, w, d))
        # inputs = self.p_pe(inputs)
        # inputs = torch.reshape(inputs, (b, -1, d))
        # inputs = self.p_ln(inputs)
        # inputs = F.relu(inputs)
        # inputs, _ = self.p_sa(inputs, inputs, inputs)

        inputs = torch.reshape(inputs, (b, h, w, d))
        inputs = self.slot_pe(inputs)
        inputs = torch.reshape(inputs, (b, -1, d))

        slots_prev = slots
        b, n, d = inputs.shape
        inputs = F.relu(self.norm_input(inputs))
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = F.relu(self.norm_slots(slots))
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn_ori = dots.softmax(dim=1) + self.eps
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)
        # slots = torch.einsum('bjd,bij->bid', v, attn)
        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )

        slots = slots.reshape(b, -1, d)
        slots = slots[:, :self.num_actor_class, :]
        slots = F.relu(slots)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def get_3d_slot(self, slots, inputs):
        b, l, h, w, d = inputs.shape
        inputs = self.pe(inputs)
        inputs = torch.reshape(inputs, (b, -1, d))

        inputs = self.LN(inputs)
        inputs = self.FC1(inputs)
        inputs = F.relu(inputs)
        inputs = self.FC2(inputs)

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
        # slots = torch.einsum('bjd,bij->bid', v, attn)
        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )
        slots = slots.reshape(b, -1, d)
        slots = slots[:, :self.num_actor_class, :]
        # slots = F.relu(slots)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def forward(self, inputs, num_slots = None):
        b, nf, h, w, d = inputs.shape
        slots_out = []
        attns = []
        slots = self.slots.expand(b,-1,-1)
        # pre-attention for the first frame
        # slots, _ = self.get_attention(slots, inputs[:,0,:,:])

        for f in range(0, 13, 2):
            # pre_slots = cur_slots
            # cur_slots = slots
            # for i in range(3):
            cur_slots, cur_attn = self.get_3d_slot(slots, inputs[:, f:f+4, :, :, :])
                # cur_slots, cur_attn = self.get_3d_slot(cur_slots,inputs[:,f:f+self.len_seg,:,:])
            # slots_out.append(cur_slots)
            # cur_slots = self.tem_gru(cur_slots.reshape(-1, d),pre_slots.reshape(-1, d))
            # cur_slots = cur_slots.reshape(b, -1, d)
            slots_out.append(cur_slots)
            attns.append(cur_attn)
            # slots = cur_slots
        slots_out = torch.stack([slot for slot in slots_out])
        slots_out = slots_out.permute(1,0,2,3)
        attns = torch.stack([attn for attn in attns])
        # b, l, n, hw
        attns = attns.permute(1,0,2,3)
        return slots_out, attns


def build_1D_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
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

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.register_buffer("grid", build_grid(resolution))
    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc



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

class SLOT_SEG(nn.Module):
    def __init__(self, args, num_ego_class, num_actor_class, num_slots=21, box=False):
        super(SLOT_SEG, self).__init__()
        self.hidden_dim = args.channel
        self.hidden_dim2 = args.channel
        self.slot_dim, self.temp_dim = args.channel, args.channel
        self.ego_c = 128
        self.num_slots = num_slots
        # self.slot_hidden_dim = 512
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
            # self.projection = nn.Sequential(
            #     nn.Conv3d(192, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            #     nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #     nn.ReLU(),
            #     nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=0),
            #     nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            #     )
            # self.projection = nn.Sequential(
            #     nn.Conv3d(192, 192, kernel_size=(3, 3, 3), dilation=(3, 2, 2), stride=(1, 1, 1), padding='same', bias=False),
            #     nn.BatchNorm3d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #     nn.ReLU(),
            #     nn.Conv3d(192, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            #     nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #     nn.ReLU(),
            #     nn.Conv3d(432, 512 , kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            #     )
            self.projection = nn.Sequential(
                nn.Conv3d(192, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
                nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv3d(256, 256, kernel_size=(3, 3, 3), dilation=(3, 1, 1), stride=(1, 1, 1), padding='same', bias=False),
                nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
            self.resnet.blocks[-1] = self.projection
            self.resnet = self.resnet.blocks
            self.in_c = 256
            # self.resolution = (6, 22)
            # self.resolution3d = (16, 6, 22)
            self.resolution = (8, 24)
            self.resolution3d = (16, 8, 24)
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
            self.resolution3d = (16, 16, 48)
        elif args.backbone == 'x3d-4':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.resnet = self.resnet.blocks[:-3]
            self.in_c = 48
            self.resolution = (32, 96)
            self.resolution3d = (16, 32, 96)

        if args.fix_slot:
            self.head = Instance_Head(self.slot_dim, num_ego_class, num_actor_class, self.ego_c)
        else:
            self.head = Head(self.slot_dim, num_ego_class, num_actor_class+1, self.ego_c)

        # 64 192
        # self.encoder_pos = SoftPositionEmbed(self.in_c, self.resolution)
        # self.encoder_pos = SoftPositionEmbed3D(self.hidden_dim2, self.resolution3d)
        # self.encoder_pos = PositionalEncoding3D(self.in_c)

        # self.fc1 = nn.Linear(self.in_c, self.hidden_dim)
        # self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim2)
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

        if args.seg:
            self.slot_attention = SlotAttention(
                num_slots=num_slots+1,
                dim=self.slot_dim,
                eps = 1e-8,
                input_dim=self.hidden_dim2,
                resolution=self.resolution,
                num_actor_class = num_actor_class
                ) 
        else:
            self.slot_attention = SlotAttention(
                num_slots=num_slots,
                dim=self.slot_dim,
                eps = 1e-8,
                input_dim=self.hidden_dim2,
                resolution=self.resolution,
                num_actor_class = num_actor_class
                ) 
        # self.LN = nn.LayerNorm([self.resolution[0]*self.resolution[1], self.in_c]) 
        # self.LN = nn.LayerNorm([self.resolution3d[0]*self.resolution3d[1]*self.resolution3d[2], self.hidden_dim2]) 
        # self.video_slot = VideoSlotAttention(
        #     num_slots=20,
        #     dim=self.hidden_dim2,
        #     eps = 1e-8, 
        #     hidden_dim = self.hidden_dim2) 
        self.drop = nn.Dropout(p=0.5)         
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)
        
        self.box = box
        if self.box:
            self.box_projection = nn.Sequential(
                nn.Linear(4, 128),
                nn.LayerNorm([35, 128]) ,
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.LayerNorm([35, 256]),
                nn.ReLU())
            self.box_attention = SelfAttention(256)

        self.self_attention = nn.MultiheadAttention(self.slot_dim, 1, batch_first=True)
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_pos = SoftPositionEmbed1D(self.slot_dim, [7])
        self.ffn = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.slot_dim, self.slot_dim),
                nn.LayerNorm(self.slot_dim),
                nn.ReLU()
                )

    def forward(self, x, box=False):
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            # l, b, c, h, w
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
        # [bs, c, n, w, h]

        if self.box:
            # [l, b, n, 4]
            box = torch.stack(box, dim=0)
            # [b, l, n, 4]
            box = torch.permute(box, (1, 0, 2, 3))
            num_box = box.shape[2]
            box = torch.reshape(box, (-1, num_box, 4))
            box = self.box_projection(box)
            box = torch.reshape(box, (batch_size, seq_len, num_box, -1))
        for i in range(len(self.resnet)):
            # x = self.resnet.blocks[i](x)
            x = self.resnet[i](x)

        ego_x = self.conv3d_ego(x)

        new_seq_len = x.shape[2]
        new_h, new_w = x.shape[3], x.shape[4]

        # # conv3d1x1
        # # [b, c, n , w, h]
        x = self.conv3d(x)

        ego_x = self.pool(ego_x)
        ego_x = torch.reshape(ego_x, (batch_size, self.ego_c))

        # x = torch.reshape(x, (-1, self.in_c, self.resolution[0], self.resolution[1]))
        # x = self.conv1(x)
        # # [bn, c, w, h]
        # x = torch.reshape(x, (batch_size, seq_len, -1, self.resolution[0], self.resolution[1]))
        # # [bs, n, c, w, h]
        # x = torch.permute(x, (0, 1, 3, 4, 2))
        # pe
        x = torch.permute(x, (0, 2, 3, 4, 1))
        # [bs, n, w, h, c]
        # x = torch.reshape(x, (batch_size*new_seq_len, new_h, new_w, -1))
        x = torch.reshape(x, (batch_size, new_seq_len, new_h, new_w, -1))
        # [bs*n, h, w, c]

        if self.box:
            x = self.box_attention(x, box)
        x, attn_masks = self.slot_attention(x)


        # # pool segment
        # new_seq_len = attn_masks.shape[1]
        # attn_masks = attn_masks.reshape(batch_size*new_seq_len,self.num_slots+1, -1)
        # attn_masks = attn_masks.view(attn_masks.shape[0],self.num_slots+1, self.resolution[0], self.resolution[1])
        # attn_masks = attn_masks.unsqueeze(-1)

        # attn_masks = attn_masks.reshape(batch_size*new_seq_len,self.num_slots+1, -1)
        # attn_masks = attn_masks.view(attn_masks.shape[0],self.num_slots+1, self.resolution[0], self.resolution[1])
        # attn_masks = attn_masks.unsqueeze(-1)
        
        # attn_masks = attn_masks.view(batch_size, new_seq_len, self.num_slots+1, attn_masks.shape[2], attn_masks.shape[3])
        
        # no pool, 3d slot
        b, s, n, thw = attn_masks.shape
        attn_masks = attn_masks.reshape(b*s,n, -1)
        attn_masks = attn_masks.view(attn_masks.shape[0], n, 4, self.resolution[0], self.resolution[1])
        attn_masks = attn_masks.unsqueeze(-1)
        # b*s, n, 4, h, w, 1
        attn_masks = attn_masks.reshape(b*s, n, -1)
        # b*s, n, 4*h*w
        attn_masks = attn_masks.view(attn_masks.shape[0], n, 4, self.resolution[0], self.resolution[1])
        # b*s, n, 4, h, w
        attn_masks = attn_masks.unsqueeze(-1)
        # b*s, n, 4, h, w, 1
        attn_masks = attn_masks.view(b, s, n, 4, attn_masks.shape[3], attn_masks.shape[4])
        attn_masks = torch.permute(attn_masks, (0, 2, 1, 3, 4, 5))
        # b, n, s, 4, h, w
        new_masks = attn_masks[:, :, 1, :, :, :]
        # b, n, 4, h, w
        for s_i in range(1, s):
            new_masks[:, :, -2:, :, :] += attn_masks[:, :, s_i, :2, :, :]
            new_masks[:, :, -2:, :, :] /= 2.0
            new_masks = torch.cat((new_masks, attn_masks[:, :, s_i, -2:, :, :]), dim=2)
        new_masks = new_masks.reshape(b, n, 16, new_h, new_w)
        new_masks = torch.permute(new_masks, (0, 2, 1, 3, 4))
        # b, l, n, h, w

        # # x = torch.sum(x, 1)
        # # b, l, n, c
        # x = torch.permute(x, (0, 2, 1, 3))
        # x = torch.reshape(x, (batch_size*self.num_slots, -1, self.slot_dim))
        # x = self.slot_pos(x)
        # # b, l, n, c
        # # b, lxn, c
        # x = F.relu(x)
        # segs, _ = self.self_attention(x, x, x)
        # x = self.slot_norm(x+segs)
        # x = self.ffn(x) + x
        # x = torch.reshape(x, (batch_size, self.num_slots, -1, self.slot_dim))
        # x = torch.permute(x, (0, 2, 1, 3))


        x = torch.sum(x, 1)
        x = self.drop(x)
        ego_x = self.drop(ego_x)
        ego_x, x = self.head(x, ego_x)
        return ego_x, x, new_masks
