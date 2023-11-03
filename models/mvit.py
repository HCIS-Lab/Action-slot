from pytorchvideo.models.hub import mvit_base_16x4
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier import Head

class pos_encode_custom(nn.Module):
    """
        override pytorchvideo SpatioTemporalClsPositionalEncoding 
    """
    def __init__(self,SpatioTemporalClsPositionalEncoding,scale,mode='linear'):
        super().__init__()
        H, W = 512/scale, 1536/scale
        self.layer = SpatioTemporalClsPositionalEncoding	
        self.spatial_patch = int(H*W/16)
        self.patch_embed_shape = [8,int(H/4),int(W/4)]
        self.mode = mode
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        B, N, C = x.shape
        if self.layer.cls_embed_on:
            cls_tokens = self.layer.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.layer.sep_pos_embed:
            spatial_embed = self.layer.pos_embed_spatial.permute(0,2,1) # B, C, N

            # 1D
            if self.mode == 'linear':
                spatial_embed = nn.functional.interpolate(
                    spatial_embed, size=(self.patch_embed_shape[1]*self.patch_embed_shape[2]), mode=self.mode, align_corners=False)
            # 2D
            else:
                spatial_embed = spatial_embed.reshape(1,C,56,56)
                spatial_embed = nn.functional.interpolate(
                    spatial_embed, size=(self.patch_embed_shape[1],self.patch_embed_shape[2]), mode=self.mode, align_corners=False)
                spatial_embed = spatial_embed.reshape(1,C,-1)

            spatial_embed = spatial_embed.permute(0,2,1)
            pos_embed = spatial_embed.repeat(
                1, self.layer.num_temporal_patch, 1
            ) + torch.repeat_interleave(
                self.layer.pos_embed_temporal,
                self.spatial_patch,
                dim=1,
            )
            if self.layer.cls_embed_on:
                pos_embed = torch.cat([self.layer.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.layer.pos_embed
        return x

class MViT(nn.Module):
    def __init__(self, num_ego_class, num_actor_class,scale,mode='linear'):
        super(MViT, self).__init__()
        self.scale = scale
        self.model = mvit_base_16x4(True)
        # model 16 blocks
        cls = Head(768, num_ego_class, num_actor_class)
        self.head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            # nn.Linear(in_features=2304, out_features=400, bias=True),
            cls,
        )
        if scale != -1.0:
            self.custom_posembed = pos_encode_custom(self.model.cls_positional_encoding,scale,mode)
    def forward(self, x):
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            # l, b, c, h, w
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
        num_block = len(self.model.blocks)
        # torch.Size([8, 3, 16, 224, 224])
        x = self.model.patch_embed(x) # torch.Size([8, 25088, 96])
        # x = self.model.cls_positional_encoding(x) # torch.Size([8, 25089, 96])
        # x = self.model.pos_drop(x)
        if self.scale == -1.0:
            x = self.model.cls_positional_encoding(x)
            thw = self.model.cls_positional_encoding.patch_embed_shape
        else:
            x = self.custom_posembed(x)
            thw = self.custom_posembed.patch_embed_shape
        attn_l = []
        for blk in self.model.blocks:
            x, thw = blk(x, thw) # B, D_index, N, N (N = *thw+1)
            attn_l.append([x[:,1:],thw])
        x = self.model.norm_embed(x)
        # https://github.com/facebookresearch/pytorchvideo/blob/702f9f42569598c5cce8c5e2dd7e37c3d6c46efd/pytorchvideo/models/head.py#L11
        x = x[:, 0]
        # x = x.mean(1)
        # x = torch.reshape(x, (batch_size, 768))
        ego, x = self.head(x)

        return ego, x, attn_l # b,D_index, N-1,N-1

class VideoMAE_pretrained(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,video_mae,num_ego_class, num_actor_class):
        super(VideoMAE_pretrained, self).__init__()
        
        self.model = video_mae
        self.head  = Head(768, num_ego_class, num_actor_class)
    
    def forward(self, x):
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[T, b, C, h, w]
            # l, b, c, h, w
            x = torch.permute(x, (1,2,0,3,4)) #[b, C, T, h, w]
        x = self.model.forward_features(x)
        ego, x  = self.head(self.model.fc_dropout(x))
        
        return ego, x
