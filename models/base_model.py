import torch
import torch.nn as nn
from pytorchvideo.models.hub import i3d_r50
from torchvision.ops import roi_align

class ROI_ALIGN(nn.Module):
    def __init__(self,kernel_size,scale=1.0):
        super().__init__()
        self.roi_align=roi_align
        self.kernel = kernel_size
        self.scale = scale
    def forward(self,features, boxes):
        return self.roi_align(features, boxes, self.kernel, self.scale, aligned = False)

class Base(nn.Module):
    def __init__(self,args,ego_c):
        super().__init__()
        self.args = args
        self.resnet = i3d_r50(True)
        self.resolution = None
        self.resolution3d = None
        self.in_c = None
        self.path_pool = None
        self.ego_c = ego_c
        self.set_backbone()
        self.conv3d_ego = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(self.in_c),
            nn.Conv3d(self.in_c, self.ego_c, (1, 1, 1), stride=1),
            nn.AdaptiveAvgPool3d(output_size=1),
        )
    
    def set_backbone(self):
        if self.args.backbone == 'i3d-2':
            self.resnet = self.resnet.blocks[:-2]
            self.resolution = (16, 48)
            self.resolution3d = (4, 16, 48)
            self.in_c = 1024
        elif self.args.backbone == 'i3d-1':
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 2048
            self.resolution = (8, 24)
            self.resolution3d = (4, 8, 24)
        elif self.args.backbone == 'x3d-1':
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
            self.resolution = (8, 24)
            self.resolution3d = (16, 8, 24)
        elif self.args.backbone == 'x3d-2':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.resnet = self.resnet.blocks[:-1]
            self.in_c = 192
            self.resolution = (8, 24)
            self.resolution3d = (16, 8, 24)
        elif self.args.backbone == 'x3d-3':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.resnet = self.resnet.blocks[:-2]
            self.in_c = 96
            self.resolution = (16, 48)
            self.resolution3d = (16, 16, 48)
        elif self.args.backbone == 'x3d-4':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.resnet = self.resnet.blocks[:-3]
            self.in_c = 48
            self.resolution = (32, 96)
            self.resolution3d = (16, 32, 96)

        elif self.args.backbone == 'slowfast':
            self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
            self.resnet = self.resnet.blocks[:-2]
            self.path_pool = nn.AdaptiveAvgPool3d((8, 8, 24))
            self.in_c = 2304
            self.resolution = (8, 24)
            self.resolution3d = (8, 8, 24)
            
    def extract_features(self,x):
        seq_len = len(x)
        if self.args.backbone == 'slowfast':
            slow_x = []
            for i in range(0, seq_len, 4):
                slow_x.append(x[i])
            if isinstance(x, list):
                x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
                slow_x = torch.stack(slow_x, dim=0)
                # l, b, c, h, w
                x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
                slow_x = torch.permute(slow_x, (1,2,0,3,4))
                x = [slow_x, x]

                for i in range(len(self.resnet)):
                    x = self.resnet[i](x)
                x[1] = self.path_pool(x[1])
                x = torch.cat((x[0], x[1]), dim=1)
        else:
            if isinstance(x, list):
                x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
                # l, b, c, h, w
                x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
            # [bs, c, n, w, h]

            for i in range(len(self.resnet)):
                # x = self.resnet.blocks[i](x)
                x = self.resnet[i](x)
        return x

class Object_based(Base):
    def __init__(self,args,ego_c,K,NFB,max_N):
        super().__init__(args,ego_c)
        
        self.NFB=NFB
        self.K=K
        self.max_N = max_N
        
        self.roi_align = ROI_ALIGN(K,1.0/64)
        self.fc_emb = nn.Sequential(
            nn.Linear(K*K*self.in_c,NFB),
            nn.LayerNorm(NFB),
            nn.ReLU(),
        )
        self.conv2d_ego_obj = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(self.in_c),
                nn.Conv2d(self.in_c,self.in_c,1),
                nn.AdaptiveAvgPool2d(output_size=K),
        )
        self.ego_emb = nn.Sequential(
            nn.Linear(K*K*self.in_c,NFB),
            nn.LayerNorm(NFB),
            nn.ReLU(),
        )
        
    def get_object_features(self,features,box):
        
        features = features.permute(0,2,1,3,4)
        B,T,_,OH,OW = features.shape
        features = features.reshape(B*T,self.in_c,OH,OW)

        
        ego_obj = self.conv2d_ego_obj(features) # b,t,d,K,K
        ego_obj = ego_obj.reshape(B,T,-1) # b,t,d*K*K
        ego_obj = self.ego_emb(ego_obj) # b,t,NFB
        
        obj_features = self.roi_align(features,box) # b*t,N,d,K,K
        obj_features = obj_features.reshape(B,T,self.max_N,-1) # b,t,N,d*K*K
        obj_features = self.fc_emb(obj_features) # b,t,N,NFB
        
        return ego_obj, obj_features
