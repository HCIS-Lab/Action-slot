import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np
from utils import *
from base_model import Object_based
from classifier import Head
import random

class MLP(nn.Module):
    def __init__(self,in_c,list_hidden_size):
        super().__init__()
        linears = []
        cur_c = in_c
        for c in list_hidden_size:
            linears.append(nn.Sequential(
                nn.Linear(cur_c,c),
                nn.Dropout(p=0.5),
                nn.ReLU()
            ))
            cur_c = c
        self.linears = nn.ModuleList(linears)
        
    def forward(self,x):
        for l in self.linears:
            x = l(x)
        return x


class ObjectRelationNetwork(nn.Module):
    def __init__(self, size_object, list_hidden_layers_size, relation_type='pairwise-inter'):
        super(ObjectRelationNetwork, self).__init__()
        # Basic Settings
        self.size_object = size_object
        self.list_hidden_layers_size = list_hidden_layers_size
        self.relation_type = relation_type

        self.nb_obj = 2

        # MLP for inferring spatio-temporal relations
        self.mlp_inter = MLP(size_object*self.nb_obj, list_hidden_layers_size)

    @staticmethod
    def create_inter_object_cat(O_1, O_2):
        list_input_mlp, input_mlp = [], None
        K = O_1.size(1)
        for k in range(K):
            O_1_k = O_2[:, k].unsqueeze(1).repeat(1, K, 1) # B,D -> B,1,D -> B,K,D
            O_1_k_input_relation = torch.cat([O_1_k, O_2], dim=2) # B,K,D + B,K,D -> B,K,2*D
            list_input_mlp.append(O_1_k_input_relation)
        # Cat
        input_mlp = torch.cat(list_input_mlp, 1)  # (B, K^2, 2*|O|) b,k,k,2*D
        return input_mlp

    @staticmethod
    def create_triwise_interactions_input(O_1, O_2):
        list_input_mlp, input_mlp = [], None
        K = O_1.size(1)
        for k1 in range(K):
            O_1_k_1 = O_2[:, k1].unsqueeze(1).repeat(1, K, 1)
            list_other_k = [x for x in range(K) if x != k1]
            for k2 in list_other_k:
                O_1_k_2 = O_2[:, k2].unsqueeze(1).repeat(1, K, 1)
                O_1_k_input_relation = torch.cat([O_1_k_1, O_1_k_2, O_2], dim=2)
                list_input_mlp.append(O_1_k_input_relation)
        # Cat
        input_mlp = torch.cat(list_input_mlp, 1)  # (B, K^2, 2*|O|)
        return input_mlp

    def create_input_mlp(self, O_t_1, O_t, D):
        K = O_t.size(1)

        # Input
        input_mlp = self.create_inter_object_cat(O_t_1, O_t)

        # Check if at least an object is involved in the input
        is_first_obj = torch.clamp(torch.sum(input_mlp[:, :, :D], -1), 0, 1)
        is_second_obj = torch.clamp(torch.sum(input_mlp[:, :, D:], -1), 0, 1)
        is_objects = is_first_obj * is_second_obj

        return input_mlp, is_objects

    def compute_O_O_interaction(self, sets_of_objects, t, previous_T, D, sampling=False):

        # Object set (the reference one)
        O_t = sets_of_objects[:, t]

        list_e_inter, list_is_object_inter = [], []
        for t_1 in previous_T:
            # Get the previous object set
            O_t_1 = sets_of_objects[:, t_1]

            # Create the input to feed!
            input_mlp_inter, is_objects_inter = self.create_input_mlp(O_t_1, O_t, D)

            # Infer the relations
            e = self.mlp_inter(input_mlp_inter)

            # Append
            list_e_inter.append(e)
            list_is_object_inter.append(is_objects_inter)

        if (len(list_e_inter) == 1 and self.training):
            # Training so only one interaction computed
            return list_e_inter[0], list_is_object_inter[0]
        else:
            # Stack
            all_e_inter = torch.stack(list_e_inter, 1)
            pooler = nn.AvgPool3d((all_e_inter.size(1), 1, 1))  # or nn.MaxPool3d((all_e_inter.size(1), 1, 1))
            all_e_inter = pooler(all_e_inter)
            B, _, T_prim, D = all_e_inter.size()
            all_e_inter = all_e_inter.view(B, T_prim, D)
            is_objects_inter = torch.stack(list_is_object_inter, 1)
            is_objects_inter = torch.clamp(torch.sum(is_objects_inter, 1), 0, 1)
            return all_e_inter, is_objects_inter

    def forward(self, sets_of_objects, D, sampling=False):

        # Number of timesteps
        B, T, K, _ = sets_of_objects.size()

        list_e, list_is_obj = [], []  # list of the global interaction between two frames
        for t in range(1, T):
            # Sample during training
            previous_T = random.sample(range(t), 1) if self.training else list(range(t))

            # Infer the relation between the two sets of objects
            e_t, is_obj = self.compute_O_O_interaction(sets_of_objects, t, previous_T, D, sampling)

            # Append
            list_e.append(e_t)
            list_is_obj.append(is_obj)

        # Stack
        all_e = torch.stack(list_e, 1)
        all_is_obj = torch.stack(list_is_obj, 1)

        return all_e, all_is_obj

class ORN(Object_based):
    """
    main module of base model for the volleyball
    """
    def __init__(
        self,
        args,
        max_N=0,
        NFB=512,
        K=3,
        ego_c=128,
        num_ego_class=None, 
        num_actor_class=None,
        list_hidden_layers_size=[512,512,512],
        ):
        super().__init__(args,ego_c,K,NFB,max_N)
        self.num_ego_class = num_ego_class
        self.add = False
        self.orn = ObjectRelationNetwork(NFB,list_hidden_layers_size)
        self.head = Head(NFB, num_ego_class, num_actor_class+1, self.ego_c)
        
        
    def forward(self, x, box=False):
        """
        Args:
            box : b,t,N,4
        """
        if not isinstance(box,list):
            box = box.reshape(-1,self.max_N,4)
            # calculate zero object
            # zeros_object = box.sum(-1) # BxT,max_N
            # zeros_object = (zeros_object == 0)
            # zeros_object = zeros_object.unsqueeze(-1).repeat(1,1,self.NFB).float()
            box = list(box)
            
        T = len(x)
        B = x[0].shape[0]
        assert len(box) == B*T
        
        features = self.extract_features(x) # b,d,t,H,W
        
        # predict ego motion
        ego_x = self.conv3d_ego(features)
        ego_x = ego_x.reshape(B, self.ego_c)

        ego_obj, obj_features = self.get_object_features(features,box) # b,t,NFB & b,t,N,NFB
        # zeros_object = zeros_object.reshape(B,T,self.max_N,self.NFB)
        # obj_features = obj_features * zeros_object
        if self.add:
            # add with ego obj
            obj_features = obj_features + ego_obj.unsqueeze(2).repeat(1,1,self.max_N,1)
            all_e, all_is_obj = self.orn(obj_features,self.NFB)
            all_is_obj = all_is_obj.unsqueeze(-1)
            # all_e *= all_is_obj # b,t-1,(N+1)^2,NFB
            all_e = all_e.reshape(B,T-1,self.max_N,self.max_N,-1)
            all_e = all_e.sum(dim=3) # b,t-1,N,NFB
        else:
            # concat with ego obj
            obj_features = torch.cat((ego_obj[:,:,None],obj_features),dim=2) # b,t,N+1,NFB
            all_e, all_is_obj = self.orn(obj_features,self.NFB)
            all_is_obj = all_is_obj.unsqueeze(-1)
            # all_e *= all_is_obj # b,t-1,(N+1)^2,NFB
            all_e = all_e.reshape(B,T-1,self.max_N+1,self.max_N+1,-1)
            all_e = all_e.sum(dim=3)[:,:,1:] # b,t-1,N,NFB
        
        if self.num_ego_class != 0:
            y_ego, y_actor = self.head(all_e,ego_x) # b,t,N, num_actor_class+1
            y_actor = y_actor.mean(dim=1) # b,N,class+1
            return y_ego, y_actor
        else:
            y_actor = self.head(all_e)
            y_actor = y_actor.mean(dim=1) # b,N,class+1
            return y_actor