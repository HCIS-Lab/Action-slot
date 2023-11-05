import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np
import sys
sys.path.append('/media/hcis-s19/DATA/Action-Slot/scripts')
from utils import *
from base_model import Object_based
from classifier import Head

def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B=X.shape[0]
    
    rx=X.pow(2).sum(dim=2).reshape((B,-1,1))
    ry=Y.pow(2).sum(dim=2).reshape((B,-1,1))
    
    dist=rx-2.0*X.matmul(Y.transpose(1,2))+ry.transpose(1,2)
    
    return torch.sqrt(dist)

class GCN_Module(nn.Module):
    def __init__(self, NFR, NG, NFG):
        super(GCN_Module, self).__init__()
        
        self.pos_threshold = 0.2
        self.NFR = NFR
        self.NG = NG
        self.NFG = NFG
        NFG_ONE=NFG
        
        
        self.fc_rn_theta_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        self.fc_rn_phi_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        
        
        self.fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])
        
        self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([NFG_ONE]) for i in range(NG) ])
        
            

        
    def forward(self,graph_boxes_features,boxes_in_flat=None,OW=None):
        
        # GCN graph modeling
        # Prepare boxes similarity relation
        B,N = graph_boxes_features.shape[:2]
        
        pos_threshold=self.pos_threshold
        
        # Prepare position mask
        # graph_boxes_positions=boxes_in_flat  #B*T*N, 4
        # graph_boxes_positions[:,0]=(graph_boxes_positions[:,0] + graph_boxes_positions[:,2]) / 2 
        # graph_boxes_positions[:,1]=(graph_boxes_positions[:,1] + graph_boxes_positions[:,3]) / 2 
        # graph_boxes_positions=graph_boxes_positions[:,:2].reshape(B,N,2)  #B*T, N, 2
        
        # graph_boxes_distances=calc_pairwise_distance_3d(graph_boxes_positions,graph_boxes_positions)  #B, N, N
        
        # position_mask=( graph_boxes_distances > (pos_threshold*OW) )
        
        
        relation_graph=None
        graph_boxes_features_list=[]
        for i in range(self.NG):
            graph_boxes_features_theta=self.fc_rn_theta_list[i](graph_boxes_features)  #B,N,NFR
            graph_boxes_features_phi=self.fc_rn_phi_list[i](graph_boxes_features)  #B,N,NFR

#             graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
#             graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph=torch.matmul(graph_boxes_features_theta,graph_boxes_features_phi.transpose(1,2))  #B,N,N

            similarity_relation_graph=similarity_relation_graph/np.sqrt(self.NFR)

            similarity_relation_graph=similarity_relation_graph.reshape(-1,1)  #B*N*N, 1
            
        
        
            # Build relation graph
            relation_graph=similarity_relation_graph

            relation_graph = relation_graph.reshape(B,N,N)

            # relation_graph[position_mask]=-float('inf')

            relation_graph = torch.softmax(relation_graph,dim=2)       
        
            # Graph convolution
            one_graph_boxes_features=self.fc_gcn_list[i]( torch.matmul(relation_graph,graph_boxes_features) )  #B, N, NFG_ONE
            one_graph_boxes_features=self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features=F.relu(one_graph_boxes_features)
            
            graph_boxes_features_list.append(one_graph_boxes_features)
        
        graph_boxes_features=torch.sum(torch.stack(graph_boxes_features_list),dim=0) #B, N, NFG
        
        return graph_boxes_features,relation_graph

class ARG(Object_based):
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
        gcn_layers=1,
        NFR=256,
        ):
        super().__init__(args,ego_c,K,NFB,max_N)
        self.num_ego_class = num_ego_class
        self.NFR = NFR
        
        self.head = Head(NFB, num_ego_class, num_actor_class+1, self.ego_c)
        
        self.gcn_list = torch.nn.ModuleList([GCN_Module(NFR, 16, NFB)  for _ in range(gcn_layers) ])    
        
        self.dropout_global=nn.Dropout(p=0.3)
        
    def forward(self, x, box=False):
        """
        Args:
            box : b,t,N,4
        """
        if not isinstance(box,list):
            box = box.reshape(-1,self.max_N,4)
            box = list(box)
            
        T = len(x)
        B = x[0].shape[0]
        assert len(box) == B*T
        
        features = self.extract_features(x) # b,d,t,H,W
        
        ego_x = self.conv3d_ego(features)
        ego_x = ego_x.reshape(B, self.ego_c)

        ego_obj, obj_features = self.get_object_features(features,box)
        # concat with ego obj
        obj_features = torch.cat((ego_obj[:,:,None],obj_features),dim=2) # b,t,N+1,NFB
        
        graph_boxes_features = obj_features.reshape(B,T*(self.max_N+1),self.NFB)
        
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features)
        
        graph_boxes_features = graph_boxes_features.reshape(B,T,self.max_N+1,self.NFB)  
        obj_features = obj_features.reshape(B,T,self.max_N+1,self.NFB)
        
        boxes_states = graph_boxes_features+obj_features
    
        boxes_states = self.dropout_global(boxes_states) # b,t,N+1,NFB
        
        if self.num_ego_class != 0:
            y_ego, y_actor = self.head(boxes_states[:,:,1:],ego_x) # b,t,N, num_actor_class+1
            y_actor = y_actor.mean(dim=1) # b,N,class+1
            return y_ego, y_actor
        else:
            y_actor = self.head(boxes_states[:,:,1:])
            y_actor = y_actor.mean(dim=1) # b,N,class+1
            return y_actor
        
