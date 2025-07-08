import torch.nn as nn
import torch
from torch.nn.functional import normalize
from typing import List,Tuple

class ProjectonHead(nn.Module):
    def __init__(self,blocks:List[Tuple[int,int,nn.Module]]):
        super(ProjectonHead,self).__init__() 
        
        self.layers = []
        for input_dim,output_dim,batch_norm,non_linearity in blocks:
            self.layers.append(nn.Linear(input_dim,output_dim))
            if batch_norm:
                self.layers.append(batch_norm)
            if non_linearity:
                self.layers.append(non_linearity)
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self,x:torch.Tensor):
        return self.layers(x)


class NNCLRProjectonHead(ProjectonHead):
    def __init__(self,input_dim:int,hidden_dim:int,output_dim:int):
        super(NNCLRProjectonHead,self).__init__([
            (input_dim,hidden_dim,nn.BatchNorm1d(hidden_dim),nn.ReLU()),
            (hidden_dim,hidden_dim,nn.BatchNorm1d(hidden_dim),nn.ReLU()),
            (hidden_dim,output_dim,nn.BatchNorm1d(output_dim),None),
        ])
        
class NNCLRPredictonHead(ProjectonHead):
    def __init__(self,input_dim:int,hidden_dim:int,output_dim:int):
        super(NNCLRPredictonHead,self).__init__([
            (input_dim,hidden_dim,nn.BatchNorm1d(hidden_dim),nn.ReLU()),
            (hidden_dim,output_dim,None,None),
        ])

class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
#         self.instance_projector = nn.Sequential(
#             nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
#             nn.ReLU(),
#             nn.Linear(self.resnet.rep_dim, self.feature_dim),
#         )
        self.instance_projector = NNCLRProjectonHead(512,512,128)
        self.instance_prediction = NNCLRPredictonHead(128,512,128)
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        #z_j = normalize(self.instance_projector(h_j), dim=1)
        z_j = normalize(self.instance_projector(h_j),dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
