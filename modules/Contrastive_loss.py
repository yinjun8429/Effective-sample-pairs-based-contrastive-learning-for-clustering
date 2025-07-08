import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np 

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device,tau_plus,beta,estimator):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.tau_plus = tau_plus
        self.beta = beta
        self.estimator = estimator
        
        self.mask = self.mask_correlated_samples(batch_size)
        #self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask 
    
    def criterion(self,z_i,z_j,device,tau_plus,beta,estimator):
        # neg score
#         z = torch.cat([z_i, z_j], dim=0)

#         neg = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
#         z_neg = neg.clone()#深拷贝 ，获得的新tensor和原来的数据不再共享内存，但仍保留在计算图中
#         mask = self.mask_correlated_samples(self.batch_size).to(device)
#         neg = neg.masked_select(mask).view(2 * self.batch_size, -1)

##########################################################
        z = torch.cat([z_i, z_j], dim=0)
        neg = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
        z_neg = neg.clone()#深拷贝 ，获得的新tensor和原来的数据不再共享内存，但仍保留在计算图中
        mask = self.mask_correlated_samples(self.batch_size).to(device)
        neg = neg.masked_select(mask).view(2 * self.batch_size, -1)

        z1 = torch.cat([z_i, z_j], dim=0)
        neg1 = torch.exp(torch.mm(z1, z1.t().contiguous()) / self.temperature)
        z_neg1 = neg1.clone()#深拷贝 ，获得的新tensor和原来的数据不再共享内存，但仍保留在计算图中
        mask = self.mask_correlated_samples(self.batch_size).to(device)
        neg1 = neg1.masked_select(mask).view(2 * self.batch_size, -1)
        
#############################################################             
        # pos score
        pos = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        # negative samples similarity scoring
        if estimator=='hard':
            N = self.batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / self.temperature))
       
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = ((- torch.log(pos / (pos + Ng) )).mean() + (- torch.sum(F.softmax(neg1.detach() / 0.04, dim=1) * F.log_softmax(neg / 0.1, dim=1), dim=1).mean()))/2
        #loss = (- torch.log(pos / Ng)).mean()########change!!!!!!!
        return loss  
    
    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(z_i,z_j,self.device,self.tau_plus,self.beta,self.estimator)
        #loss /= N
        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
       
    
    
#         c_i = F.normalize(c_i, dim=1)
#         c_j = F.normalize(c_j, dim=1)
#         c_i = c_i.t()
#         c_j = c_j.t()
#         N, C = c_i.shape 
#         device = c_i.device

#         indicator = (torch.ones((N, N), dtype=torch.float32) - torch.eye(N, dtype=torch.float32)).cuda()
#         pos = torch.exp(torch.matmul(c_i.view(N, 1, C), c_j.view(N, C, 1)).view(-1) / self.temperature)
#         s_ii = torch.exp(torch.matmul(c_i, c_i.T) / self.temperature)
#         s_ij = torch.exp(torch.matmul(c_j, c_j.T) / self.temperature)
#         S_i = torch.sum(s_ii * indicator + s_ij, dim=1)
#         loss_i = -1 * torch.sum(torch.log(pos / S_i)) / N

#         s_jj = torch.exp(torch.matmul(c_j, c_j.T) / self.temperature)
#         s_ji = s_ij.T
#         S_j = torch.sum(s_jj * indicator + s_ji, dim=1)
#         loss_j = -1 * torch.sum(torch.log(pos / S_j)) / N
    
#         loss = (loss_i + loss_j) / 2
        
        
        return loss + ne_loss
