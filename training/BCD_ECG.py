#Definition for singly-linked list.

from __future__ import print_function, division
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

# import torchvision
# from torchvision import datasets, models, transforms, utils
# from torch.utils.data import Dataset, DataLoader
import time
import os
import copy
from sklearn import preprocessing
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

print("PyTorch Version:", torch.__version__)
# print("Torchvision Version:", torchvision.__version__)
print("GPU is available?", torch.cuda.is_available())

dtype = torch.float
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

with open('patient_list.txt', 'rb') as f:
    patient_list = pickle.load(f)
paitent_num = len(patient_list)
x_dim = len(patient_list[0]['x'][0])
y_dim = max(patient_list[0]['y'])+1
print("Keys :", patient_list[0].keys())
print("Patient Num =", paitent_num)
print("x dimension =", x_dim)
print("y dimension =", y_dim)

# Pretrain Dataset
pretrain_bound = int(paitent_num * 0.05) # Pretrain:Main = 1:4
pre_x_y = np.empty([0,61])
for patient_idx in range(0, paitent_num):
    patient = patient_list[patient_idx]
    temp_x = preprocessing.MinMaxScaler().fit_transform(np.array(patient['x']))  
    x_y = np.append(temp_x, np.array(patient['y'])[:,np.newaxis], axis=1)
    pre_x_y = np.concatenate((pre_x_y, x_y))

np.random.shuffle(pre_x_y)

bound = int(len(pre_x_y)*0.8)  # Training:Validation = 4:1

pre_train_x = torch.from_numpy(pre_x_y[:bound,:60].T).float()
pre_train_x = pre_train_x.to(device=device)

pre_train_y = torch.from_numpy(pre_x_y[:bound,60]).long()


pre_train_y_onehot = torch.zeros(len(pre_train_y), 2).scatter_(1, torch.reshape(pre_train_y, (len(pre_train_y), 1)), 1)
pre_train_y = pre_train_y.to(device=device)
pre_train_y_onehot = torch.t(pre_train_y_onehot).to(device=device)
pre_valid_x = torch.from_numpy(pre_x_y[bound:,:60].T).float()
pre_valid_x = pre_valid_x.to(device=device)

pre_valid_y = torch.from_numpy(pre_x_y[bound:,60]).long()
pre_valid_y = pre_valid_y.to(device=device)

main_dataset = []
external_info_list = []

for patient_idx in range(0, paitent_num):
    dataset_dict = {}
    patient = patient_list[patient_idx]
    temp_x = preprocessing.MinMaxScaler().fit_transform(np.array(patient['x']))  
    x_y = np.append(temp_x, np.array(patient['y'])[:,np.newaxis], axis=1)
    np.random.shuffle(x_y)
    bound_1 , bound_2 = int(len(x_y)*0.6), int(len(x_y)*0.8) # Training:Validation:test = 3:1:1
    dataset_dict['train_x'] = torch.from_numpy(x_y[:bound_2,:60].T).float().to(device=device)
    train_y = torch.from_numpy(x_y[:bound_2,60]).long()
    train_y_onehot = torch.zeros(len(train_y), 2).scatter_(1, torch.reshape(train_y, (len(train_y), 1)), 1)
    dataset_dict['train_y'] = train_y.to(device=device)
    dataset_dict['train_y_onehot'] = torch.t(train_y_onehot).to(device=device)
    
#     dataset_dict['valid_x'] = torch.from_numpy(x_y[bound_1:bound_2,:60].T).float().to(device=device)
#     dataset_dict['valid_y'] =torch.from_numpy(x_y[bound_1:bound_2,60]).long().to(device=device)
    dataset_dict['valid_x'] = torch.from_numpy(x_y[bound_2:,:60].T).float().to(device=device)
    dataset_dict['valid_y'] =torch.from_numpy(x_y[bound_2:,60]).long().to(device=device)
    
    dataset_dict['test_x'] = torch.from_numpy(x_y[bound_2:,:60].T).float().to(device=device)
    dataset_dict['test_y'] =torch.from_numpy(x_y[bound_2:,60]).long().to(device=device)
    
    main_dataset.append(dataset_dict)
    external_info_list.append([patient['age'], patient['sex'], patient['bmi']])

external_info_arr = preprocessing.MinMaxScaler().fit_transform(np.array(external_info_list))
distance_matrix = squareform(pdist(external_info_arr, metric='cosine'))
similarity_matrix = 1-distance_matrix

# Hyperparameters
gamma = 2
rho = 1
alpha = 10

def weighted_average_parameters_from_neighbor(similarity_dict, agent_dict, layer_idx):
    agent_idx = similarity_dict.keys()[0]
    sum_Wi = torch.zeros(eval('agent_dict["agent_{}"].W{}'.format(str(agent_idx), layer_idx)).shape)
    sum_bi = torch.zeros(eval('agent_dict["agent_{}"].b{}'.format(str(agent_idx), layer_idx)).shape)
    tot_sim = sum(similarity_dict.values())
    for agent_idx, similiarity in similarity_dict.items():
        sum_Wi += eval('agent_dict["agent_{}"].W{}'.format(str(agent_idx), layer_idx))*similiarity
        sum_bi += eval('agent_dict["agent_{}"].b{}'.format(str(agent_idx), layer_idx))*similiarity
    return sum_Wi/tot_sim, sum_bi/tot_sim

def updateWb(U, V, W, b, alpha, rho, similarity_dict=None, agent_dict=None, layer_idx=None, miu=None): #add extra loss here  -- Rex
    I = torch.eye(V.size()[0], device=device)
    W_star = torch.mm(alpha*W+rho*torch.mm(U-b.repeat(1,U.size()[1]),torch.t(V)),torch.inverse(alpha*I+rho*(torch.mm(V,torch.t(V)))))
    b_star = (alpha*b+rho*torch.sum(U-torch.mm(W,V), dim=1).reshape(b.size()))/(rho*V.size()[1]+alpha)
    if similarity_dict and agent_dict:
        W_neighbor, b_neighbor = weighted_average_parameter_from_neighbor(similarity_dict, agent_dict, layer_idx)
        W_star = (W_star + miu * W_neighbor)/(1+miu)
        b_star = (b_star + miu * b_neighbor)/(1+miu)
    return W_star, b_star

def updateV(U1,U2,W,b,rho,gamma): 
    I = torch.eye(W.size()[1], device=device)
    U1 = nn.ReLU()(U1)
    Vstar = torch.mm(torch.inverse(rho*(torch.mm(torch.t(W),W))+gamma*I), rho*torch.mm(torch.t(W),U2-b.repeat(1,U2.size()[1]))+gamma*U1)
    return Vstar

def relu_prox(a, b, gamma, d, N):
    val = torch.empty(d,N, device=device)
    x = (a+gamma*b)/(1+gamma)
    y = torch.min(b,torch.zeros(d,N, device=device))

    val = torch.where(a+gamma*b < 0, y, torch.zeros(d,N, device=device))
    val = torch.where(((a+gamma*b >= 0) & (b >=0)) | ((a*(gamma-np.sqrt(gamma*(gamma+1))) <= gamma*b) & (b < 0)), x, val)
    val = torch.where((-a <= gamma*b) & (gamma*b <= a*(gamma-np.sqrt(gamma*(gamma+1)))), b, val)
    return val

def block_update(Wn, bn, Wn_1, bn_1, Vn, Un, Vn_1, Un_1, Vn_2, dn_1, dim,  
                 similarity_dict=None, agent_dict=None, layer_idx=None, miu=None,
                 alpha = alpha, gamma = gamma, rho = rho):
    # update W(n) and b(n)
    Wn, bn = updateWb(Un, Vn_1, Wn, bn, alpha, rho, similarity_dict, agent_dict, layer_idx, miu)
    # update V(n-1)
    Vn_1 = updateV(Un_1, Un, Wn, bn, rho, gamma)
    # update U(n-1)
    Un_1 = relu_prox(Vn_1, (rho*torch.addmm(bn_1.repeat(1,dim), Wn_1, Vn_2) + \
                            alpha*Un_1)/(rho + alpha), (rho + alpha)/gamma, dn_1, dim)
    return Wn, bn, Vn_1, Un_1

def initialize(dim_in, dim_out):
    W = 0.01*torch.randn(dim_out, dim_in, device=device)
    b = 0.1*torch.ones(dim_out, 1, device=device)
    return W, b

def feed_forward(weight, bias, activation):
    _, N = activation.size()
    U = torch.addmm(bias.repeat(1, N), weight, activation)
    V = nn.ReLU()(U)
    return U, V

def Multifilter_Conv1D(x, kernel_group):
    def FixedConv1D(x, kernel):
        x = x.T.unsqueeze(1).float()
        weight = torch.tensor(kernel).expand(1, 1,-1).float()
        weight = nn.Parameter(data=weight, requires_grad=False).to(device=device)
        result = nn.functional.conv1d(x, weight, padding=0)
        return result.squeeze(1).T
    
    output = FixedConv1D(x, kernel_group[0])
    for kernel in kernel_group[1:]:
        output = torch.cat((output, FixedConv1D(x, kernel)), 0)
        
    return output
    
    
# BCD 6 L
seed = 16
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.manual_seed(seed)
    
# Model
class MLP_BCD_6_layer():
    def __init__(self, dim_list, x, y, similarity_dict=None):
        super(MLP_BCD_6_layer, self).__init__()
        self.x_original, self.y = x, y
        self.x = Multifilter_Conv1D(x, kernel_group)
        self.N = self.x.size()[1]
        self.dim_1, self.dim_2, self.dim_3, self.dim_4, self.dim_5, self.dim_6, self.dim_7 = dim_list
        if self.dim_1 == -1:
            self.dim_1 = self.x.size()[0]
        if self.dim_2 == -1:
            self.dim_2 = self.x.size()[0]
        self.sim_dict = similarity_dict
        self.miu = 1
        
        # Layer 1
        self.W1, self.b1 = initialize(self.dim_1, self.dim_2)
        self.U1, self.V1 = feed_forward(self.W1, self.b1, self.x)
        
        # Layer 2
        self.W2, self.b2 = initialize(self.dim_2, self.dim_3)
        self.U2, self.V2 = feed_forward(self.W2, self.b2, self.V1)
        
        # Layer 3
        self.W3, self.b3 = initialize(self.dim_3, self.dim_4)
        self.U3, self.V3 = feed_forward(self.W3, self.b3, self.V2)
        
        # Layer 4
        self.W4, self.b4 = initialize(self.dim_4, self.dim_5)
        self.U4, self.V4 = feed_forward(self.W4, self.b4, self.V3)
        
        # Layer 5
        self.W5, self.b5 = initialize(self.dim_5, self.dim_6)
        self.U5, self.V5 = feed_forward(self.W5, self.b5, self.V4)
        
        # Layer 6
        self.W6, self.b6 = initialize(self.dim_6, self.dim_7)
        self.U6 = torch.addmm(self.b6.repeat(1, self.N), self.W6, self.V5)
        self.V6 = self.U6
        
    
    def forward(self, **kwargs):
        x = kwargs.get('x', self.x_original)
        x = Multifilter_Conv1D(x, kernel_group)
        _, x = feed_forward(self.W1, self.b1, x)
        _, x = feed_forward(self.W2, self.b2, x)
        _, x = feed_forward(self.W3, self.b3, x)
        _, x = feed_forward(self.W4, self.b4, x)
        _, x = feed_forward(self.W5, self.b5, x)
        x = torch.argmax(torch.addmm(self.b6.repeat(1, x.size()[1]), self.W6, x), dim=0)
        return x
    
    def update(self, agent_dict=None):
        self.V6 = (self.y + gamma*self.U6 + alpha*self.V6)/(1 + gamma + alpha)
        self.U6 = (gamma*self.V6 + rho*(torch.mm(self.W6, self.V5) + self.b6.repeat(1, self.N)))/(gamma + rho)
        self.W6, self.b6, self.V5, self.U5 = block_update(self.W6, self.b6, self.W5, self.b5, self.V6, self.U6, self.V5, self.U5, self.V4, self.dim_6, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=6, miu=self.miu)
        self.W5, self.b5, self.V4, self.U4 = block_update(self.W5, self.b5, self.W4, self.b4, self.V5, self.U5, self.V4, self.U4, self.V3, self.dim_5, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=5, miu=self.miu)
        self.W4, self.b4, self.V3, self.U3 = block_update(self.W4, self.b4, self.W3, self.b3, self.V4, self.U4, self.V3, self.U3, self.V2, self.dim_4, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=4, miu=self.miu)
        self.W3, self.b3, self.V2, self.U2 = block_update(self.W3, self.b3, self.W2, self.b2, self.V3, self.U3, self.V2, self.U2, self.V1, self.dim_3, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=3, miu=self.miu)
        self.W2, self.b2, self.V1, self.U1 = block_update(self.W2, self.b2, self.W1, self.b1, self.V2, self.U2, self.V1, self.U1, self.x, self.dim_2, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=2, miu=self.miu)
        self.W1, self.b1 = updateWb(self.U1, self.x, self.W1, self.b1, alpha, rho, similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=1, miu=self.miu)
        
    def loss(self, **kwargs):
        y = kwargs.get('y', self.y)
        x = kwargs.get('x', self.x)
        def compute_loss(weight, bias, activation, preactivation, rho = rho):
            return rho/2*torch.pow(torch.dist(torch.addmm(bias.repeat(1,activation.size()[1]), weight, activation), preactivation, 2), 2).cpu().numpy()
        sq_loss = gamma/2*torch.pow(torch.dist(self.V6, y, 2),2).cpu().numpy()
        tot_loss = sq_loss \
        + compute_loss(self.W1, self.b1, x, self.U1) \
        + compute_loss(self.W2, self.b2, self.V1, self.U2) \
        + compute_loss(self.W3, self.b3, self.V2, self.U3) \
        + compute_loss(self.W4, self.b4, self.V3, self.U4)
        + compute_loss(self.W5, self.b5, self.V4, self.U5)
        + compute_loss(self.W6, self.b6, self.V5, self.U6)
        return sq_loss, tot_loss
        
        
class MLP_BCD_10_layer():
    def __init__(self, dim_list, x, y, similarity_dict=None):
        super(MLP_BCD_10_layer, self).__init__()
        self.x_original, self.y = x, y
        self.x = Multifilter_Conv1D(x, kernel_group)
        self.N = self.x.size()[1]
        self.dim_1, self.dim_2, self.dim_3, self.dim_4, self.dim_5, self.dim_6, self.dim_7,\
        self.dim_8, self.dim_9, self.dim_10 = dim_list
        if self.dim_1 == -1:
            self.dim_1 = self.x.size()[0]
        if self.dim_2 == -1:
            self.dim_2 = self.x.size()[0]
        self.sim_dict = similarity_dict
        self.miu = 1
        
        # Layer 1
        self.W1, self.b1 = initialize(self.dim_1, self.dim_2)
        self.U1, self.V1 = feed_forward(self.W1, self.b1, self.x)
        
        # Layer 2
        self.W2, self.b2 = initialize(self.dim_2, self.dim_3)
        self.U2, self.V2 = feed_forward(self.W2, self.b2, self.V1)
        
        # Layer 3
        self.W3, self.b3 = initialize(self.dim_3, self.dim_4)
        self.U3, self.V3 = feed_forward(self.W3, self.b3, self.V2)
        
        # Layer 4
        self.W4, self.b4 = initialize(self.dim_4, self.dim_5)
        self.U4, self.V4 = feed_forward(self.W4, self.b4, self.V3)
        
        # Layer 5
        self.W5, self.b5 = initialize(self.dim_5, self.dim_6)
        self.U5, self.V5 = feed_forward(self.W5, self.b5, self.V4)
        
        # Layer 6
        self.W6, self.b6 = initialize(self.dim_6, self.dim_9)
        self.U6, self.V6 = feed_forward(self.W6, self.b6, self.V5)
        
        # Layer 7
        self.W7, self.b7 = initialize(self.dim_7, self.dim_8)
        self.U7, self.V7 = feed_forward(self.W7, self.b7, self.V6)
        
        # Layer 8
        self.W8, self.b8 = initialize(self.dim_8, self.dim_9)
        self.U8, self.V8 = feed_forward(self.W8, self.b8, self.V7)
        
        # Layer 9
        self.W9, self.b9 = initialize(self.dim_9, self.dim_10)
        self.U9 = torch.addmm(self.b9.repeat(1, self.N), self.W9, self.V8)
        self.V9 = self.U9
        
    
    def forward(self, **kwargs):
        x = kwargs.get('x', self.x_original)
        x = Multifilter_Conv1D(x, kernel_group)
        _, x = feed_forward(self.W1, self.b1, x)
        _, x = feed_forward(self.W2, self.b2, x)
        _, x = feed_forward(self.W3, self.b3, x)
        _, x = feed_forward(self.W4, self.b4, x)
        _, x = feed_forward(self.W5, self.b5, x)
        _, x = feed_forward(self.W6, self.b6, x)
        _, x = feed_forward(self.W7, self.b7, x)
        _, x = feed_forward(self.W8, self.b8, x)
        x = torch.argmax(torch.addmm(self.b9.repeat(1, x.size()[1]), self.W9, x), dim=0)
        return x
    
    def update(self, agent_dict=None):
        self.V9 = (self.y + gamma*self.U9 + alpha*self.V9)/(1 + gamma + alpha)
        self.U9 = (gamma*self.V9 + rho*(torch.mm(self.W9, self.V8) + self.b9.repeat(1, self.N)))/(gamma + rho)
        self.W9, self.b9, self.V8, self.U8 = block_update(self.W9, self.b9, self.W8, self.b8, self.V9, self.U9, self.V8, self.U8, self.V7, self.dim_9, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=9, miu=self.miu)
        self.W8, self.b8, self.V7, self.U7 = block_update(self.W8, self.b8, self.W7, self.b7, self.V8, self.U8, self.V7, self.U7, self.V6, self.dim_8, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=8, miu=self.miu)
        self.W7, self.b7, self.V6, self.U6 = block_update(self.W7, self.b7, self.W6, self.b6, self.V7, self.U7, self.V6, self.U6, self.V5, self.dim_7, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=7, miu=self.miu)
        self.W6, self.b6, self.V5, self.U5 = block_update(self.W6, self.b6, self.W5, self.b5, self.V6, self.U6, self.V5, self.U5, self.V4, self.dim_6, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=6, miu=self.miu)
        self.W5, self.b5, self.V4, self.U4 = block_update(self.W5, self.b5, self.W4, self.b4, self.V5, self.U5, self.V4, self.U4, self.V3, self.dim_5, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=5, miu=self.miu)
        self.W4, self.b4, self.V3, self.U3 = block_update(self.W4, self.b4, self.W3, self.b3, self.V4, self.U4, self.V3, self.U3, self.V2, self.dim_4, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=4, miu=self.miu)
        self.W3, self.b3, self.V2, self.U2 = block_update(self.W3, self.b3, self.W2, self.b2, self.V3, self.U3, self.V2, self.U2, self.V1, self.dim_3, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=3, miu=self.miu)
        self.W2, self.b2, self.V1, self.U1 = block_update(self.W2, self.b2, self.W1, self.b1, self.V2, self.U2, self.V1, self.U1, self.x, self.dim_2, self.N,\
                                                          similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=2, miu=self.miu)
        self.W1, self.b1 = updateWb(self.U1, self.x, self.W1, self.b1, alpha, rho, similarity_dict=self.sim_dict, agent_dict=agent_dict, layer_idx=1, miu=self.miu)
        
    def loss(self, **kwargs):
        y = kwargs.get('y', self.y)
        x = kwargs.get('x', self.x)
        def compute_loss(weight, bias, activation, preactivation, rho = rho):
            return rho/2*torch.pow(torch.dist(torch.addmm(bias.repeat(1,activation.size()[1]), weight, activation), preactivation, 2), 2).cpu().numpy()
        sq_loss = gamma/2*torch.pow(torch.dist(self.V9, y, 2),2).cpu().numpy()
        tot_loss = sq_loss \
        + compute_loss(self.W1, self.b1, x, self.U1) \
        + compute_loss(self.W2, self.b2, self.V1, self.U2) \
        + compute_loss(self.W3, self.b3, self.V2, self.U3) \
        + compute_loss(self.W4, self.b4, self.V3, self.U4)\
        + compute_loss(self.W5, self.b5, self.V4, self.U5)\
        + compute_loss(self.W6, self.b6, self.V5, self.U6)\
        + compute_loss(self.W7, self.b7, self.V6, self.U7)\
        + compute_loss(self.W8, self.b8, self.V7, self.U8)\
        + compute_loss(self.W9, self.b9, self.V8, self.U9)
        return sq_loss, tot_loss
        
        
iter_num = 200
loss1 = np.zeros(shape=(iter_num))
loss2 = np.zeros(shape=(iter_num))
accuracy_train = np.zeros(shape=(iter_num))
accuracy_test = np.zeros(shape=(iter_num))
valid_metric = np.zeros(shape=(iter_num,4))

                                        
kernel_group = [np.array([1, 1, 1])/3, 
                np.array([1, -2, 1]),  
                np.array([-1, 3, -1]), 
                np.array([-2, 0, 2]), 
             
                np.array([0.02,0.2,0.98,1,1,1,0.98,0.2,0.02])/5.4,
                np.array([0.1,0.4,1,1,1,1,1,1,1,0.4,0.1])/7] 




kernel_group = [np.array([1, 1, 1])/3,
                np.array([-1, 3, -1]),
#                 [1],
                np.array([1, -2, 1]),
                np.array([-2, 0, 2]),
               np.array([0.02,0.2,0.98,1,1,1,0.98,0.2,0.02])/5.4,
#                np.array([0.1,0.4,1,1,1,1,0.4,0.1])/5,
               np.array([0.1,0.4,1,1,1,1,1,1,0.4,0.1])/7,
#                np.array([0.1,0.4,1,1,1,1,1,1,1,1,0.4,0.1])/9
               ]

    
iter_num = 200
agent_num = len(main_dataset)
neighbor_num = 60
loss1 = np.zeros(shape=(iter_num,agent_num))
loss2 = np.zeros(shape=(iter_num,agent_num))
valid_metric = np.zeros(shape=(iter_num,agent_num,4))
# accuracy_test = np.zeros(shape=(iter_num,agent_num))


# Similarity Matrix
similarity_dict_list = []
for key in range(similarity_matrix.shape[0]):
    similarity_list = similarity_matrix[key]
    similarity_dict = {}
    for agent_idx in similarity_list.argsort()[agent_num-neighbor_num:]:
        if key == agent_idx:
            continue
        similarity_dict[agent_idx] = similarity_list[agent_idx]
    similarity_dict_list.append(similarity_dict)

    

# Agents Initialization
agent_dict = dict()
for agent_idx in range(agent_num):
    agent_name = "agent_"+str(agent_idx)
    agent_model = MLP_BCD_6_layer([-1,8000, 8000, 8000, 8000, 8000, y_dim],
                                  main_dataset[agent_idx]['train_x'],
                                  main_dataset[agent_idx]['train_y_onehot'],
                                  similarity_dict_list[agent_idx]
                                  )
    agent_dict.update({agent_name : agent_model})

# Iterations
# print('Train on', N, 'samples, validate on', N_test, 'samples')
for iteration_idx in range(iter_num):
    print('Epoch', iteration_idx + 1, '/', iter_num)
    for agent_idx in range(agent_num):
        print('    Agent_{}:'.format(agent_idx), end=' ')
        start = time.time()
        
        # BCD update
        agent_dict['agent_{}'.format(agent_idx)].update()
        
        # compute training loss
        loss1[iteration_idx, agent_idx], loss2[iteration_idx, agent_idx] = agent_dict['agent_{}'.format(agent_idx)].loss()

        # compute validation accuracy
        test_output = agent_dict['agent_{}'.format(agent_idx)].forward(x=main_dataset[agent_idx]['valid_x'])
        TP = np.mean((test_output*main_dataset[agent_idx]['valid_y'] == torch.ones(test_output.size())).cpu().numpy())
        FP = np.mean((test_output*((main_dataset[agent_idx]['valid_y']-1)*-1) == torch.ones(test_output.size())).cpu().numpy())
        FN = np.mean(((test_output-1)*-1*main_dataset[agent_idx]['valid_y'] == torch.ones(test_output.size())).cpu().numpy())
        TN = np.mean(((test_output-1)*-1*((main_dataset[agent_idx]['valid_y']-1)*-1) == torch.ones(test_output.size())).cpu().numpy())
        valid_metric[iteration_idx, agent_idx] = [TP, FP, FN, TN]
    
        # training time
        stop = time.time()
        duration = round(stop - start, 3)
    
        # print results
        print('acc={} rec={} pre={} spe={}'.format(round(TP+TN,4), round(TP/(TP+FN),4), round(TP/(TP+FP),4), round(TN/(TN+FP),4)))
        with open('result-Dec-Com-N60.txt', 'wb') as f:
            pickle.dump([duration, loss1, loss2, valid_metric], f)
    
