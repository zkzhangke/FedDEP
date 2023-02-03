import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch_geometric.loader import NeighborSampler

loss=torch.zeros(3,2)
crt = nn.MSELoss()
input = torch.randn(3,5, requires_grad=True)
h = nn.Linear(5,10)(input)
out=h.view(-1,2,5)
target = torch.randn(3,2, 5)
for k,i,t in zip(range(3),out,target):
    for j,ii,tt in zip(range(2),i,t):
        loss[k][j] = crt(ii, tt)
print(loss)
loss=loss.mean(1)
print(loss)
loss=loss.mean(0)
print(loss)
grad=torch.autograd.grad(loss, input, retain_graph=True)
grad1=torch.autograd.grad(crt(out,target), input, retain_graph=True)
print(grad)
print(grad1)