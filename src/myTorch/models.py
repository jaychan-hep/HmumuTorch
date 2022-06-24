#!/usr/bin/env python3
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class FCN(nn.Module):
    def __init__(self, number_of_variables, number_of_nodes=[64, 32, 16, 8], dropouts=[1]):
        super(FCN, self).__init__()
        self.linear_relu_stack = nn.Sequential()
        for i, nodes in enumerate(number_of_nodes):
            self.linear_relu_stack.add_module(f"linear_{i}", nn.Linear(number_of_variables if i==0 else number_of_nodes[i-1], nodes))
            self.linear_relu_stack.add_module(f"relu_{i}", nn.ReLU())
            self.linear_relu_stack.add_module(f"batchNorm_{i}", nn.BatchNorm1d(nodes))
            if i in dropouts:
                self.linear_relu_stack.add_module(f"dropout_{i}", nn.Dropout(p=0.1))
        self.linear_relu_stack.add_module(f"linear_{i+1}", nn.Linear(number_of_nodes[-1], 1))

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits 
