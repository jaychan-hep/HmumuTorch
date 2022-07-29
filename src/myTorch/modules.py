#!/usr/bin/env python3
import torch
from torch import nn

class FCN(nn.Module):
    def __init__(self, number_of_inputs, number_of_nodes=[64, 32, 16, 8], dropouts=[1], return_final_score=False, sigmoid=False):
        super(FCN, self).__init__()
        self.linear_relu_stack = nn.Sequential()
        for i, nodes in enumerate(number_of_nodes):
            self.linear_relu_stack.add_module(f"linear_{i}", nn.Linear(number_of_inputs if i==0 else number_of_nodes[i-1], nodes))
            self.linear_relu_stack.add_module(f"relu_{i}", nn.ReLU())
            self.linear_relu_stack.add_module(f"batchNorm_{i}", nn.BatchNorm1d(nodes))
            if i in dropouts:
                self.linear_relu_stack.add_module(f"dropout_{i}", nn.Dropout(p=0.1))
        if return_final_score:
            self.linear_relu_stack.add_module(f"linear_{i+1}", nn.Linear(number_of_nodes[-1], 1))
        if sigmoid:
            self.linear_relu_stack.add_module(f"sigmoid_{i+1}", nn.Sigmoid())

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class GRU(nn.Module):
    def __init__(self, return_hidden_states=False, return_series=False, *args, **kwargs):
        super(GRU, self).__init__()
        self.return_hidden_states = return_hidden_states
        self.return_series = return_series
        self.GRU = nn.GRU(*args, **kwargs)

    def forward(self, x):
        output, hn = self.GRU(x)
        if not self.return_series: output = output[:,-1]
        if self.return_hidden_states:
            return output, hn
        return output

class sequential_GRU(nn.Module):
    def __init__(self, number_of_inputs_per_object, number_of_objects, number_of_GRUnodes=[16, 12], GRUdropouts=[1]):
        super(sequential_GRU, self).__init__()
        self.gru_stack = nn.Sequential()
        for i, nodes in enumerate(number_of_GRUnodes):
            self.gru_stack.add_module(f"gru_{i}", GRU(False, False if i==len(number_of_GRUnodes)-1 else True, number_of_inputs_per_object if i==0 else number_of_GRUnodes[i-1], nodes, 1))
            self.gru_stack.add_module(f"batchNorm_{i}", nn.BatchNorm1d(nodes if i==len(number_of_GRUnodes)-1 else number_of_objects))
            if i in GRUdropouts:
                self.gru_stack.add_module(f"dropout_{i}", nn.Dropout(p=0.1))

    def forward(self, x):
        logits = self.gru_stack(x)
        return logits
