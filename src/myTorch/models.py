#!/usr/bin/env python3
import torch
from torch import nn
from . import modelBase

class FCN(modelBase.hmumuModel):
    def __init__(self, params, *args, **kwargs):
        super(FCN, self).__init__(*args, **kwargs)
        number_of_variables = params["number_of_variables"]
        number_of_nodes = params.get("number_of_nodes", [64, 32, 16, 8])
        dropouts = params.get("dropouts", [1])
        self.linear_relu_stack = nn.Sequential()
        for i, nodes in enumerate(number_of_nodes):
            self.linear_relu_stack.add_module(f"linear_{i}", nn.Linear(params["number_of_variables"] if i==0 else number_of_nodes[i-1], nodes))
            self.linear_relu_stack.add_module(f"relu_{i}", nn.ReLU())
            self.linear_relu_stack.add_module(f"batchNorm_{i}", nn.BatchNorm1d(nodes))
            if i in dropouts:
                self.linear_relu_stack.add_module(f"dropout_{i}", nn.Dropout(p=0.1))
        self.linear_relu_stack.add_module(f"linear_{i+1}", nn.Linear(number_of_nodes[-1], 1))
        self.linear_relu_stack.add_module(f"sigmoid_{i+1}", nn.Sigmoid())

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class RNNGRU(modelBase.hmumuModel):
    def __init__(self, params, *args, **kwargs):
        super(FCN, self).__init__(*args, **kwargs)
        number_of_variables = params["number_of_variables"]
        number_of_nodes = params.get("number_of_nodes", [64, 32, 16, 8])
        dropouts = params.get("dropouts", [1])
        self.linear_relu_stack = nn.Sequential()
        for i, nodes in enumerate(number_of_nodes):
            self.linear_relu_stack.add_module(f"linear_{i}", nn.Linear(params["number_of_variables"] if i==0 else number_of_nodes[i-1], nodes))
            self.linear_relu_stack.add_module(f"relu_{i}", nn.ReLU())
            self.linear_relu_stack.add_module(f"batchNorm_{i}", nn.BatchNorm1d(nodes))
            if i in dropouts:
                self.linear_relu_stack.add_module(f"dropout_{i}", nn.Dropout(p=0.1))
        self.linear_relu_stack.add_module(f"linear_{i+1}", nn.Linear(number_of_nodes[-1], 1))
        self.linear_relu_stack.add_module(f"sigmoid_{i+1}", nn.Sigmoid())

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

