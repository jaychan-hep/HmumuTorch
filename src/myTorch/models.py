#!/usr/bin/env python3
import torch
from torch import nn
from . import modelBase
from . import modules

class FCN(modelBase.hmumuModel):
    def __init__(self, number_of_variables, number_of_nodes=[64, 32, 16, 8], dropouts=[1], *args, **kwargs):
        super(FCN, self).__init__(*args, **kwargs)
        self.FCN = modules.FCN(number_of_variables, number_of_nodes, dropouts, return_final_score=True, sigmoid=True)

    def forward(self, x):
        logits = self.FCN(x)
        return logits

class RNNGRU(modelBase.hmumuModel):
    def __init__(self, number_of_inputs_per_object, number_of_objects, number_of_other_variables, number_of_GRUnodes=[16, 12], GRUdropouts=[1], number_of_postGRUnodes=[16], postGRUdropouts=list(), number_of_nodes=[64, 32, 16], dropouts=list(), *args, **kwargs):
        super(RNNGRU, self).__init__(*args, **kwargs)

        self.GRU = modules.sequential_GRU(number_of_inputs_per_object, number_of_objects, number_of_GRUnodes, GRUdropouts)
        self.post_GRU = modules.FCN(number_of_GRUnodes[-1], number_of_postGRUnodes, postGRUdropouts, return_final_score=False, sigmoid=False)
        self.FCN = modules.FCN(number_of_postGRUnodes[-1]+number_of_other_variables, number_of_nodes, dropouts, return_final_score=True, sigmoid=True)

    def forward(self, x):
        objects, others = x
        objects = self.GRU(objects)
        objects = self.post_GRU(objects)
        x = torch.cat([objects, others], dim=1)
        logits = self.FCN(x)
        return logits

