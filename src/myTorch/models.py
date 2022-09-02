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

class DeepSets(modelBase.hmumuModel):
    def __init__(self, number_of_inputs_per_object, number_of_other_variables, number_of_Phinodes=[16, 16, 16], Phidropouts=list(), number_of_nodes=[64, 32, 16], dropouts=list(), *args, **kwargs):
        super(DeepSets, self).__init__(*args, **kwargs)

        self.Phi = modules.timeDistributed(modules.FCN(number_of_inputs_per_object, number_of_Phinodes, Phidropouts, return_final_score=False, sigmoid=False))
        self.FCN = modules.FCN(number_of_Phinodes[-1]+number_of_other_variables, number_of_nodes, dropouts, return_final_score=True, sigmoid=True)

    def forward(self, x):
        objects, others = x
        objects = self.Phi(objects)
        objects = torch.sum(objects, dim=1)
        x = torch.cat([objects, others], dim=1)
        logits = self.FCN(x)
        return logits

class SelfAttention(modelBase.hmumuModel):
    def __init__(self, number_of_inputs_per_object, number_of_other_variables, number_of_Phinodes=[16, 16, 16], Phidropouts=list(), number_of_nodes=[64, 32, 16], dropouts=list(), *args, **kwargs):
        super(SelfAttention, self).__init__(*args, **kwargs)

        self.Phi1 = modules.timeDistributed(modules.FCN(number_of_inputs_per_object, number_of_Phinodes, Phidropouts, return_final_score=False, sigmoid=False))
        self.query = modules.timeDistributed(nn.Linear(number_of_Phinodes[-1], number_of_Phinodes[-1]))
        self.key = modules.timeDistributed(nn.Linear(number_of_Phinodes[-1], number_of_Phinodes[-1]))
        self.value = modules.timeDistributed(nn.Linear(number_of_Phinodes[-1], number_of_Phinodes[-1]))
        self.att = nn.MultiheadAttention(number_of_Phinodes[-1], 1)
        self.layer_norm1 = modules.timeDistributed(nn.LayerNorm(number_of_Phinodes[-1]))
        self.Phi2 = modules.timeDistributed(modules.FCN(number_of_Phinodes[-1], number_of_Phinodes, Phidropouts, return_final_score=False, sigmoid=False))
        self.layer_norm2 = modules.timeDistributed(nn.LayerNorm(number_of_Phinodes[-1]))
        self.FCN = modules.FCN(number_of_Phinodes[-1]+number_of_other_variables, number_of_nodes, dropouts, return_final_score=True, sigmoid=True)

    def forward(self, x):
        objects, others = x
        objects = self.Phi1(objects)
        query = self.query(objects)
        key = self.key(objects)
        value = self.value(objects)
        objects_attention, _ = self.att(query, key, value)
        objects = torch.add(objects, objects_attention)
        objects = self.layer_norm1(objects)
        objects_dense = self.Phi2(objects)
        objects = torch.add(objects, objects_dense)
        objects = self.layer_norm1(objects)
        objects = torch.sum(objects, dim=1)
        x = torch.cat([objects, others], dim=1)
        logits = self.FCN(x)
        return logits

