import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2/(fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return weights.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2/fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return weights.tolist()
        


    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str
    ) -> List[float]:

        torch.manual_seed(0)
        weights = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            if init_type == "xavier":
                std = math.sqrt(2 / (in_dim + out_dim))
            elif init_type == "kaiming":
                std = math.sqrt(2 / in_dim)
            else:
                std = 1.0
            W = torch.randn(out_dim, in_dim) * std
            weights.append(W)

        x = torch.randn(1, input_dim) 
        stds = []
        for W in weights:
            x = torch.relu(x @ W.T)  
            stds.append(round(x.std().item(), 2))
        return stds
