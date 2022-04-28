# Copyright (c) 2022 Dai HBG

"""
定义训练模型
2022-04-16
- init
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

import sys
sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/QBG')
sys.path.append('C:/Users/HBG/Desktop/Repositories/Daily-Frequency-Quant/QBG')

from Model.MyDeepModel import Gate


class GateNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.7, alpha: float = 0.2):
        super(GateNet, self).__init__()
        self.input_dim = input_dim
        self.output_Dim = output_dim

        self.Dense1 = nn.Linear(input_dim, 16)
        self.Dense2 = nn.Linear(16, 8)
        self.Dense3 = nn.Linear(8, 8)
        self.Dense4 = nn.Linear(8, output_dim)

        self.gate0 = Gate(input_dim)
        self.gate1 = Gate(16)
        self.gate2 = Gate(8)
        self.gate3 = Gate(8)

        self.act = nn.LeakyReLU(alpha)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.Dense1(self.gate0(self.dropout(x))))
        x = self.act(self.Dense2(self.gate1(self.dropout(x))))
        x = x + self.dropout(self.gate3(self.act(self.Dense3(self.gate2(self.dropout(x))))))
        x = self.Dense4(x)
        return x


class NN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.7, alpha: float = 0.2):
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.output_Dim = output_dim

        self.Dense1 = nn.Linear(input_dim, 4)
        self.Dense2 = nn.Linear(4, 4)
        self.Dense4 = nn.Linear(4, output_dim)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.act(self.Dense1(x))
        x = self.act(self.Dense2(x))
        return self.Dense4(x)
