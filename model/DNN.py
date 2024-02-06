# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : DNN.py
@Project  : ChiralityGNN_HLM_3
@Time     : 2023/12/13 19:42
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/12/13 19:42        1.0             None
"""
from torch import nn
class DNN_network(nn.Module):
    def __init__(self,input_dim,input_drop_rate,num_hidden_layer,num_hidden_units_list,hidden_drop_rate):
        super().__init__()
        assert num_hidden_layer == len(num_hidden_units_list)

        # input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, num_hidden_units_list[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=input_drop_rate),
            nn.BatchNorm1d(num_hidden_units_list[0])
        )

        # hidden layer
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layer-1):
            self.hidden_layers.append(self.construct_hidden_layer(\
                num_hidden_units_list[i],num_hidden_units_list[i+1],hidden_drop_rate))

        # output layers
        self.output_layer = nn.Linear(num_hidden_units_list[-1],1)


    def forward(self,X):
        x = self.input_layer(X)
        for layer in self.hidden_layers:
            x = layer(x)

        out = self.output_layer(x)
        return out

    def construct_hidden_layer(self,in_dim,out_dim,hidden_drop_rate):
        return nn.Sequential(
            nn.Linear(in_dim,out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=hidden_drop_rate),
            nn.BatchNorm1d(out_dim)
        )