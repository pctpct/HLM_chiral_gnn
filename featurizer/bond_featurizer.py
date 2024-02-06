# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : bond_featurizer.py
@Project  : ChiralityGNN_HLM
@Time     : 2023/8/29 11:22
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/8/29 11:22        1.0             None
"""

import numpy as np
bondTypes = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

def one_hot_embedding(value, options):
    embedding = [0]*(len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding
def getEdgeFeatures(list_rdkit_bonds):
    F_e = (len(bondTypes)+1) + 2 + (6+1) # 14

    edge_features = np.zeros((len(list_rdkit_bonds)*2, F_e))
    for edge_index, edge in enumerate(list_rdkit_bonds):
        features = one_hot_embedding(str(edge.GetBondType()), bondTypes) # dim=4+1
        features += [int(edge.GetIsConjugated())] # dim=1
        features += [int(edge.IsInRing())] # dim=1
        features += one_hot_embedding(edge.GetStereo(), list(range(6))) #dim=6+1

        # Encode both directed edges to get undirected edge
        edge_features[2*edge_index: 2*edge_index+2, :] = features

    return np.array(edge_features, dtype = np.float32)