# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : atom_featurizer.py
@Project  : ChiralityGNN_HLM
@Time     : 2023/8/29 11:21
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/8/29 11:21        1.0             None
"""
import numpy as np
from rdkit import Chem


def one_hot_embedding(value, options):
    embedding = [0]*(len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding


atomTypes = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
formalCharge = [-1, -2, 1, 2, 0]
degree = [0, 1, 2, 3, 4, 5, 6]
num_Hs = [0, 1, 2, 3, 4]
local_chiral_tags = [0, 1, 2, 3]
hybridization = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    ]


def getNodeFeatures(list_rdkit_atoms):
    F_v = (len(atomTypes) + 1) + \
          (len(degree) + 1) + \
          (len(formalCharge) + 1) + \
          (len(num_Hs) + 1) + \
          (len(hybridization) + 1) + \
          2 + 5  # 48

    node_features = np.zeros((len(list_rdkit_atoms), F_v))
    for node_index, node in enumerate(list_rdkit_atoms):
        features = one_hot_embedding(node.GetSymbol(), atomTypes)  # atom symbol, dim=12 + 1
        features += one_hot_embedding(node.GetTotalDegree(), degree)  # total number of bonds, H included, dim=7 + 1
        features += one_hot_embedding(node.GetFormalCharge(), formalCharge)  # formal charge, dim=5+1
        features += one_hot_embedding(node.GetTotalNumHs(), num_Hs)  # total number of bonded hydrogens, dim=5 + 1
        features += one_hot_embedding(node.GetHybridization(), hybridization)  # hybridization state, dim=7 + 1
        features += [int(node.GetIsAromatic())]  # whether atom is part of aromatic system, dim = 1
        features += [node.GetMass() * 0.01]  # atomic mass / 100, dim=1
        # local chiral tag
        features += one_hot_embedding(node.GetChiralTag(),
                                      local_chiral_tags)  # chiral tag of atom, dim=4+1 (local chiral features)

        node_features[node_index, :] = features

    return np.array(node_features, dtype=np.float32)