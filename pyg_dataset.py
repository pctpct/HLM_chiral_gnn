# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : pyg_dataset.py
@Project  : ChiralityGNN_HLM
@Time     : 2023/8/29 10:15
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/8/29 10:15        1.0             None
"""
import os.path as osp
import tqdm
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import size_repr
from torch_geometric.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from featurizer.utils import smiles_to_mol_with_tetraHs,mol_to_data
from featurizer.utils import smiles_to_3d_mol,get_chiro_data_from_mol
from featurizer.utils import get_atom_dim,get_bond_dim
# transform 函数中定义是否 mask chiral tag(***)

def get_mask_chiral(data):
    data.x[:,-5:] = 0.0
    # data.edge_attr[:,-7:] = 0.0
    return data

def gen_ECFP4(smi_list,radius=2,nBits=1024):
    fps = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol,radius,nBits=nBits))
        fps.append(fp)
    return np.asarray(fps,dtype=np.int8)

class MoleculeDataset(InMemoryDataset):
    def __init__(self,root,file_name,smi_col='smiles',label_col='labels',graph_type='basic',\
                 transform=None, pre_transform=None, pre_filter=None):
        self.file_name = file_name
        self.smi_col = smi_col
        self.label_col = label_col
        self.graph_type = graph_type
        self.atom_feat_dim = get_atom_dim()
        self.bond_feat_dim = get_bond_dim()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.file_name]

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def process(self):
        mol_df = pd.read_csv(self.raw_paths[0])
        data_list = []
        for smi,label in zip(mol_df.loc[:,self.smi_col],mol_df.loc[:,self.label_col]):
            if self.graph_type == 'basic':
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)
                data = mol_to_data(mol)
            elif self.graph_type == 'tetra':
                mol = smiles_to_mol_with_tetraHs(smi)
                data = mol_to_data(mol)
            else:
                raise ValueError(f"{self.graph_type} doesn't support yet")
            data.y = label
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])


class MoleculeDataset_3D(Dataset):
    def __init__(self,root,file_name,smi_col='smiles',label_col='labels',transform=None,pre_transform=None,pre_filter=None):
        self.file_name = file_name
        self.smi_col = smi_col
        self.label_col = label_col
        self.atom_feat_dim = get_atom_dim()
        self.bond_feat_dim = get_bond_dim()
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.file_name

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in list(self.data.index)]

    def process(self):
        mol_df = pd.read_csv(self.raw_paths[0])
        idx = 0
        invalid_mol = 0
        for smi,label in zip(mol_df.loc[:,self.smi_col],mol_df.loc[:,self.label_col]):
            mol = smiles_to_3d_mol(smi)
            if mol is None:
                invalid_mol += 1
                continue
            try:
                data = get_chiro_data_from_mol(mol)
                data.y = label
                torch.save(data,osp.join(self.processed_dir,f'data_{idx}.pt'))
                idx += 1
            except Exception as e:
                invalid_mol += 1
                print(f'Omitting molecule {self.smiles} as cannot be properly embedded. '
                      f'The original error message was: {e}.')  # probably it does not have sufficient number of paths of length 3.


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir,f'data_{idx}.pt'))
        return data


class RevIndexedData(torch_geometric.data.Data):
    def __init__(self, orig):
        super(RevIndexedData, self).__init__()
        if orig:
            for key in orig.keys:
                self[key] = orig[key]
            edge_index = self["edge_index"]
            revedge_index = torch.zeros(edge_index.shape[1]).long()
            for k, (i, j) in enumerate(zip(*edge_index)): # i -> j ,reverse,j -> i
                edge_to_i = edge_index[1] == i
                edge_from_j = edge_index[0] == j
                revedge_index[k] = torch.where(edge_to_i & edge_from_j)[0].item()
            self["revedge_index"] = revedge_index

    def __inc__(self, key, value, *args, **kwargs):
        if key == "revedge_index":
            return self.revedge_index.max().item() + 1
        else:
            return super().__inc__(key, value)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))


class RevIndexedDataset(Dataset):
    def __init__(self, orig):
        super(RevIndexedDataset, self).__init__()
        self.dataset = [RevIndexedData(data) for data in orig]
        self.atom_feat_dim = orig.atom_feat_dim
        self.bond_feat_dim = orig.bond_feat_dim

    # def __getitem__(self, idx):
    #     return self.dataset[idx]
    #
    # def __len__(self):
    #     return len(self.dataset)
    def get(self,idx):
        return self.dataset[idx]

    def len(self):
        return len(self.dataset)



class FPDataset(torch.utils.data.Dataset):
    def __init__(self,smi_list,label_list):
        super().__init__()
        self.smi_list = smi_list
        self.label_list = torch.tensor(label_list,dtype=torch.float32)
        self.fp_list = torch.tensor(gen_ECFP4(self.smi_list),dtype=torch.float32)

    def __getitem__(self,idx):
        return self.fp_list[idx],self.label_list[idx]

    def __len__(self):
        return self.fp_list.shape[0]

if __name__ == "__main__":
    print('hello')
    # dataset = MoleculeDataset_3D(root='data/larger_dataset/type_3/',file_name='HLM_processed_mlminkg_stere_keeped.csv',label_col='log HLM(ml/min/kg)',transform=get_mask_chiral)
    # smi = 'CCC'
    # mol = Chem.MolFromSmiles(smi)
    # data = mol_to_data(mol)
    # data = revindexData(data)
    # data_list = [data,data]
    # from torch_geometric.loader import DataLoader
    # loader = DataLoader(data_list,batch_size=2)
    # batch = next(iter(loader))
    # print(batch['revedge_index'])
    # print(batch)
    # print(type(data))
    # data_root = './data/ulminmg_only/'
    # file_name = 'HLM_processed_mlminkg_stere_keeped.csv'
    #
    # import os
    # import shutil
    # print(osp.exists(data_root))
    # raw_path = osp.join(data_root,'raw')
    # if osp.exists(raw_path):
    #     if osp.exists(osp.join(raw_path,file_name)):
    #         pass
    #     else:
    #         shutil.copy(osp.join(data_root, file_name), raw_path)
    # else:
    #     os.mkdir(raw_path)
    #     shutil.copy(osp.join(data_root,file_name),raw_path)
    #
    import os
    data_root = './data/larger_dataset/type_3/'
    file_name = 'HLM_processed_mlminkg_stere_keeped.csv'
    label = 'log HLM(ml/min/kg)'
    mask_chiral_tag = get_mask_chiral
    dataset = MoleculeDataset_3D(root=data_root, file_name=file_name, label_col=label, transform=mask_chiral_tag)

# def construct_dataset(args):
#     # specify the data path
#     data_dir = os.path.split(args.data_path)[0]
#     file_name = os.path.split(args.data_path)[1]
#     if os.path.exists(os.path.join(data_dir,'raw')):
#         if os.path.exists(os.path.join(data_dir,'raw',file_name)):
#             pass
#         else:
#             shutil.copy(args.data_path,os.path.join(data_dir,'raw'))
#     else:
#         os.mkdir(os.path.join(data_dir,'raw'))
#         shutil.copy(args.data_path,os.path.join(data_dir,'raw'))
#     if args.chiral_tag:
#         mask_chiral_tag = None
#     else:
#         mask_chiral_tag = get_mask_chiral
#
#     if args.model == 'ChIRo':
#         dataset = MoleculeDataset_3D(root=data_dir,file_name=file_name,label_col=args.label,transform=mask_chiral_tag)
#     elif args.model == 'TetraDMPNN':
#         dataset = MoleculeDataset(root=data_dir,file_name=file_name,label_col=args.label,graph_type=args.graph_type,transform=mask_chiral_tag)
#     elif args.model == 'DMPNN':
#         Moldataset = MoleculeDataset(root=data_dir, file_name=file_name, label_col=args.label,
#                                      transform=mask_chiral_tag)
#         dataset = RevIndexedDataset(Moldataset)
#     else:
#         raise ValueError(f'{args.model} does not supported yet.')
#     return dataset

