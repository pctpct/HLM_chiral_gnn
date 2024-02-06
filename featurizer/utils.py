# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : utils.py
@Project  : ChiralityGNN_HLM
@Time     : 2023/8/29 15:07
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/8/29 15:07        1.0             None
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType

import torch_geometric
from torch_geometric.data.data import size_repr
from featurizer.bond_featurizer import getEdgeFeatures
from featurizer.atom_featurizer import getNodeFeatures
from featurizer.embed import embedConformerWithAllPaths


# chiralTag parity
CHIRALTAG_PARITY = {
    ChiralType.CHI_TETRAHEDRAL_CW: +1,
    ChiralType.CHI_TETRAHEDRAL_CCW: -1,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0, # default
}

def parity_features(atom: Chem.rdchem.Atom) -> int:
    """
    Returns the parity of an atom if it is a tetrahedral center.
    +1 if CW, -1 if CCW, and 0 if undefined/unknown

    :param atom: An RDKit atom.
    """
    return CHIRALTAG_PARITY[atom.GetChiralTag()]


def smiles_to_mol_with_tetraHs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    H_ids = [a.GetIdx() for a in mol.GetAtoms() if CHIRALTAG_PARITY[a.GetChiralTag()] != 0]
    if H_ids:
        mol = Chem.AddHs(mol, onlyOnAtoms=H_ids)
    # reomve stereochem label from atoms with less/more than 4 neighbors
    for i in H_ids:
        a = mol.GetAtomWithIdx(i)
        if len(a.GetNeighbors()) != 4:
            a.SetChiralTag(ChiralType.CHI_UNSPECIFIED)
    return mol


def smiles_to_3d_mol(smiles: str, max_number_of_atoms: int = 100, max_number_of_attempts: int = 5000):
    """
    Embeds the molecule in 3D space.
    Args:
        smiles: a smile representing molecule
        max_number_of_atoms: maximal number of atoms in a molecule. Molecules with more atoms will be omitted.
            max_number_of_attempts: maximal number of attempts during the embedding.
        max_number_of_attempts: max number of embeddings attempts.

    Returns:
        Embedded molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if len(mol.GetAtoms()) > max_number_of_atoms:
        print(f'Omitting molecule {smiles} as it contains more than {max_number_of_atoms} atoms.')
        return None
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, maxAttempts=max_number_of_attempts, randomSeed=0)
    if res < 0:  # try to embed with different method
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=max_number_of_attempts,
                                    randomSeed=0)
    if res < 0:
        print(f'Omitting molecule {smiles} as cannot be embedded in 3D space properly.')
        return None
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        print(
            f"Omitting molecule {smiles} as cannot be properly optimized. "
            f"The original error message was: {e}."
        )
        return None
    return mol


def mol_to_data(mol):
    adj = Chem.GetAdjacencyMatrix(mol)
    edge_index = adjacency_to_undirected_edge_index(adj)

    # edge Features
    bonds = []
    for b in range(int(edge_index.shape[1] / 2)):
        bond_index = edge_index[:, ::2][:, b]
        bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = getEdgeFeatures(bonds)

    # atom features
    atoms = Chem.rdchem.Mol.GetAtoms(mol)
    node_features = getNodeFeatures(atoms)

    #
    parity_feat = []
    for atom in atoms:
        parity_feat.append(parity_features(atom))

    data = torch_geometric.data.Data(
        x=torch.as_tensor(node_features),\
        edge_index=torch.as_tensor(edge_index,dtype=torch.long),\
        edge_attr=torch.as_tensor(edge_features)
    )
    data.parity_atoms = torch.tensor(parity_feat,dtype=torch.long)
    return data

def get_chiro_data_from_mol(mol):
    """
    Copied from `ChIRo.model.datasets_samplers.MaskedGraphDataset.process_mol`. It encoded molecule with some basic
    chemical features. It also provides chiral tag, which can be then masked in `graphgps.dataset.rs_dataset.RS`.
    """
    atom_symbols, edge_index, edge_features, \
        node_features, bond_distances, \
        bond_distance_index, bond_angles, \
        bond_angle_index, dihedral_angles, dihedral_angle_index = embedConformerWithAllPaths(
        mol, repeats=False, chiral_atom=False)

    bond_angles = bond_angles % (2 * np.pi)
    dihedral_angles = dihedral_angles % (2 * np.pi)
    pos = get_positions(mol)

    data = torch_geometric.data.Data(x=torch.as_tensor(node_features),
                                     edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                                     edge_attr=torch.as_tensor(edge_features),
                                     pos=torch.as_tensor(pos, dtype=torch.float))
    data.bond_distances = torch.as_tensor(bond_distances)
    data.bond_distance_index = torch.as_tensor(bond_distance_index, dtype=torch.long).T
    data.bond_angles = torch.as_tensor(bond_angles)
    data.bond_angle_index = torch.as_tensor(bond_angle_index, dtype=torch.long).T
    data.dihedral_angles = torch.as_tensor(dihedral_angles)
    data.dihedral_angle_index = torch.as_tensor(dihedral_angle_index, dtype=torch.long).T

    return data

def adjacency_to_undirected_edge_index(adj):
    adj = np.triu(np.array(adj, dtype = int)) #keeping just upper triangular entries from sym matrix
    array_adj = np.array(np.nonzero(adj), dtype = int) #indices of non-zero values in adj matrix
    edge_index = np.zeros((2, 2*array_adj.shape[1]), dtype = int) #placeholder for undirected edge list
    edge_index[:, ::2] = array_adj
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index


def get_positions(mol):
    conf = mol.GetConformer()
    return np.array(
        [
            [
                conf.GetAtomPosition(k).x,
                conf.GetAtomPosition(k).y,
                conf.GetAtomPosition(k).z,
            ]
            for k in range(mol.GetNumAtoms())
        ]
    )


def get_atom_dim():
    return 48

def get_bond_dim():
    return 14