# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : embed_mol.py
@Project  : ChiralityGNN_HLM
@Time     : 2023/8/29 21:39
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/8/29 21:39        1.0             None
"""
import numpy as np
import networkx as nx
import rdkit
from rdkit.Chem import rdMolTransforms
# from featurizer.utils import adjacency_to_undirected_edge_index
from featurizer.atom_featurizer import getNodeFeatures
from featurizer.bond_featurizer import getEdgeFeatures
def adjacency_to_undirected_edge_index(adj):
    adj = np.triu(np.array(adj, dtype = int)) #keeping just upper triangular entries from sym matrix
    array_adj = np.array(np.nonzero(adj), dtype = int) #indices of non-zero values in adj matrix
    edge_index = np.zeros((2, 2*array_adj.shape[1]), dtype = int) #placeholder for undirected edge list
    edge_index[:, ::2] = array_adj
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index

def embedConformerWithAllPaths(rdkit_mol3D, repeats=False, chiral_atom=False):
    if isinstance(rdkit_mol3D, rdkit.Chem.rdchem.Conformer):
        mol = rdkit_mol3D.GetOwningMol()
        conformer = rdkit_mol3D
    elif isinstance(rdkit_mol3D, rdkit.Chem.rdchem.Mol):
        mol = rdkit_mol3D
        conformer = mol.GetConformer()

    # Edge Index
    adj = rdkit.Chem.GetAdjacencyMatrix(mol)
    edge_index = adjacency_to_undirected_edge_index(adj)

    # Edge Features
    bonds = []
    for b in range(int(edge_index.shape[1] / 2)):
        bond_index = edge_index[:, ::2][:, b]
        bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = getEdgeFeatures(bonds)

    # Node Features
    atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
    atom_symbols = [atom.GetSymbol() for atom in atoms]
    node_features = getNodeFeatures(atoms)

    bond_distances, bond_distance_indices, bond_angles, bond_angle_indices, dihedral_angles, dihedral_angle_indices = getInternalCoordinatesFromAllPaths(
        conformer, adj, repeats=repeats)

    return atom_symbols, edge_index, edge_features, node_features, bond_distances, bond_distance_indices, bond_angles, bond_angle_indices, dihedral_angles, dihedral_angle_indices


def getInternalCoordinatesFromAllPaths(mol, adj, repeats=False):
    if isinstance(mol, rdkit.Chem.rdchem.Conformer):
        conformer = mol
    if isinstance(mol, rdkit.Chem.rdchem.Mol):
        conformer = mol.GetConformer()

    graph = nx.from_numpy_array(adj, parallel_edges=False, create_using=None)

    distance_paths, angle_paths, dihedral_paths = get_all_paths(graph, N=1), get_all_paths(graph, N=2), get_all_paths(
        graph, N=3)

    if len(dihedral_paths) == 0:
        raise Exception('No Dihedral Angle Detected')

    bond_distance_indices = np.array(distance_paths, dtype=int)
    bond_angle_indices = np.array(angle_paths, dtype=int)
    dihedral_angle_indices = np.array(dihedral_paths, dtype=int)

    if not repeats:  # only taking (0,1) vs. (1,0); (1,2,3) vs (3,2,1); (1,3,6,7) vs (7,6,3,1)
        bond_distance_indices = bond_distance_indices[bond_distance_indices[:, 0] < bond_distance_indices[:, 1]]
        bond_angle_indices = bond_angle_indices[bond_angle_indices[:, 0] < bond_angle_indices[:, 2]]
        dihedral_angle_indices = dihedral_angle_indices[dihedral_angle_indices[:, 1] < dihedral_angle_indices[:, 2]]

    bond_distances = np.array(
        [rdMolTransforms.GetBondLength(conformer, int(index[0]), int(index[1])) for index in bond_distance_indices],
        dtype=np.float32)
    bond_angles = np.array(
        [rdMolTransforms.GetAngleRad(conformer, int(index[0]), int(index[1]), int(index[2])) for index in
         bond_angle_indices], dtype=np.float32)
    dihedral_angles = np.array(
        [rdMolTransforms.GetDihedralRad(conformer, int(index[0]), int(index[1]), int(index[2]), int(index[3])) for index
         in dihedral_angle_indices], dtype=np.float32)

    return bond_distances, bond_distance_indices, bond_angles, bond_angle_indices, dihedral_angles, dihedral_angle_indices


def get_all_paths(G, N=3):
    # adapted from: https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph

    def findPaths(G, u, n):
        if n == 0:
            return [[u]]
        paths = [[u] + path for neighbor in G.neighbors(u) for path in findPaths(G, neighbor, n - 1) if u not in path]
        return paths

    allpaths = []
    for node in G:
        allpaths.extend(findPaths(G, node, N))

    return allpaths