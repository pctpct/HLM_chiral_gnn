# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : main.py
@Project  : ChiralityGNN_HLM
@Time     : 2023/8/30 15:07
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/8/30 15:07        1.0             None
"""
import os
import random
import json
import pickle
import numpy as np
import shutil
import argparse
import multiprocessing as mp

from sklearn.metrics import mean_squared_error,r2_score
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pyg_dataset import MoleculeDataset,RevIndexedDataset,MoleculeDataset_3D
from model.DMPNN import DMPNN
from model.Tetra_DMPNN import TetraDMPNN
from model.ChIRo_gnn import Encoder

import wandb
# os.environ['WANDB_DIR'] = '/data/ctpu/ChiralityGNN_HLM/tetra_exp2_wandb'
def get_mask_chiral(data):
    data.x[:,-5:] = 0.0
    # data.edge_attr[:,-7:] = 0.0
    return data


def get_split_idx(len_data,val_ratio,test_ratio,seed):
    index = list(range(len_data))
    random.seed(seed)
    random.shuffle(index)
    val_index = index[:int(len_data * val_ratio)]
    test_index = index[int(len_data * val_ratio):int(len_data *(val_ratio+test_ratio))]
    train_index = index[int(len_data *(val_ratio+test_ratio)):]

    split_idx = {'train':train_index,'val':val_index,'test':test_index}
    return split_idx


def get_model(in_atom_feat,in_bond_feat,args):
    with open(args.model_params) as f:
        params = json.load(f)
    if args.model == 'DMPNN':
        model = DMPNN(num_node_features=in_atom_feat,num_edge_features=in_bond_feat,**params)
    elif args.model == 'TetraDMPNN':
        model = TetraDMPNN(num_node_features=in_atom_feat,num_edge_features=in_bond_feat,**params)
    elif args.model == 'ChIRo':
        model = Encoder(F_H_embed=in_atom_feat,F_E_embed=in_bond_feat,**params)
    else:
        raise ValueError('{args.model} does not supported yet!')

    return model


def train(model,loss,optimizer,loader,device):
    model.train()
    batch_loss_all = []
    for batch_idx,data in enumerate(loader):
        data = data.to(device)
        preds = model(data)

        optimizer.zero_grad()
        batch_loss = loss(preds.squeeze(-1),data.y.float())
        batch_loss.backward()
        optimizer.step()

        batch_loss_all.append(batch_loss.item())

    return np.average(batch_loss_all)


def eval(model,loss,loader,device):
    model.eval()
    batch_loss_all = []
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch_id,data in enumerate(loader):
            data = data.to(device)
            preds = model(data)

            batch_loss = loss(preds.squeeze(-1),data.y.float())
            batch_loss_all.append(batch_loss.item())
            preds_all.extend(preds.squeeze(-1).cpu())
            labels_all.extend(data.y.float().squeeze(-1).cpu())

    return np.average(batch_loss_all),mean_squared_error(labels_all,preds_all,squared=False),r2_score(labels_all,preds_all)

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

def construct_dataset(args):
    # specify the data path
    data_dir = os.path.split(args.data_path)[0]
    file_name = os.path.split(args.data_path)[1]
    if os.path.exists(os.path.join(data_dir,'raw')):
        if os.path.exists(os.path.join(data_dir,'raw',file_name)):
            pass
        else:
            shutil.copy(args.data_path,os.path.join(data_dir,'raw'))
    else:
        os.mkdir(os.path.join(data_dir,'raw'))
        shutil.copy(args.data_path,os.path.join(data_dir,'raw'))
    if args.chiral_tag:
        mask_chiral_tag = None
    else:
        mask_chiral_tag = get_mask_chiral

    if args.model == 'ChIRo':
        dataset = MoleculeDataset_3D(root=data_dir,file_name=file_name,label_col=args.label,transform=mask_chiral_tag)
    elif args.model == 'TetraDMPNN':
        dataset = MoleculeDataset(root=data_dir,file_name=file_name,label_col=args.label,graph_type=args.graph_type,transform=mask_chiral_tag)
    elif args.model == 'DMPNN':
        Moldataset = MoleculeDataset(root=data_dir, file_name=file_name, label_col=args.label,
                                     transform=mask_chiral_tag)
        dataset = RevIndexedDataset(Moldataset)
    else:
        raise ValueError(f'{args.model} does not supported yet.')
    return dataset

def run(args,run_id,fold_idx,device,results_squeue):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # init wandb run
    reset_wandb_env()
    run_name = f'{run_id}_{fold_idx}'
    run = wandb.init(
        project = args.prj_name,
        group = args.model,
        name = run_name
    )

    # directory to save model
    checkpoint_path = os.path.join(args.checkpoints_dir,run_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path,exist_ok=True)

    data_dir = os.path.split(args.data_path)[0]
    file_name = os.path.split(args.data_path)[1]
    if os.path.exists(os.path.join(data_dir,'raw')):
        if os.path.exists(os.path.join(data_dir,'raw',file_name)):
            pass
        else:
            shutil.copy(args.data_path,os.path.join(data_dir,'raw'))
    else:
        os.mkdir(os.path.join(data_dir,'raw'))
        shutil.copy(args.data_path,os.path.join(data_dir,'raw'))

    # model hyper parameters
    with open(args.model_params) as f:
        params = json.load(f)

    # constrcut dataset and model
    if args.chiral_tag:
        mask_chiral_tag = None
    else:
        mask_chiral_tag = get_mask_chiral
    if args.model == 'ChIRo':
        dataset = MoleculeDataset_3D(root=data_dir,file_name=file_name,label_col=args.label,transform=mask_chiral_tag)
        # dataset = MoleculeDataset(root=data_dir,file_name=file_name,label_col=args.label,)
        model = Encoder(F_H_embed=dataset.atom_feat_dim,F_E_embed=dataset.bond_feat_dim,**params).to(device)
        # if args.aux_loss:
        #     pass
    elif args.model == 'TetraDMPNN':
        dataset = MoleculeDataset(root=data_dir,file_name=file_name,label_col=args.label,graph_type=args.graph_type,transform=mask_chiral_tag)
        model = TetraDMPNN(num_node_features=dataset.atom_feat_dim,num_edge_features=dataset.bond_feat_dim,**params).to(device)
    elif args.model == 'DMPNN':
        Moldataset = MoleculeDataset(root=data_dir, file_name=file_name, label_col=args.label, transform=mask_chiral_tag)
        dataset = RevIndexedDataset(Moldataset)
        model = DMPNN(num_node_features=dataset.atom_feat_dim,num_edge_features=dataset.bond_feat_dim,**params).to(device)
    elif args.model == 'DNN':
        dataset = None
        model = None

    else:
        raise ValueError(f'{args.model} does not supported yet.')

    # 划分训练，测试，验证集
    split_idx = get_split_idx(len(dataset),val_ratio=0.1,test_ratio=0.1,seed=args.seed+fold_idx)
    train_loader = DataLoader(dataset[split_idx['train']],batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(dataset[split_idx['val']],batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(dataset[split_idx['test']],batch_size=args.batch_size,shuffle=False)

    # define optimzer / schduler
    optimizer = Adam(model.parameters(),lr=args.lr) # weight_decay = args.weight_decay
    scheduler = None

    loss = nn.MSELoss(reduction='sum')
    # if args.aux_loss:
    #     pass

    best_val_rmse,best_val_r2 = np.inf,np.inf
    best_epoch = 0
    for epoch in range(args.n_epochs):
        train_epoch_loss = train(model,loss,optimizer,train_loader,device)
        val_epoch_loss,val_rmse,val_r2 = eval(model,loss,val_loader,device)
        _,test_rmse,test_r2 = eval(model,loss,test_loader,device)

        # logging loss
        run.log({'Train Epoch Loss' : train_epoch_loss,\
                   'Val Epoch Loss' : val_epoch_loss,})

        # Output the final results
        if val_rmse < best_val_rmse:
            best_epoch = epoch
            best_val_rmse = val_rmse
            best_val_r2 = val_r2

            # save the model
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()
            },os.path.join(checkpoint_path,"best_saved_model.pt"))

            # save the split  idx
            with open(os.path.join(checkpoint_path,'split_idx.pkl'),'wb') as f:
                pickle.dump(split_idx,f)

    results_squeue.put({
        'best_epoch':best_epoch,
        'best_val_rmse':best_val_rmse,
        'best_val_r2':best_val_r2,
        'test_rmse':test_rmse,
        'test_r2':test_r2
    })

    run.summary[f'best_epoch'] = best_epoch
    run.summary[f'best_val_rmse'] = best_val_rmse
    run.summary[f'best_val_r2'] = best_val_r2
    run.summary[f'test_rmse'] = test_rmse
    run.summary[f'test_r2'] = test_r2

    wandb.finish()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',type=str,
                        help='Detailed path of the data file.(xxx/xxx.csv)')
    parser.add_argument('--label',type=str,
                        help='The column name of the task in csv file.')
    parser.add_argument('--model',type=str,\
                        help='Type of gnn model {DMPNN,TetraDMPNN,ChIRo}.')
    parser.add_argument('--num_folds',type=int,default=10,\
                        help='The total number of run folds')
    #
    parser.add_argument('--n_epochs',type=int,default=100,
                        help='number of epochs to train.')
    parser.add_argument('--lr',type=float,default=1e-4,
                        help='learning rate fro training.')
    parser.add_argument('--batch_size',type=int,default=32,
                        help='Input batch size for training.')

    # model parameters
    parser.add_argument('--model_params',type=str,
                        help='The parameters for constructing the model.(xxx.json)')
    parser.add_argument('--graph_type',type=str,
                        help="The graph data type for different models.('basic'/'tetra')")

    #
    parser.add_argument('--seed',type=int,default=42,
                        help='random seed.')
    parser.add_argument('--prj_name',type=str,
                        help='The project name of wandb.')
    parser.add_argument('--checkpoints_dir',type=str,\
                        help='The directory to save trained model.')
    parser.add_argument('--chiral_tag',action='store_true',default=False,\
                        help='Whether to incorporate chiral tag(CW/CCW) in node feature.')
    parser.add_argument('--use_gpu',action='store_true',default=False,
                        help='Whether to use GPU')
    parser.add_argument('--test', action='store_true',
                        help='quick test')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    mp.set_start_method('spawn')
    summary_run = wandb.init(project=args.prj_name,
                             group=args.model)

    run_name = summary_run.name

    results_queue = mp.Queue()

    # only for test
    if args.test:
        fold_idx = 0
        run(args,run_name,fold_idx,device,results_queue)
        exit()

    # running
    num_free = 1
    fold_idx = 0
    folds_result_list  = []
    while len(folds_result_list) < args.num_folds:
        if num_free > 0 and fold_idx < args.num_folds:
            p = mp.Process(
                target=run,args=(args,run_name,fold_idx,device,results_queue))
            fold_idx += 1
            num_free -= 1
            p.start()
        else:
            folds_result_list.append(results_queue.get())
            num_free += 1

    best_epoch = [fold_result['best_epoch'] for fold_result in folds_result_list]
    best_val_rmse = [fold_result['best_val_rmse'] for fold_result in folds_result_list]
    best_val_r2 = [fold_result['best_val_r2'] for fold_result in folds_result_list]
    test_rmse_list = [fold_result['test_rmse'] for fold_result in folds_result_list]
    test_r2_list = [fold_result['test_r2'] for fold_result in folds_result_list]

    val_rmse_mean = np.mean(best_val_rmse)
    val_rmse_std = np.std(best_val_rmse)
    val_r2_mean = np.mean(best_val_r2)
    val_r2_std = np.std(best_val_r2)
    test_rmse_mean = np.mean(test_rmse_list)
    test_rmse_std = np.std(test_rmse_list)
    test_r2_mean = np.mean(test_r2_list)
    test_r2_std = np.std(test_r2_list)

    summary_run.summary[f'val_rmse_mean'] = val_rmse_mean
    summary_run.summary[f'val_rmse_std'] = val_rmse_std
    summary_run.summary[f'val_r2_mean'] = val_r2_mean
    summary_run.summary[f'val_r2_std'] = val_r2_std
    summary_run.summary[f'test_rmse_mean'] = test_rmse_mean
    summary_run.summary[f'test_rmse_std'] = test_rmse_std
    summary_run.summary[f'test_r2_mean'] = test_r2_mean
    summary_run.summary[f'test_r2_std'] = test_r2_std

    wandb.finish()