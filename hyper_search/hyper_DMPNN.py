# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : hyper_DMPNN.py
@Project  : ChiralityGNN_HLM
@Time     : 2023/8/25 21:09
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/8/25 21:09        1.0             None
"""
import os
import random
import shutil
import math
import pickle
import pandas as pd
import wandb
import optuna
import joblib

import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from utils import create_logger
from sklearn.metrics import mean_squared_error,r2_score
from model.DMPNN import DMPNN
from pyg_dataset import MoleculeDataset,RevIndexedDataset

PROJECT = 'HLM_prj_hyper_local'
GROUP_NAME = 'hyper_DMPNN'

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

def get_split_idx(len_data,val_ratio,test_ratio,seed):
    index = list(range(len_data))
    random.seed(seed)
    random.shuffle(index)
    val_index = index[:int(len_data * val_ratio)]
    test_index = index[int(len_data * val_ratio):int(len_data *(val_ratio+test_ratio))]
    train_index = index[int(len_data *(val_ratio+test_ratio)):]

    split_idx = {'train':train_index,'val':val_index,'test':test_index}
    return split_idx


def optimize(trial, args):
    batch_size = int(trial.suggest_categorical('batch_size', [16,32,64,128]))
    gnn_hidden_size = trial.suggest_int('hidden_size',300,1200,step=300)
    mlp_hidden_size = trial.suggest_int('mlp_hidden_size',300,1200,step=300)
    dropout = trial.suggest_float('dropout',0,0.8,step=0.2)
    depth = trial.suggest_int('depth', 2, 6, 1)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    graph_pool = trial.suggest_categorical('graph_pool', ['sum', 'mean', 'max'])
    # setattr(args, 'hidden_size', int(trial.suggest_discrete_uniform('hidden_size', 300, 1200, 300)))
    # setattr(args, 'depth', int(trial.suggest_discrete_uniform('depth', 2, 6, 1)))
    # setattr(args, 'dropout', int(trial.suggest_discrete_uniform('dropout', 0, 1, 0.2)))
    # setattr(args, 'lr', trial.suggest_loguniform('lr', 1e-5, 1e-3))
    # setattr(args, 'batch_size', int(trial.suggest_categorical('batch_size', [25, 50, 100])))
    # setattr(args, 'graph_pool', trial.suggest_categorical('graph_pool', ['sum', 'mean', 'max', 'attn', 'set2set']))

    setattr(args, 'log_dir', os.path.join(args.hyperopt_dir, str(trial._trial_id)))

    config = dict(trial.params)
    wandb.init(
        project=PROJECT,
        entity="pct",  # NOTE: this entity depends on your wandb account.
        config=config,
        group=GROUP_NAME,
        reinit=True,
    )
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_logger = create_logger('train', args.log_dir)

    # construct dataset and dataloader
    # construct root dir
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

    # constrcut dataset and loader
    Moldataset = MoleculeDataset(root=data_dir, file_name=file_name, label_col=args.label)
    dataset = RevIndexedDataset(Moldataset)
    split_idx = get_split_idx(len(dataset),val_ratio=0.1,test_ratio=0.1,seed=args.seed)
    train_loader = DataLoader(dataset[split_idx['train']],batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(dataset[split_idx['val']],batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    test_loader = DataLoader(dataset[split_idx['test']],batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)

    # construct model
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model = DMPNN(gnn_hidden_size, Moldataset.atom_feat_dim, Moldataset.bond_feat_dim, depth, \
                  mlp_hidden_size=mlp_hidden_size, drop_rate=dropout, graph_pool=graph_pool).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = None
    loss = nn.MSELoss(reduction='sum')
    best_val_rmse, best_val_r2 = np.inf, np.inf
    best_epoch = 0

    # log info
    train_logger.info('Arguments are...')
    for key,value in config.items():
        train_logger.info(f'{key}: {value}')

    train_logger.info("Starting training...")
    for epoch in range(args.n_epochs):
        train_epoch_loss = train(model, loss, optimizer, train_loader, device)
        train_logger.info(f"Epoch {epoch}: Training Loss {train_epoch_loss}")
        val_epoch_loss, val_rmse, val_r2 = eval(model, loss, val_loader, device)
        train_logger.info(f"Epoch {epoch}: Validation Loss {val_epoch_loss}")
        wandb.log({'Train_epoch_loss':train_epoch_loss,'Val_epoch_loss':val_epoch_loss})

        if val_rmse < best_val_rmse:
            best_epoch = epoch
            wandb.run.summary["best_val_rmse"] = val_rmse
            best_val_rmse = val_rmse
            wandb.run.summary["best_val_r2"] = val_r2
            best_val_r2 = val_r2
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
            }, os.path.join(args.log_dir, 'best_saved_model.pt'))

        # report intermediate results for early stopping
        trial.report(val_epoch_loss,epoch)

        # handle pruning based on the intermediate
        if trial.should_prune():
            train_logger.handlers = []
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()

    train_logger.info(f"Best Validation Loss {best_val_rmse} on Epoch {best_epoch}")

    #
    checkpoint = torch.load(os.path.join(args.log_dir,"best_saved_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])

    _,train_rmse,train_r2 = eval(model,loss,train_loader,device)
    _,test_rmse,test_r2 = eval(model,loss,test_loader,device)
    wandb.run.summary['train_rmse'] = train_rmse
    wandb.run.summary['train_r2'] = train_r2
    wandb.run.summary['test_rmse']= test_rmse
    wandb.run.summary['test_r2'] = test_r2
    wandb.finish(quiet=True)

    return best_val_rmse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # common parameters
    parser.add_argument('--data_path',type=str,
                        help='Path to CSV file with smiles and labels.')
    parser.add_argument('--label',type=str,
                        help='The lable col of csv file.')
    parser.add_argument('--seed',type=int,
                        help='Random seed.')
    parser.add_argument('--n_epochs',type=int,default=128,\
                        help='Number of epochs to run')
    parser.add_argument('--use_gpu',action='store_true',default=False)
    parser.add_argument('--restart',type=bool,default=False)
    parser.add_argument('--log_dir',type=str,default=None,\
                        help='Directory where model checkpoints will be saved')

    # hyperparamters
    parser.add_argument('--hyperopt_dir', type=str,
                        help='Directory to save all results')
    parser.add_argument('--n_trials', type=int, default=25,\
                        help='Number of hyperparameter choices to try')

    args = parser.parse_args()

    if not os.path.exists(args.hyperopt_dir):
        os.makedirs(args.hyperopt_dir)

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler(os.path.join(args.hyperopt_dir, "hyperopt.log"), mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    if args.restart:
        study = joblib.load(os.path.join(args.hyperopt_dir, "study.pkl"))
    else:
        study = optuna.create_study(
            pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=args.n_epochs, reduction_factor=2),
            sampler=optuna.samplers.CmaEsSampler()
        )
        # The Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a powerful optimization algorithm for continuous search spaces.
        # This type of space works best for continuous search spaces and less well for discrete or categorical hyperparameters.
    joblib.dump(study, os.path.join(args.hyperopt_dir, "study.pkl"))

    logger.info("Running optimization...")
    study.optimize(lambda trial: optimize(trial, args), n_trials=args.n_trials)
