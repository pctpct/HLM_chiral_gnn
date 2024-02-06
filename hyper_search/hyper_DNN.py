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
from torch.utils.data import DataLoader
from utils import create_logger
from sklearn.metrics import mean_squared_error,r2_score
from model.DNN import DNN_network
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT = 'HLM_prj_hyper_unified'
GROUP_NAME = 'hyper_DNN'


def gen_ECFP4(smi_list,radius=2,nBits=1024):
    fps = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol,radius,nBits=nBits))
        fps.append(fp)
    return np.asarray(fps,dtype=np.int8)


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

def train(model,loss,optimizer,loader,device):
    model.train()
    batch_loss_all = []
    for X,y in loader:
        X = X.to(device)
        y = y.to(device)
        preds = model(X)

        optimizer.zero_grad()
        batch_loss = loss(preds.squeeze(-1),y.float())
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
        for X,y in loader:
            X = X.to(device)
            y = y.to(device)
            preds = model(X)

            batch_loss = loss(preds.squeeze(-1),y.float())
            batch_loss_all.append(batch_loss.item())
            preds_all.extend(preds.squeeze(-1).cpu())
            labels_all.extend(y.float().squeeze(-1).cpu())

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
    batch_size = int(trial.suggest_categorical('batch_size', [128,256,512,1024]))
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 2, 4, 1)
    dropout = trial.suggest_float('dropout', 0.2, 0.5, step=0.1)
    min_exp_hidden_units = trial.suggest_int('exp_hidden_units',6,8,step=1)
    hidden_units_list = [int(np.exp2(min_exp_hidden_units + i)) for i in range(num_hidden_layers)]
    hidden_units_list.reverse()

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
    df = pd.read_csv(args.data_path)
    smiles_list = df.loc[:,'smiles'].values
    label_list = df.loc[:,args.label].values

    split_idx = get_split_idx(len(smiles_list), val_ratio=0.1, test_ratio=0.1, seed=args.seed)
    train_dataset = FPDataset(smiles_list[split_idx['train']],label_list[split_idx['train']])
    val_dataset = FPDataset(smiles_list[split_idx['val']],label_list[split_idx['val']])
    test_dataset = FPDataset(smiles_list[split_idx['test']],label_list[split_idx['test']])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)

    # construct model
    device = torch.device("cuda:1" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model = DNN_network(input_dim=1024,input_drop_rate=0.1,num_hidden_layer=num_hidden_layers,\
                        num_hidden_units_list=hidden_units_list,hidden_drop_rate=dropout).to(device)

    # paramter initialization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

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

    joblib.dump(study, os.path.join(args.hyperopt_dir, "study.pkl"))
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
    parser.add_argument('--restart',action='store_true',default=False)
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

