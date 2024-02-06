# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : hyper_ChRIo.py
@Project  : ChiralityGNN_HLM_2
@Time     : 2023/9/3 11:19
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/9/3 11:19        1.0             None
"""
import os
import random
import shutil
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
from model.ChIRo_gnn import Encoder
from pyg_dataset import MoleculeDataset_3D
PROJECT = 'HLM_prj_hyper_exp1'
GROUP_NAME = 'hyper_ChIRo'

def get_mask_chiral(data):
    data.x[:,-5:] = 0.0
    # data.edge_attr[:,-7:] = 0.0
    return data

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
    #
    batch_size = int(trial.suggest_categorical('batch_size', [16,32,64,128,256,512]))
    lr = trial.suggest_loguniform('lr',1e-4,1e-3)

    #
    F_z = int(trial.suggest_categorical('F_z',[8,16,32,64]))
    F_z_list = [F_z] * 3
    F_H = int(trial.suggest_categorical('F_H',[8,16,32,64]))
    F_H_EConv = F_H ## 保留

    #
    Econv_mlp_hidden_size = trial.suggest_categorical('EConv_mlp_hidden_size',[32,64,128,256])
    EConv_mlp_hidden_layer_number = trial.suggest_categorical('EConv_mlp_hidden_layer_number',[1,2])
    EConv_mlp_hidden_sizes = [Econv_mlp_hidden_size] * EConv_mlp_hidden_layer_number

    GAT_hidden_node_size = trial.suggest_categorical('GAT_hidden_node_size',[16,32,64])
    GAT_hidden_layer_number = trial.suggest_categorical('GAT_hidden_layer_number',[1,2,3]) # pay attention
    GAT_hidden_node_sizes = [GAT_hidden_node_size] * GAT_hidden_layer_number

    encoder_hidden_size = trial.suggest_categorical('encoder_hidden_size', [32, 64, 128, 256])
    encoder_hidden_layer_number = trial.suggest_categorical('encoder_hidden_layer_number',[1,2,3,4])
    encoder_hidden_sizes = [encoder_hidden_size] * encoder_hidden_layer_number

    encoder_sinusoidal_shift_hidden_size = trial.suggest_categorical('encoder_sinusoidal_shift_hidden_size', [32, 64, 128, 256])
    encoder_sinusoidal_shift_hidden_layer_number = trial.suggest_categorical('encoder_sinusoidal_shift_hidden_layer_number',[1,2,3,4])
    encoder_hidden_sizes_sinusoidal_shift = [encoder_sinusoidal_shift_hidden_size] * encoder_sinusoidal_shift_hidden_layer_number


    output_mlp_hidden_size = trial.suggest_categorical('output_mlp_hidden_size', [32, 64, 128, 256])
    output_mlp_hidden_layer_number = trial.suggest_categorical('output_mlp_hidden_layer_number',[1,2,3,4])
    output_mlp_hidden_sizes = [output_mlp_hidden_size] * output_mlp_hidden_layer_number

    layers_dict = {
        "EConv_mlp_hidden_sizes" : EConv_mlp_hidden_sizes,
        "GAT_hidden_node_sizes" : GAT_hidden_node_sizes,
        "encoder_hidden_sizes_D": encoder_hidden_sizes,
        "encoder_hidden_sizes_phi": encoder_hidden_sizes,
        "encoder_hidden_sizes_c": encoder_hidden_sizes,
        "encoder_hidden_sizes_alpha": encoder_hidden_sizes,
        "encoder_hidden_sizes_sinusoidal_shift": encoder_hidden_sizes_sinusoidal_shift,
        "output_mlp_hidden_sizes": output_mlp_hidden_sizes,
    }

    activation_dict = {
        "encoder_hidden_activation_D": "torch.nn.LeakyReLU(negative_slope=0.01)",
        "encoder_hidden_activation_phi": "torch.nn.LeakyReLU(negative_slope=0.01)",
        "encoder_hidden_activation_c": "torch.nn.LeakyReLU(negative_slope=0.01)",
        "encoder_hidden_activation_alpha": "torch.nn.LeakyReLU(negative_slope=0.01)",
        "encoder_hidden_activation_sinusoidal_shift": "torch.nn.LeakyReLU(negative_slope=0.01)",

        "encoder_output_activation_D": "torch.nn.Identity()",
        "encoder_output_activation_phi": "torch.nn.Identity()",
        "encoder_output_activation_c": "torch.nn.Identity()",
        "encoder_output_activation_alpha": "torch.nn.Identity()",
        "encoder_output_activation_sinusoidal_shift": "torch.nn.Identity()",

        "EConv_mlp_hidden_activation": "torch.nn.LeakyReLU(negative_slope=0.01)",
        "EConv_mlp_output_activation": "torch.nn.Identity()",

        "output_mlp_hidden_activation": "torch.nn.LeakyReLU(negative_slope=0.01)",
        "output_mlp_output_activation": "torch.nn.Identity()"
    }

    GAT_N_heads = trial.suggest_categorical('GAT_N_heads',[1,2,4,8])

    encoder_reduction = trial.suggest_categorical('encoder_reduction',['sum','mean'])
    dropout = trial.suggest_float('dropout',0,0.8,step=0.2)

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

    # construct dataset and dataloader
    dataset = MoleculeDataset_3D(root=data_dir,file_name=file_name,label_col=args.label,transform=get_mask_chiral)
    split_idx = get_split_idx(len(dataset), val_ratio=0.1, test_ratio=0.1, seed=args.seed)
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=batch_size, shuffle=True,num_workers=4)
    val_loader = DataLoader(dataset[split_idx['val']], batch_size=batch_size, shuffle=True,num_workers=4)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=batch_size, shuffle=False,num_workers=4)

    # construct model
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model = Encoder(F_z_list,F_H,dataset.atom_feat_dim, dataset.bond_feat_dim,F_H_EConv,layers_dict,activation_dict,GAT_N_heads,\
            chiral_message_passing=False,CMP_EConv_MLP_hidden_sizes=[64],CMP_GAT_N_layers=1,CMP_GAT_N_heads=2,\
            c_coefficient_normalization='sigmoid',encoder_reduction=encoder_reduction,output_concatenation_mode='both',\
            EConv_bias=True,GAT_bias=True,encoder_biases=True,dropout=dropout).to(device)


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