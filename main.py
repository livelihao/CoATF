#coding=UTF-8
import os
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from torch.optim import SGD, Adam
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, Subset

from tqdm import tqdm
from sklearn.model_selection import KFold

import models
import config
import utils as utils
from utils import RegLoss
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=2024):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True    
        torch.backends.cudnn.benchmark = False       
        torch.backends.cudnn.enabled = False         
    
    # 额外设置避免NVIDIA库的随机性
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

def prepare_data(train_ratio):
    train_data, test_data, all_data_num = utils.load_data(train_ratio)
    dataHandler = utils.DataHandler()
    dataHandler.align_Data(train_data, Training=True)
    dataHandler.align_Data(test_data, Training=False)
    return train_data, test_data, dataHandler

def create_model(user_num, item_num, cont_num, args):
    model_map = {
        'COATF': models.COATF,
        # 'CoSTCo': models.CoSTCo
    }
    model = model_map[args.model](user_num, item_num, cont_num, factor_num=args.factor_num, dropout=args.dropout, nonlinear=args.nonlinear)
    return model

def train_model(model, train_loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    for batch_x, batch_y in tqdm(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        model.zero_grad()
        prediction = model(batch_x[:, 0], batch_x[:, 1], batch_x[:, 2])
        data_loss = loss_function(prediction, batch_y)
        loss = data_loss + args.reg * RegLoss(model)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_model(model, test_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    total_squared_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = model(batch_x[:, 0], batch_x[:, 1], batch_x[:, 2])
            batch_size = batch_y.size(0)
            total_samples += batch_size

            loss = loss_function(prediction, batch_y)
            total_loss += loss.item() * batch_size  

            abs_errors = torch.abs(prediction - batch_y)
            total_abs_error += abs_errors.sum().item()

            squared_errors = (prediction - batch_y) ** 2
            total_squared_error += squared_errors.sum().item()

    avg_loss = total_loss / total_samples  
    mae = total_abs_error / total_samples
    rmse = (total_squared_error / total_samples) ** 0.5

    return avg_loss, mae, rmse


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    
    full_data, test_data, dataHandler = prepare_data(1.0)  
    user_num, item_num, cont_num = dataHandler.userIDMapper.getNumIDs(), dataHandler.itemIDMapper.getNumIDs(), dataHandler.contIDMapper.getNumIDs()

    full_dataset = TensorDataset(
        torch.tensor(dataHandler.trainData_x).long(),
        torch.tensor(dataHandler.trainData_y).float()
    )
    
    kfold = 5 
    kfold = KFold(n_splits=kfold, shuffle=True, random_state=args.seed)
    fold_results = {'mae': [], 'rmse': []}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n=== Fold {fold+1}/{kfold} ===")
        

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed)
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        model = create_model(user_num, item_num, cont_num, args).to(device)
        loss_function = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
        
        best_mae = float('inf')
        for epoch in range(args.epochs):
            train_loss = train_model(model, train_loader, optimizer, loss_function, device)
            val_loss, val_mae, val_rmse = test_model(model, val_loader, loss_function, device)
            
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
            
            # early stopping
            if val_mae < best_mae:
                best_mae = val_mae
                best_rmse = val_rmse
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    print("Early stopping")
                    break
        
        fold_results['mae'].append(best_mae)
        fold_results['rmse'].append(best_rmse)
        print(f"Fold {fold+1} Best - MAE: {best_mae:.4f}, RMSE: {best_rmse:.4f}")
    
    print("\n=== Cross Validation Results ===")
    print(f"MAE: {np.mean(fold_results['mae']):.4f} ± {np.std(fold_results['mae']):.4f}")
    print(f"RMSE: {np.mean(fold_results['rmse']):.4f} ± {np.std(fold_results['rmse']):.4f}")
    
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learn rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="training epoches")
    parser.add_argument("--factor_num", '-r', type=int, default=30, help="predictive factors numbers in the model")
    parser.add_argument("--out", type=bool, default=True, help="is or not output model")
    parser.add_argument("--reg", type=float, default=0.01, help="L2 regularization reg")
    parser.add_argument("--model", type=str, default="COATF", help="which of models")
    parser.add_argument("--optim", type=str, default="Adam", help="optimizer:[Adam, SGD]")
    parser.add_argument("--seed", type=int, default=2024, help="seed")
    parser.add_argument("--train_ratio", default=0.9, help="seed")
    parser.add_argument("--gpu", type=str, default=0)
    parser.add_argument("--nonlinear", type=str, default=None)
    args = parser.parse_args()
    main(args)