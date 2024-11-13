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
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import models
import config
import utils as utils
from utils import RegLoss

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_data(train_ratio):
    train_data, test_data, all_data_num = utils.load_data(train_ratio)
    dataHandler = utils.DataHandler()
    dataHandler.align_Data(train_data, Training=True)
    dataHandler.align_Data(test_data, Training=False)
    return train_data, test_data, dataHandler

def create_model(user_num, item_num, cont_num, args):
    model_map = {
        'COATF': models.COATF,
        'CoSTCo': models.CoSTCo
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
        loss = loss_function(prediction, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_model(model, test_loader, loss_function, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = model(batch_x[:, 0], batch_x[:, 1], batch_x[:, 2])
            loss = loss_function(prediction, batch_y)
            total_loss += loss.item()

            mae = torch.abs(prediction - batch_y).mean().item()
            total_mae += mae

            rmse = torch.sqrt(torch.pow(prediction - batch_y, 2).mean()).item()
            total_rmse += rmse

            total_count += 1

    avg_mae = total_mae / total_count
    avg_rmse = total_rmse / total_count
    avg_loss = total_loss / len(test_loader)

    return avg_loss, avg_mae, avg_rmse

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    train_data, test_data, dataHandler = prepare_data(args.train_ratio)
    user_num, item_num, cont_num = dataHandler.userIDMapper.getNumIDs(), dataHandler.itemIDMapper.getNumIDs(), dataHandler.contIDMapper.getNumIDs()
    train_dataset = TensorDataset(torch.tensor(dataHandler.trainData_x).long(), torch.tensor(dataHandler.trainData_y).float())
    test_dataset = TensorDataset(torch.tensor(dataHandler.testData_x).long(), torch.tensor(dataHandler.testData_y).float())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = create_model(user_num, item_num, cont_num, args).to(device)
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    best_mae = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_model(model, train_loader, optimizer, loss_function, device)
        test_loss, test_mae, test_rmse = test_model(model, test_loader, loss_function, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")
        if test_mae < best_mae:
            best_mae = test_mae
            if args.out:
                torch.save(model.state_dict(), f"{config.model_path}/{args.model}_{args.factor_num}_{best_mae:.4f}.pt")
    print(f"Best Test MAE: {best_mae:.4f}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learn rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="training epoches")
    parser.add_argument("--factor_num", '-r', type=int, default=30, help="predictive factors numbers in the model")
    parser.add_argument("--out", type=bool, default=True, help="is or not output model")
    parser.add_argument("--reg", type=float, default=0.01, help="L2 regularization reg")
    parser.add_argument("--model", type=str, default="COATF", help="which of models")
    parser.add_argument("--optim", type=str, default="Adam", help="optimizer:[Adam, SGD]")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--train_ratio", default=0.9, help="seed")
    parser.add_argument("--gpu", type=str, default=0)
    parser.add_argument("--nonlinear", type=str, default=None)
    args = parser.parse_args()
    main(args)