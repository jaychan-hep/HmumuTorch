#!/usr/bin/env python3
import os
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from tabulate import tabulate
import matplotlib.pyplot as plt
from tqdm import tqdm
from pdb import set_trace
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from myTorch import hmumuDataSets, models, loss
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', action='store', default='configs/training_config_FCN.json', help='Region to process')
    parser.add_argument('-i', '--input-dir', action='store', default='skimmed_ntuples', help='directory of training inputs')
    parser.add_argument('-o', '--output-dir', action='store', default='models', help='directory for outputs')
    parser.add_argument('-r', '--region', action='store', choices=['two_jet', 'one_jet', 'zero_jet', 'VBF', 'all_jet'], default='zero_jet', help='Region to process')
    parser.add_argument('-p', '--params', action='store', type=json.loads, default=None, help='json string.') #type=json.loads
    parser.add_argument('--save', action='store_true', help='Save model weights to HDF5 file')
    parser.add_argument('--roc', action='store_true', help='Plot ROC')
    parser.add_argument('--skopt', action='store_true', default=False, help='Run hyperparameter tuning using skopt')
    parser.add_argument('--skopt-plot', action='store_true', default=False, help='Plot skopt results')
    return parser.parse_args()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y, w) in enumerate(dataloader):
        X, y, w = X.to(device), y.to(device), w.to(device)
        # Compute prediction and loss
        pred = model(X)
        ls = loss_fn(pred, y.unsqueeze(1).double(), w.unsqueeze(1).double())

        # Backpropagation
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        if batch % 1000 == 0:
            ls, current = ls.item(), batch * len(X)
            print(f"loss: {ls:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y, w in dataloader:
            X, y, w = X.to(device), y.to(device), w.to(device)
            pred = model(X)
            test_loss += (loss_fn(pred, y.unsqueeze(1).double(), w.unsqueeze(1).double())).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():

    args=getArgs()
    with open(args.config, 'r') as stream:
        configs = json.loads(stream.read())
        config = configs[args.region]

    data = hmumuDataSets.RootDataSets(
        input_dir=args.input_dir,
        sigs=config["train_signal"],
        bkgs=config["train_background"],
        tree=config["inputTree"],
        variables=config["train_variables"],
        cut=config.get("preselections"),
        sig_cut=config.get("signal_preselections"),
        bkg_cut=config.get("background_preselections"),
        normalize=f"{args.output_dir}/std_scaler.pkl",
        save=args.save
        )

    train_data, val_data, test_data = random_split(
    	data, 
    	[int(len(data)*0.6), int(len(data)*0.2), len(data)-int(len(data)*0.6)-int(len(data)*0.2)],
    	generator=torch.Generator().manual_seed(42)
    	)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    if config["algorithm"] == "FCN":
    	model = models.FCN(len(config["train_variables"]), number_of_nodes=config.get("number_of_nodes", [64, 32, 16, 8]), dropouts=config.get("dropouts", [1])).double().to(device)

    loss_fn = loss.EventWeightedBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.0001), eps=1e-07)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(val_dataloader, model, loss_fn)
    print("Done!")

    if args.save:
        torch.save(model, f'{args.output_dir}/{config["algorithm"]}_{args.region}.pth')

    return


if __name__ == '__main__':
    main()