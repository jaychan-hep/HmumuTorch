#!/usr/bin/env python3
import os
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
# import random
# from sklearn.preprocessing import StandardScaler, QuantileTransformer
# from tabulate import tabulate
# import matplotlib.pyplot as plt
from tqdm import tqdm
from pdb import set_trace
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from myTorch import hmumuDataSets, models, loss
from utils import metric, train
dvc = "cuda" if torch.cuda.is_available() else "cpu"
device = "gpu" if torch.cuda.is_available() else "cpu"
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


def main():

    args=getArgs()
    with open(args.config, 'r') as stream:
        configs = json.loads(stream.read())
        config = configs[args.region]

    data = hmumuDataSets.RootDataSets(
        input_dir=args.input_dir,
        sigs=config["train_signal"],
        bkgs="ttbar", #config["train_background"],
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

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=12)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=12)

    if config["algorithm"] == "FCN":
    	model = models.FCN(
            len(config["train_variables"]),
            number_of_nodes=config.get("number_of_nodes", [64, 32, 16, 8]),
            dropouts=config.get("dropouts", [1]),
            lr=config.get("learning_rate", 0.0001),
            device=dvc
        ).double().to(dvc)

    trainer = Trainer(accelerator=device, devices=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)])
    trainer.fit(model, train_dataloader, val_dataloader)

    # val_scores, val_ys, val_ws = test_loop(val_dataloader, model, loss_fn, return_output=True)

    trainer.test(dataloaders=test_dataloader)
    test_scores, test_ys, test_ws = model.test_scores, model.test_ys, model.test_ws

    signal_test_scores = test_scores[np.where(test_ys == 1)]
    train.transform_score(signal_test_scores, save=args.save, output_path=f'{args.output_dir}/{config["algorithm"]}_{args.region}_tsf.pkl')

    if args.save:
        os.makedirs(f'{args.output_dir}', exist_ok=True)
        torch.save(model, f'{args.output_dir}/{config["algorithm"]}_{args.region}.pth')

    return


if __name__ == '__main__':
    main()