#!/usr/bin/env python3
import os, sys
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from myTorch import hmumuDataSets, models, loss
import uproot
from utils import metric, train
dvc = "cuda" if torch.cuda.is_available() else "cpu"
device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser()
    parser.add_argument('-ic', '--inputConfig', action='store', default='configs/inputs_config.json', help='Input config')
    parser.add_argument('-c', '--config', action='store', nargs=2, default=['configs/training_config_FCN.json', 'configs/apply_config.json'], help='training config + apply config')
    parser.add_argument('-i', '--input-dir', action='store', default='skimmed_ntuples', help='directory of training inputs')
    parser.add_argument('-m', '--model-dir', action='store', default='models', help='directory of NN models')
    parser.add_argument('-o', '--output-dir', action='store', default='outputs', help='directory for outputs')
    parser.add_argument('-r', '--region', action='store', choices=['two_jet', 'one_jet', 'zero_jet', 'all_jet'], default='zero_jet', help='Region to process')
    parser.add_argument('-s', '--samples', action='store', nargs='+', help='apply only for specific samples')
    return parser.parse_args()


def apply(sample, train_configs, apply_config, args, variables, output_subdir):
    data = hmumuDataSets.RootApplyDataSets(
        input_dir=args.input_dir,
        samples=sample,
        tree=args.region,
        variables=variables,
        observables=apply_config.get("observables", list()),
        cut=apply_config.get("preselections"),
        normalize=None
    )
    for model in apply_config["models"]:
        apply_data = hmumuDataSets.RootApplyDataSets(
            input_dir=args.input_dir,
            samples=sample,
            tree=args.region,
            variables=train_configs[model]["train_variables"],
            cut=apply_config.get("preselections"),
            normalize=f"{args.model_dir}/std_scaler_{model}.pkl",
        )
        dataloader = DataLoader(apply_data, batch_size=64, shuffle=False, num_workers=12)

        if train_configs[model]["algorithm"] == "FCN":
            md = models.FCN(
                len(train_configs[model]["train_variables"]),
                number_of_nodes=train_configs[model].get("number_of_nodes", [64, 32, 16, 8]),
                dropouts=train_configs[model].get("dropouts", [1]),
                lr=train_configs[model].get("learning_rate", 0.0001),
                device=dvc
            ).double().to(dvc)

        trainer = Trainer(accelerator=device, devices=1)
        trainer.test(model=md, dataloaders=dataloader, ckpt_path=f'{args.model_dir}/{train_configs[model]["algorithm"]}_{model}/test.ckpt')
        scores, ys, ws = md.test_scores, md.test_ys, md.test_ws

        scores_t = train.transform_score(scores, f'{args.model_dir}/{train_configs[model]["algorithm"]}_{args.region}_tsf.pkl')
        data.extended_data[apply_config["models"][model]] = scores
        data.extended_data[apply_config["models"][model] + "_t"] = scores_t
    with uproot.recreate(f"{output_subdir}/{sample}.root") as f:
        f["test"] = data.extended_data


def main():

    args=getArgs()
    if os.path.isdir(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        print("ERROR: Output directory not empty!! Please delete or move the directory.")
        sys.exit(1)

    with open(args.inputConfig) as f:
        config = json.load(f)
    sample_list = config['sample_list']
    with open(args.config[0], 'r') as stream:
        train_configs = json.loads(stream.read())
    with open(args.config[1], 'r') as stream:
        configs = json.loads(stream.read())
        apply_config = configs[args.region]

    variables = set()
    for model in apply_config["models"]:
        variables = variables|set(train_configs[model]["train_variables"])
    variables = list(variables)
    print(variables)

    output_subdir = f"{args.output_dir}/{args.region}"
    os.makedirs(output_subdir, exist_ok=True)

    for sample in sample_list:
        if args.samples and sample not in args.samples: continue
        apply(sample, train_configs, apply_config, args, variables, output_subdir)

    return


if __name__ == '__main__':
    main()