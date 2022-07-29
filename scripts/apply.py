#!/usr/bin/env python3
import os, sys
from argparse import ArgumentParser
import json
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer
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
    data = hmumuDataSets.ApplyRootDataSets(
        input_dir=args.input_dir,
        samples=sample,
        variables=variables,
        **apply_config["dataset_settings"],
        normalize=None
    )
    for model in apply_config["models"]:
        train_dataset_settings = {key: value for key, value in train_configs[model]["dataset_settings"].items()
            if key not in {"sigs", "bkgs", "tree", "cut", "sig_cut", "bkg_cut", "weight"}}
        apply_data = getattr(hmumuDataSets, f'Apply{train_configs[model]["dataset_class"]}')(
            input_dir=args.input_dir,
            samples=sample,
            **train_dataset_settings,
            **apply_config["dataset_settings"],
            normalize=f"{args.model_dir}/std_scaler_{model}.pkl",
        )
        dataloader = DataLoader(apply_data, batch_size=64, shuffle=False, num_workers=12)

        md = getattr(models, train_configs[model]["algorithm"])(
            **train_configs[model]["params"],
            lr=train_configs[model].get("learning_rate", 0.0001),
            device=dvc
        ).double().to(dvc)

        trainer = Trainer(accelerator=device, devices=1)
        trainer.test(model=md, dataloaders=dataloader, ckpt_path=f'{args.model_dir}/{train_configs[model]["algorithm"]}_{model}/test.ckpt', verbose=False)
        scores, ys, ws = md.test_scores, md.test_ys, md.test_ws

        scores_t = train.transform_score(scores, f'{args.model_dir}/{train_configs[model]["algorithm"]}_{model}_tsf.pkl')
        data.extended_data[apply_config["models"][model]] = scores
        data.extended_data[apply_config["models"][model] + "_t"] = scores_t
    with uproot.recreate(f"{output_subdir}/{sample}.root") as f:
        f["test"] = data.extended_data


def main():

    args=getArgs()

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
        variables = variables|set(train_configs[model]["dataset_settings"]["variables"])
    variables = list(variables)
    # print(variables)

    output_subdir = f"{args.output_dir}/{args.region}"
    os.makedirs(output_subdir, exist_ok=True)

    if os.path.isdir(output_subdir) and len(os.listdir(output_subdir)) > 0:
        print("ERROR: Output directory not empty!! Please delete or move the directory.")
        sys.exit(1)

    for sample in sample_list:
        if args.samples and sample not in args.samples: continue
        apply(sample, train_configs, apply_config, args, variables, output_subdir)

    return


if __name__ == '__main__':
    main()