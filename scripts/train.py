#!/usr/bin/env python3
import os, sys
from argparse import ArgumentParser
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from myTorch import hmumuDataSets, models, loss
from utils import train, plotting
dvc = "cuda" if torch.cuda.is_available() else "cpu"
device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', action='store', default='configs/training_config_FCN.json', help='Region to process')
    parser.add_argument('-i', '--input-dir', action='store', default='skimmed_ntuples', help='directory of training inputs')
    parser.add_argument('-o', '--output-dir', action='store', default='outputs/models', help='directory for outputs')
    parser.add_argument('-r', '--region', action='store', choices=['two_jet', 'one_jet', 'zero_jet', 'VBF', 'all_jet'], default='zero_jet', help='Region to process')
    parser.add_argument('--inspect', action='store_true', help='Inspect the trained model (skip the training part)')
    parser.add_argument('--resume', action='store_true', help='Resume the training')
    # parser.add_argument('-p', '--params', action='store', type=json.loads, default=None, help='json string.') #type=json.loads
    # parser.add_argument('--skopt', action='store_true', default=False, help='Run hyperparameter tuning using skopt')
    # parser.add_argument('--skopt-plot', action='store_true', default=False, help='Plot skopt results')
    return parser.parse_args()

def main():

    args=getArgs()
    with open(args.config, 'r') as stream:
        configs = json.loads(stream.read())
        config = configs[args.region]
    if not args.inspect and not args.resume and os.path.isdir(f'{args.output_dir}/{config["algorithm"]}_{args.region}') and len(f'{args.output_dir}/{config["algorithm"]}_{args.region}') > 0:
        print("ERROR: Output directory not empty!! Please delete or move the directory.")
        sys.exit(1)

    train_data, val_data, test_data = hmumuDataSets.load_data_train_val_test_split(
        dataset_class=config.get("dataset_class", "RootDataSets"),
        input_dir=args.input_dir,
        **config["dataset_settings"],
        normalize=f"{args.output_dir}/std_scaler_{args.region}.pkl"
        )

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=12)
    train_for_test_dataloader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=12)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=12)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=12)

    model = getattr(models, config["algorithm"])(
        **config["params"],
        # optimizer=config["optimizer"],
        # optimizer_params=config["optimizer_params"],
        device=dvc
    ).double().to(dvc)
    print(model)

    callbacks = [EarlyStopping(**config["early_stopping"])]
    callbacks.append(ModelCheckpoint(
        save_top_k=1,
        monitor=config["early_stopping"]["monitor"],
        mode=config["early_stopping"]["mode"],
        dirpath=f'{args.output_dir}/{config["algorithm"]}_{args.region}',
        filename='best',
    ))

    trainer = Trainer(accelerator=device, devices=1, callbacks=callbacks, default_root_dir=f'{args.output_dir}/{config["algorithm"]}_{args.region}')
    if not args.inspect:
        if args.resume:
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=f'{args.output_dir}/{config["algorithm"]}_{args.region}/best.ckpt')
        else:
            trainer.fit(model, train_dataloader, val_dataloader)

    print("Done.")
    print("===========================================================================================================")
    print("                                      Evaluating whole training set")
    print("===========================================================================================================")
    trainer.test(model=model, dataloaders=train_for_test_dataloader, ckpt_path=f'{args.output_dir}/{config["algorithm"]}_{args.region}/best.ckpt')
    train_scores, train_ys, train_ws = model.test_scores, model.test_ys, model.test_ws

    print("===========================================================================================================")
    print("                                     Evaluating whole validation set")
    print("===========================================================================================================")
    trainer.test(model=model, dataloaders=val_dataloader, ckpt_path=f'{args.output_dir}/{config["algorithm"]}_{args.region}/best.ckpt')
    val_scores, val_ys, val_ws = model.test_scores, model.test_ys, model.test_ws

    print("===========================================================================================================")
    print("                                       Evaluating whole test set")
    print("===========================================================================================================")
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=f'{args.output_dir}/{config["algorithm"]}_{args.region}/best.ckpt')
    test_scores, test_ys, test_ws = model.test_scores, model.test_ys, model.test_ws

    plotting.plotROC(test_scores, test_ys, test_ws, train_scores, train_ys, train_ws, val_scores, val_ys, val_ws, outname=f'{args.output_dir}/plots/{config["algorithm"]}_{args.region}_roc.pdf', save=True)

    plotting.plot_score(test_scores, test_ys, test_ws, outname=f'{args.output_dir}/plots/{config["algorithm"]}_{args.region}_scores.pdf', save=True)

    signal_test_scores = test_scores[np.where(test_ys == 1)]
    signal_test_scores_t, tsf = train.fit_and_transform_score(signal_test_scores, save=True, output_path=f'{args.output_dir}/{config["algorithm"]}_{args.region}_tsf.pkl')

    test_scores_t = train.transform_score(test_scores, tsf=tsf)
    plotting.plot_score(test_scores_t, test_ys, test_ws, outname=f'{args.output_dir}/plots/{config["algorithm"]}_{args.region}_scores_t.pdf', save=True)

    return

if __name__ == '__main__':
    main()