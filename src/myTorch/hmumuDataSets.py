#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from tqdm import tqdm
import uproot
from sklearn.preprocessing import StandardScaler
import pickle

class RootDataSets(Dataset):
    def __init__(self, input_dir, sigs, bkgs, tree, variables=None, weight="weight", cut=None, sig_cut=None, bkg_cut=None, normalize=None, save=True, step_size=5000000):
        if cut:
            sig_cut = cut
            bkg_cut = cut
        self.save = save

        self.data = pd.DataFrame()
        self.labels = []
        for data in tqdm(uproot.iterate(f"{input_dir}/{sigs}/*.root:{tree}", expressions=variables+[weight], cut=sig_cut, step_size=step_size, library='pd'), desc='[hmumuDataSets]: Loading signals', bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}'):
            self.data = self.data.append(data, ignore_index=True)
            self.labels += [1]*len(data)

        for data in tqdm(uproot.iterate(f"{input_dir}/{bkgs}/*.root:{tree}", expressions=variables+[weight], cut=bkg_cut, step_size=step_size, library='pd'), desc='[hmumuDataSets]: Loading backgrounds', bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}'):
            self.data = self.data.append(data, ignore_index=True)
            self.labels += [0]*len(data)
        self.weights = self.data[weight]
        self.data = self.data[variables].to_numpy()

        self.normalize = normalize
        if normalize:
            self.normalize_data()

    def normalize_data(self):
        if not os.path.isfile(self.normalize):
            self.scaler = StandardScaler()
            print(self.data.shape)
            self.data = self.scaler.fit_transform(self.data)
            if self.save:
                os.makedirs(os.path.dirname(self.normalize), exist_ok=True)
                with open(self.normalize, 'wb') as f:
                    pickle.dump(self.scaler, f, -1)

        else:
            self.scaler = pickle.load(open(self.normalize, 'rb' ), encoding='latin1')
            self.data = self.scaler.transform(self.data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        label = self.labels[idx]
        weight = self.weights[idx]
        return data, label, weight
