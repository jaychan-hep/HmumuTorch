#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import uproot
from sklearn.preprocessing import StandardScaler
import pickle

class RootDataSets(Dataset):
    def __init__(self, input_dir, tree, sigs, bkgs, variables=list(), weight="weight", random_index="eventNumber", split=None, cut=None, sig_cut=None, bkg_cut=None, normalize=None, save=True, step_size=5000000):
        if cut:
            sig_cut = cut
            bkg_cut = cut
        if split == "train":
            sig_cut = f"({sig_cut})&(({random_index}%4==0)|({random_index}%4==1))" if sig_cut else f"({random_index}%4==0)|({random_index}%4==1)"
            bkg_cut = f"({bkg_cut})&(({random_index}%4==0)|({random_index}%4==1))" if bkg_cut else f"({random_index}%4==0)|({random_index}%4==1)"
        elif split == "val":
            sig_cut = f"({sig_cut})&({random_index}%4==2)" if sig_cut else f"{random_index}%4==2)"
            bkg_cut = f"({bkg_cut})&({random_index}%4==2)" if bkg_cut else f"{random_index}%4==2)"
        elif split == "test":
            sig_cut = f"({sig_cut})&({random_index}%4==3)" if sig_cut else f"{random_index}%4==3)"
            bkg_cut = f"({bkg_cut})&({random_index}%4==3)" if bkg_cut else f"{random_index}%4==3)"
        self.save = save

        self.labels = []
        self.sig_data = pd.DataFrame()
        self.bkg_data = pd.DataFrame()
        for data in tqdm(uproot.iterate(f"{input_dir}/{sigs}/*.root:{tree}", expressions=set(variables+[weight]), cut=sig_cut, step_size=step_size, library='pd'), desc=f'[hmumuDataSets]: Loading {split if split else "full"} signals', bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}'):
            self.sig_data = self.sig_data.append(data, ignore_index=True)
        for data in tqdm(uproot.iterate(f"{input_dir}/{bkgs}/*.root:{tree}", expressions=set(variables+[weight]), cut=bkg_cut, step_size=step_size, library='pd'), desc=f'[hmumuDataSets]: Loading {split if split else "full"} backgrounds', bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}'):
            self.bkg_data = self.bkg_data.append(data, ignore_index=True)
        self.data = np.concatenate([self.sig_data[variables].to_numpy(), self.bkg_data[variables].to_numpy()])
        self.labels = [1]*len(self.sig_data) + [0]*len(self.bkg_data)
        self.sig_weights = self.sig_data[weight]*(len(self.data)/2.)/(self.sig_data[weight].sum())
        self.bkg_weights = self.bkg_data[weight]*(len(self.data)/2.)/(self.bkg_data[weight].sum()) if bkgs else np.array([])
        self.weights = np.concatenate([self.sig_weights, self.bkg_weights])

        self.normalize = normalize
        if normalize:
            self.normalize_data()

    def normalize_data(self):
        if not os.path.isfile(self.normalize):
            self.scaler = StandardScaler()
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

class RootApplyDataSets(Dataset):
    def __init__(self, input_dir, samples, tree, variables=list(), observables=list(), random_index="eventNumber", split=None, cut=None, normalize=None, save=True, step_size=5000000):
        if split == "train":
            cut = f"({cut})&(({random_index}%4==0)|({random_index}%4==1))" if cut else f"({random_index}%4==0)|({random_index}%4==1)"
        elif split == "val":
            cut = f"({cut})&({random_index}%4==2)" if cut else f"{random_index}%4==2)"
        elif split == "test":
            cut = f"({cut})&({random_index}%4==3)" if cut else f"{random_index}%4==3)"
        self.save = save
        self.extended_data = pd.DataFrame()
        for data in tqdm(uproot.iterate(f"{input_dir}/{samples}/*.root:{tree}", expressions=set(variables+observables), cut=cut, step_size=step_size, library='pd'), desc=f'[hmumuDataSets]: Loading {samples}', bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}'):
            self.extended_data = self.extended_data.append(data, ignore_index=True)
        self.data = self.extended_data[variables].to_numpy()
        
        self.normalize = normalize
        if normalize:
            self.normalize_data()

    def normalize_data(self):
        self.scaler = pickle.load(open(self.normalize, 'rb' ), encoding='latin1')
        self.data = self.scaler.transform(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        return data, 1, 1.
