# Welcome to HmumuTorch

This is a package to train pytorch model and evaluate the number significance for the H->mumu analysis

## Setup

This project requires to use python3 and either conda or virtual environment. If you want to use a conda environment, install Anaconda or Miniconda before setting up. Below provides instructioon for setting up the virtual environment on lxplus with pip.

### First time setup on lxplus

First checkout the code:

```
git clone set-url gitlab ssh://git@gitlab.cern.ch:7999/wisc_atlas/HmumuTorch.git [-b your_branch]
cd HmumuTorch
```

Then,

```
source scripts/install.sh
```

### Normal setup on lxplus

After setting up the environment for the first time, you can return to this setup by doing `source scripts/setup.sh`

## Scripts to run the tasks

### Prepare the training inputs (skimmed ntuples)

The core script is `skim_ntuples.py`. The script will apply the given skimming cuts to input samples and produce corresponding skimmed ntuples. You can run it locally for any specified input files. In H->mumu, it is more convenient and time-efficient to use `submit_skim_ntuples.py` which will find all of the input files specified in `data/inputs_config.json` and run the core script by submitting the condor jobs.

- The script is hard-coded currently, meaning one needs to directly modify the core script to change the variable calculation and the skimming cuts.
- The output files (skimmed ntuples) will be saved to the folder named `skimmed_ntuples` by default.
- `submit_skim_ntuples.py` will only submit the jobs for those files that don't exist in the output folder.

```
python scripts/submit_skim_ntuples.py
```
or
```
python scripts/skim_ntuples.py [-i input_file_path] [-o output_file_path]
```

### Check the completeness of the jobs or recover the incomplete jobs

The script `run_skim_ntuples.py` will check if all of the outputs from last step exist and if the resulting number of events is correct.

```
python scripts/run_skim_ntuples.py [-c]

usage:
  -c   check the completeness only (without recovering the incomplete jobs)
  -s   skip checking the resulting number of events
``` 

Please be sure to do `python run_skim_ntuples.py -c` at least once before starting the BDT training exercise.


### Start pytorch analysis!

The whole ML task consists of training, applying the weights, optimizing the score boundaries for categorization, and calculating the number counting significances.

#### Training a model

The training script `train.py` will train the model, and transform the output scores such that the unweighted signal distribution is flat. The detailed settings, including the preselections, training variables, hyperparameters, etc, are specified in the config file such as `configs/training_config_FCN.json`.

```
python scripts/train.py --help
usage: train.py [-h] [-c CONFIG] [-i INPUT_DIR] [-o OUTPUT_DIR]
                [-r {two_jet,one_jet,zero_jet,VBF,all_jet}]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Region to process
  -i INPUT_DIR, --input-dir INPUT_DIR
                        directory of training inputs
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        directory for outputs
  -r {two_jet,one_jet,zero_jet,VBF,all_jet}, --region {two_jet,one_jet,zero_jet,VBF,all_jet}
                        Region to process
```

#### Applying the weights

Applying the trained model (as well as the score transformation) to the skimmed ntuples to get BDT scores for each event can be done by doing:
```
python scripts/apply.py --help
usage: apply.py [-h] [-ic INPUTCONFIG] [-c CONFIG CONFIG] [-i INPUT_DIR]
                [-m MODEL_DIR] [-o OUTPUT_DIR]
                [-r {two_jet,one_jet,zero_jet,all_jet}]
                [-s SAMPLES [SAMPLES ...]]

optional arguments:
  -h, --help            show this help message and exit
  -ic INPUTCONFIG, --inputConfig INPUTCONFIG
                        Input config
  -c CONFIG CONFIG, --config CONFIG CONFIG
                        training config + apply config
  -i INPUT_DIR, --input-dir INPUT_DIR
                        directory of training inputs
  -m MODEL_DIR, --model-dir MODEL_DIR
                        directory of NN models
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        directory for outputs
  -r {two_jet,one_jet,zero_jet,all_jet}, --region {two_jet,one_jet,zero_jet,all_jet}
                        Region to process
  -s SAMPLES [SAMPLES ...], --samples SAMPLES [SAMPLES ...]
                        apply only for specific samples
```
The script will take the settings specified in the training config file and the applying config file such as `configs/apply_config.json`.

#### Optimizing the score boundaries

`categorization_1D.py` will take the Higgs classifier scores of the samples and optimize the boundaries that give the best combined significance. `categorization_2D.py`, on the other hand, takes both the Higgs classifier scores and the VBF classifier scores of the samples and optimizes the 2D boundaries that give the best combined significance.

```
python categorization_1D.py [-r REGION] [-f NUMBER OF FOLDS] [-b NUMBER OF CATEGORIES] [-n NSCAN] [--floatB] [--minN minN] [--skip]

Usage:
  -f, --fold          Number of folds of the categorization optimization. Default is 1.
  -b, --nbin          Number of score categories
  -n, --nscan         Number of scans. Default is 100
  --minN,             minN is the minimum number of events required in the mass window. The default is 5.
  --floatB            To float the last score boundary, which means to veto the lowest score score events
  --skip              To skip the hadd step (if you have already merged signal and background samples)
```

```
python categorization_2D.py [-r REGION] [-f NUMBER OF FOLDS] [-b NUMBER OF CATEGORIES] [-b NUMBER OF ggF CATEGORIES] [-n NSCAN] [--floatB] [--minN minN] [--skip]

Usage:
  -f, --fold          Number of folds of the categorization optimization. Default is 1.
  -b, --nbin          Number of score categories
  -n, --nscan         Number of scans. Default is 100
  --minN,             minN is the minimum number of events required in the mass window. The default is 5.
  --floatB            To float the last score boundary, which means to veto the lowest BDT score events
  --skip              To skip the hadd step (if you have already merged signal and background samples)
```
