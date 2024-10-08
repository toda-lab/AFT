# AFT

This repository contains the implementation code for the **AFT** algorithm proposed in the paper 'Approximation-guided Fairness Testing through Discriminatory Space Analysis'. 

In addition, the experimental results from the paper can be accessed via the following link:  [https://doi.org/10.5281/zenodo.13898828](https://doi.org/10.5281/zenodo.13898828).
## Contents
- [Files Overview](#files-overview)
- [Requirements](#requirements)
- [Usage](#usage)

## Files Overview

- The folder [`Datasets`](Datasets): three datasets used to train the Classifier under Tests (CuTs).
- The folder [`FairnessTestCases`](FairnessTestCases): twelve trained CuTs.
- The folder [`FairnessTestMethods`](FairnessTestMethods): state-of-the-art individual fairness testing algorithms for comparison.
- The folder [`utils`](utils) and the file [`aft.py`](aft.py): the implementation code of the AFT algorithm.
- The file [`exp.py`](exp.py): the code to run experimental evaluation.

## Requirements
The experiments were based on `Python 3.8.10`.
We recommend using a virtual environment to run and test AFT,
to avoid dependency conflicts between packages.
We show two ways to create a **virtual environment**.
- If you already have `Python 3.8.10` installed, you can use `venv` to create a virtual environment.
- If not, you can
  - install `Python 3.8.10` use [`pyenv`](https://github.com/pyenv/pyenv), which enables you to install and manage multiple different versions of Python,
and then create a virtual environment using `venv`,
  - or use [`conda`](https://github.com/conda/conda) to directly create a virtual environment that already contains `Python 3.8.10`.

### Using venv
Make sure `Python 3.8.10` and `venv` are installed.

Create a virtual environment called `env_aft`:
```
python3 -m venv env_aft
```

Activate the environment:
```
source env_aft/bin/activate
```

Install required packages:
```
pip install -r requirements.txt   
```

### Using conda
Make sure `conda` is installed.

Create a virtual environment called `env_aft`, and install `Python 3.8.10` and `pip` in it:
```
conda create --name env_aft python=3.8.10 pip
```

Activate the environment:
```
conda activate env_aft
```

Install required packages:
```
pip install -r requirements.txt   
```

## Usage
Use the following command to run AFT:
```
python exp.py [--dataset_name {Adult,Credit,Bank}] [--protected_attr {sex,race,age}] [--model_name {LogReg,RanForest,DecTree,MLP}] [--method {aft,vbtx,vbt,themis}] [--runtime RUNTIME] [--repeat REPEAT]
```
The possible values for each parameter are listed below:
- dataset_name: _Adult_, _Credit_, _Bank_
- protected_attr: _sex_, _race_, _age_
- model_name: _LogReg_, _RanForest_, _DecTree_, _MLP_
- method: _aft_, _vbtx_, _vbt_, _themis_
- runtime: the running time in seconds (_default=3600_)
- repeat: the number of repeated runs (_default=30_)

The folder [`FairnessTestCases`](FairnessTestCases) contains 12 machine learning models prepared for fairness testing.
The folder [`Datasets`](Datasets) contains 3 training datasets, on which the 12 models were trained.  

### Example
As an example, the following command means to use `AFT` to perform a fairness testing on the configuration _(Adult_, _sex_, _LogReg)_ within 60 seconds:
```
python exp.py --dataset_name Adult --protected_attr sex --model_name LogReg --method aft --runtime 60 --repeat 1
```

### Outputs
Each run outputs two `.csv` files, located in two folders `DiscData` and `TestData`, respectively.
The results contained in each folder are as follows:
- [`DiscData`](DiscData): the results of detected discriminatory instances
- [`TestData`](TestData): the results of generated test cases

For the running example, suppose we obtain a result file `DiscData/aft-LogReg-Adult-sex-60-0.csv`, and the first six rows of this file are shown below:
```
$ head DiscData/aft-LogReg-Adult-sex-60-0.csv
8,2,0,14,4,3,4,2,0,1,40,61,19,0
8,2,0,14,4,3,4,2,1,1,40,61,19,1
4,1,63,15,6,2,0,0,0,17,5,10,32,0
4,1,63,15,6,2,0,0,1,17,5,10,32,1
3,2,60,5,0,13,5,1,0,5,15,15,28,0
3,2,60,5,0,13,5,1,1,5,15,15,28,1
6,6,71,10,0,7,4,0,0,1,10,15,39,0
6,6,71,10,0,7,4,0,1,1,10,15,39,1
2,3,69,2,2,1,1,2,0,2,0,84,34,0
2,3,69,2,2,1,1,2,1,2,0,84,34,1
```

For interpreting this file, 
a row represents an individual,
and a pair of two consecutive rows represents a discriminatory instance.
For example, the pair of row 1 and row 2 is a discriminatory instance, the pair of row 3 and row 4 is another discriminatory instance, and so on.

An individual (i.e., a row) is represented as a sequence of attribute values.
The order of these attributes is the same as the order of the attributes of the training data.
For example, you can find the list of attributes for _Adult_ with the following command:
```
$ head -1 Datasets/Adult.csv
age,workclass,fnlwgt,education,martial_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,Class
```