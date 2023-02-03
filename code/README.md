# Code

## Description

The code implementation of ASTRAL, along with other baseline and competittive methods.

## Installation

### ASTRAL, FerFaid*, FairFL*, FedAvg
Create a new Python 3.9 environment using conda or venv then :

```bash
cd ASTRAL
pip install -r requirements.txt
```

### FCFL
Create a separate Python 3.7 environment using conda or venv then :

```bash
cd FCFL
pip install -r requirements.txt
```

## Usage
### FedAVg
To launch the baseline FedAvg, ASTRAL/Astral.py can be used.

Example: for launching FedAvg on a FL scenario consisting of: KDD, 5 clients, the command is the following:

```bash
cd ASTRAL
python Astral.py ../../settings/KDD/FedAvg/FedAvg-single-SA.json
```

### ASTRAL, FerFaid*, FairFL*

To launch an experiment applying the bias mitigation method "X", use the script ```{X}/{X}.py``` and the json configuration file correponding to the experiment. Json files for different datasets and methods can be found in ```../../settings/{dataset}/{X}/```


Example: for launching ASTRAL on a FL scenario consisting of: KDD, 5 clients, a single sensitive attribute considered by ASTRAL; the command is the following:

```bash
cd ASTRAL
python Astral.py ../../settings/KDD/ASTRAL/ASTRAL-single-SA.json
```



### FCFL
See ```FCFL/README.md```.
