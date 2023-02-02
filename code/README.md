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
### ASTRAL, FerFaid*, FairFL*, FedAvg
To launch an experiment, use the script ```ASTRAL/Astral.py``` and the json configuration file correponding to the experiment. Json files for different experiments can be found in ```../../settings/```

Example: for launching ASTRAL on a FL scenario consisting of: Adult, 10 clients, a single sensitive attribute considered by ASTRAL; the command is the following:

```bash
cd ASTRAL
python Astral.py ../../settings/Adult/ASTRAL/ASTRAL-single-SA.json
```
### FCFL
See ``FCFL/README.md```.
