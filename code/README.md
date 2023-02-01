# Code

## Description

The code implementation of ASTRAL, along with other baseline and competittive methods.

## Installation

- For ASTRAL, FedAvg, FairFL*, and FairFed*, create a new Python 3.9 environment using conda or venv then :

```bash
cd ASTRAL
pip install -r requirements.txt
```

- For FCFL create a separate Python 3.7 environment using conda or venv then :

```bash
cd FCFL
pip install -r requirements.txt
```

## Usage
- To launch an experiment, use the script ASTRAL/code/Astral.py and the json configuration file correponding to the experiment. Json files for different datasets and data distribution can be found in ASTRAL/settings/

Example: launching ASTRAL on a FL scenario consisting of: Adult, 10 clients, single the sensitive attribute considered by ASTRAL.
The command is the following:

```bash
cd ASTRAL
python ./code/ASTRAL/Astral.py settings/Adult/ASTRAL/ASTRAL-single-SA.json
```
