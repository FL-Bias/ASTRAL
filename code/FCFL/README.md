# Fair and Consistent Federated Learning

# References

The base repository comes from the official PyTorch implementation of FCFL [Fair and Consistent Federated Learning](https://github.com/cuis15/FCFL) as described in the following paper:

```
@misc{https://doi.org/10.48550/arxiv.2108.08435,
  doi = {10.48550/ARXIV.2108.08435},
  url = {https://arxiv.org/abs/2108.08435},
  author = {Cui, Sen and Pan, Weishen and Liang, Jian and Zhang, Changshui and Wang, Fei},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Addressing Algorithmic Disparity and Performance Inconsistency in Federated Learning},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Requirements

Python 3.7, the needed libraries are in requirements.txt (warning: you may need conda)
Create a new python environment Python 3.7 with pip (using conda or venv) then run the following :

```bash
pip install -r requirements.txt
```

## Running it
Part 1 and part 2 are completly optional as for reproduction purposes, as the dataset provided for FCFL are already included. (Go to Part 3)

1. If necessary, make sure to convert client configuration from ASTRAL to FCFL using:

```bash  
python DBconverter.py {name of the ASTRAL DB folder (should be in astral-datasets/version_ASTRAL)}
```

2. Create the Temp_experiments folder if necessary

3.  
    - {A} name of the folder you want to save in (will be automatically created)
    - {B} learning rate, optimal are {MajorityFair:  0.04, MajorityUnfair: 0.035, BalancedFairUnfair: 0.025}
    - {C} disparity treshold, usually in [0.01 -- 0.05]
    - {D} usually 6000, adapt for convergence
    - {E} usually 9000, adapt for convergence
    - {F} Scenario, in {MajorityFair, MajorityUnfair, BalancedFairUnfair, etc.} (you should have created it step 1)

```bash 
python main.py --target_dir_name ../../astral-experiments/Temp_experiments/{A} --step_size {B} --eps_g {C} --sensitive_attr sex --max_epoch_stage1 {D} --max_epoch_stage2 {E} --seed 1 --uniform_eps --dataset {F}```
