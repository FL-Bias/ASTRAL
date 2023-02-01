# utils functions
import numpy as np 
import random
import os
from time import time
import pickle
import pdb
import json
import torch
import logging

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname
    
### ready for rewright
def concave_fun(x, delta_l, delta_g):

    def f1(x):
        n = len(x)
        dx = np.linalg.norm(x - 1. / np.sqrt(n))
        return 1 - np.exp(-dx**2)

    def f2(x):
        n = len(x)
        dx = np.linalg.norm(x + 1. / np.sqrt(n))
        return 1 - np.exp(-dx**2)

    f1_dx = grad(f1)
    f2_dx = grad(f2)    

    """
    return the function values and gradient values
    """
    return np.stack([f1(x), f2(x)]), np.stack([f1_dx(x), f2_dx(x)])

def construct_log(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    os.makedirs(args.log_dir, exist_ok = True)
    handler = logging.FileHandler(os.path.join(args.log_dir ,args.log_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler) 

    return logger

def read_data(train_data_dir, test_data_dir, validation_data_dir):
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    validation_data = {}

    if "eicu" in train_data_dir:    
        train_files = os.listdir(train_data_dir)
        train_files = [f for f in train_files if f.endswith('.npy')]
        for f in train_files:
            file_path = os.path.join(train_data_dir,f)
            cdata = np.load(file_path, allow_pickle=True).tolist()
            train_data.update(cdata['user_data'])
        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.npy')]
        for f in test_files:
            file_path = os.path.join(test_data_dir,f)
            cdata = np.load(file_path, allow_pickle=True).tolist()
            test_data.update(cdata['user_data'])        

    elif True:
        train_files = os.listdir(train_data_dir)
        train_files = [f for f in train_files if f.endswith('.json')]
        for f in train_files:
            file_path = os.path.join(train_data_dir,f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            train_data.update(cdata['user_data'])
        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.json')]
        for f in test_files:
            file_path = os.path.join(test_data_dir,f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            test_data.update(cdata['user_data'])
        validation_files = os.listdir(validation_data_dir)
        validation_files = [f for f in validation_files if f.endswith('.json')]
        for f in validation_files:
            file_path = os.path.join(validation_data_dir,f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            validation_data.update(cdata['user_data'])

    elif "health" in train_data_dir:
        train_files = os.listdir(train_data_dir)
        train_files = [f for f in train_files if f.endswith('.npy')]
        for f in train_files:
            file_path = os.path.join(train_data_dir, f)
            cdata = np.load(file_path, allow_pickle=True).tolist()
            train_data.update(cdata['user_data'])
        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.npy')]
        for f in test_files:
            file_path = os.path.join(test_data_dir, f)
            cdata = np.load(file_path, allow_pickle=True).tolist()
            test_data.update(cdata['user_data'])
    clients = list(train_data.keys())

    return clients, groups, train_data, test_data, validation_data
