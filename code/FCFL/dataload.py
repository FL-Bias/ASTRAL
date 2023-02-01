import numpy as np 
import os
import random
import pdb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils import read_data

class Federated_Dataset(Dataset):
    def __init__(self, X, Y, A):
        self.X = X
        self.Y = Y
        self.A = A

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        A = self.A[index]
        return X, Y, A 

    def __len__(self):
        return self.X.shape[0]

#### adult dataset x("51 White", "52 Asian-Pac-Islander", "53 Amer-Indian-Eskimo", "54 Other", "55 Black", "56 Female", "57 Male")
def LoadDataset(args):
    clients_name, groups, train_data, test_data, validation_data = read_data(args.train_dir, args.test_dir, args.validation_dir)
    # client_name [phd, non-phd]
    client_train_loads = []
    client_test_loads = []
    client_validation_loads = []
    args.n_clients = len(clients_name)
    name = args.warmup
    # clients_name = clients_name[:1]
    for client in clients_name:



        if (name == 'Dutch') | (name == 'MEPS') :
            sc = StandardScaler()
            X = np.array(train_data[client]["x"]).astype(np.float32)[:, :-2]
            X1 = np.array(train_data[client]["x"]).astype(np.float32)[:, -2:]
            #X1 = np.expand_dims(X1, axis=0)

            X = sc.fit_transform(X)
            X = np.concatenate([X, X1], axis=1)
            Y = np.array(train_data[client]["y"]).astype(np.float32)

        elif (name == 'Adult') | (name == 'KDD'):
            sc = StandardScaler()
            X = np.array(train_data[client]["x"]).astype(np.float32)[:, :-3]
            X1 = np.array(train_data[client]["x"]).astype(np.float32)[:, -3:]
            #X1 = np.expand_dims(X1, axis=0)

            X = sc.fit_transform(X)
            X = np.concatenate([X, X1], axis=1)
            Y = np.array(train_data[client]["y"]).astype(np.float32)

        else:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)
            print("Standarization not supported for the dataset.")

        print(X)
        print(Y)
        if args.sensitive_attr == "adult-race":
            A = 1 - X[:, -3] # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "adult-sex":
            A = 1 - X[:, -2] # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "adult-age":
            A = 1 - X[:, -1] # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]


        elif args.sensitive_attr == "MEPS-race":
            A = 1 - X[:, -2] # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "MEPS-sex":
            A = 1 - X[:, -1] # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]


        elif args.sensitive_attr == "KDD-age":
            A = 1 - X[:, -3] # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "KDD-race":
            A = 1 - X[:, -2] # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "KDD-sex":
            A = 1 - X[:, -1] # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]

        elif args.sensitive_attr == "RACE": #for meps
            A = 1 - X[:, -1] # [1: privileged, 0: non-privilegedmale]
            args.n_feats = X.shape[1]

        elif args.sensitive_attr == "race":
            A = 1 - X[:, -1] # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        
        elif args.sensitive_attr == "sex":
            A = 1 - X[:, -1] # [0: female, 1: male]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "age":
            A = 1 - X[:, -1] # [0: outside 30-60, 1: 30-60]
            args.n_feats = X.shape[1]

        else:
            print("error sensitive attr")
            exit()
        dataset = Federated_Dataset(X, Y, A)
        client_train_loads.append(DataLoader(dataset, X.shape[0],
        shuffle = args.shuffle,
        num_workers = args.num_workers,
        pin_memory = True,
        drop_last = args.drop_last))

    for client in clients_name:
        if (name == 'Dutch') | (name == 'MEPS'):
            sc = StandardScaler()
            X = np.array(test_data[client]["x"]).astype(np.float32)[:, :-2]
            X1 = np.array(test_data[client]["x"]).astype(np.float32)[:, -2:]
            #X1 = np.expand_dims(X1, axis=0)

            X = sc.fit_transform(X)
            X = np.concatenate([X, X1], axis=1)
            Y = np.array(test_data[client]["y"]).astype(np.float32)

        elif (name == 'Adult') | (name == 'KDD'):
            sc = StandardScaler()
            X = np.array(test_data[client]["x"]).astype(np.float32)[:, :-3]
            X1 = np.array(test_data[client]["x"]).astype(np.float32)[:, -3:]
            #X1 = np.expand_dims(X1, axis=0)

            X = sc.fit_transform(X)
            X = np.concatenate([X, X1], axis=1)
            Y = np.array(test_data[client]["y"]).astype(np.float32)

        else:
            X = np.array(test_data[client]["x"]).astype(np.float32)

            Y = np.array(test_data[client]["y"]).astype(np.float32)
            print("Standarization not supported for the dataset.")

        if args.sensitive_attr == "adult-race":
            A = 1 - X[:, -3]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "adult-sex":
            A = 1 - X[:, -2]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "adult-age":
            A = 1 - X[:, -1]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]


        elif args.sensitive_attr == "MEPS-race":
            A = 1 - X[:, -2]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "MEPS-sex":
            A = 1 - X[:, -1]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]


        elif args.sensitive_attr == "KDD-age":
            A = 1 - X[:, -3]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "KDD-race":
            A = 1 - X[:, -2]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "KDD-sex":
            A = 1 - X[:, -1]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]

        elif args.sensitive_attr == "RACE":  # for meps
            A = 1 - X[:, -1]  # [1: privileged, 0: non-privilegedmale]
            args.n_feats = X.shape[1]

        elif args.sensitive_attr == "race":
            A = 1 - X[:, -1]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]

        elif args.sensitive_attr == "sex":
            A = 1 - X[:, -1]  # [0: female, 1: male]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "age":
            A = 1 - X[:, -1]  # [0: outside 30-60, 1: 30-60]
            args.n_feats = X.shape[1]

        else:
            print("error sensitive attr")
            exit()

        dataset = Federated_Dataset(X, Y, A)
        client_test_loads.append(DataLoader(dataset, X.shape[0],
        shuffle = args.shuffle,
        num_workers = args.num_workers,
        pin_memory = True,
        drop_last = args.drop_last)) 


    for client in clients_name:

        if (name == 'Dutch') | (name == 'MEPS'):
            sc = StandardScaler()
            X = np.array(validation_data[client]["x"]).astype(np.float32)[:, :-2]
            X1 = np.array(validation_data[client]["x"]).astype(np.float32)[:, -2:]
            #X1 = np.expand_dims(X1, axis=0)

            X = sc.fit_transform(X)
            X = np.concatenate([X, X1], axis=1)

            Y = np.array(validation_data[client]["y"]).astype(np.float32)

        elif (name == 'Adult') | (name == 'KDD'):
            sc = StandardScaler()
            X = np.array(validation_data[client]["x"]).astype(np.float32)[:, :-3]
            X1 = np.array(validation_data[client]["x"]).astype(np.float32)[:, -3:]
            #X1 = np.expand_dims(X1, axis=0)

            X = sc.fit_transform(X)
            X = np.concatenate([X, X1], axis=1)

            Y = np.array(validation_data[client]["y"]).astype(np.float32)

        else:
            X = np.array(validation_data[client]["x"]).astype(np.float32)


            Y = np.array(validation_data[client]["y"]).astype(np.float32)
            print("Standarization not supported for the dataset.")
        if args.sensitive_attr == "adult-race":
            A = 1 - X[:, -3]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "adult-sex":
            A = 1 - X[:, -2]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "adult-age":
            A = 1 - X[:, -1]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]


        elif args.sensitive_attr == "MEPS-race":
            A = 1 - X[:, -2]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "MEPS-sex":
            A = 1 - X[:, -1]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]


        elif args.sensitive_attr == "KDD-age":
            A = 1 - X[:, -3]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "KDD-race":
            A = 1 - X[:, -2]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "KDD-sex":
            A = 1 - X[:, -1]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]

        elif args.sensitive_attr == "RACE":  # for meps
            A = 1 - X[:, -1]  # [1: privileged, 0: non-privilegedmale]
            args.n_feats = X.shape[1]

        elif args.sensitive_attr == "race":
            A = 1 - X[:, -1]  # [0: non-privileged, 1: privileged]
            args.n_feats = X.shape[1]

        elif args.sensitive_attr == "sex":
            A = 1 - X[:, -1]  # [0: female, 1: male]
            args.n_feats = X.shape[1]
        elif args.sensitive_attr == "age":
            A = 1 - X[:, -1]  # [0: outside 30-60, 1: 30-60]
            args.n_feats = X.shape[1]

        else:
            print("error sensitive attr")
            exit()


        dataset = Federated_Dataset(X, Y, A)
        client_validation_loads.append(DataLoader(dataset, X.shape[0],
        shuffle = args.shuffle,
        num_workers = args.num_workers,
        pin_memory = True,
        drop_last = args.drop_last))

    return client_train_loads, client_test_loads, client_validation_loads
