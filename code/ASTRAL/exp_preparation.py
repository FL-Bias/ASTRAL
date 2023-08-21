import pandas as pd
import time
import pickle
import json
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import models
import numpy as np
from models import *
from datasets import *
from server_client import *
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import CelebA
from torchvision.transforms import transforms
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter


def get_train_files(repo, nb_clients):
    filepaths = []

    files = [f for f in listdir(repo) if isfile(join(repo, f))]
    print(files)
    if len(files) != nb_clients:
        print("The total number of trainsets is not equal to the number of clients")
    else:
        for i in range(nb_clients):
            filepaths.append(join(repo, files[i]))
        print(filepaths)
    return filepaths

#def normalize_col(s):
#    for idx, line in enumerate(s):
#        s[idx] = line / sum(line) * 5000
#    return np.cumsum(np.around(s, 0), axis=1)

def normalize(s, sensitive_att, data):
    size = list(data[sensitive_att].value_counts())
    for idx, line in enumerate(s):
        s[idx] = line / sum(line) * size[idx]
    return np.cumsum(np.around(s, 0), axis=1)



def standarize_data2(sensitive_att, target_var, data):
    data_with_SA = data.copy()
    cols = data.columns.tolist()
    cols.remove(target_var)
    x_0 = data_with_SA.iloc[:, :-1]
    y_0 = data_with_SA.iloc[:, -1]
    sc = StandardScaler()
    x_0 = sc.fit_transform(x_0)
    data_std = pd.DataFrame(data=x_0, columns=cols, index=data.index)
    data_std[target_var] = y_0
    return data_std

def standarize_data3(sensitive_att, target_var, data):
    data_without_SA = data.drop(sensitive_att, 1)
    cols = data.columns.tolist()
    if isinstance(sensitive_att,list):
        for elem in sensitive_att:
            cols.remove(elem)
    else:
        cols.remove(sensitive_att)
    cols.remove(target_var)
    y_0 = data_without_SA.iloc[:, -1]
    categorical, numerical = [], []
    for c in cols:
        if len(data_without_SA[c].unique()) <= 2:
            print(c)
            print(data_without_SA[c].unique())
            categorical.append(c)
        else :
            numerical.append(c)
    x_0 = data_without_SA[numerical]
    sc = StandardScaler()
    x_0 = sc.fit_transform(x_0)
    data_std = pd.DataFrame(data=x_0, columns=numerical, index=data.index)
    print(data_std.columns)
    data_std = pd.concat([data_std, data_without_SA[categorical]], axis = 1)
    print(data_std.columns)
    data_std[sensitive_att] = data[sensitive_att]
    data_std[target_var] = y_0
    return data_std

def standarize_data_old(sensitive_att, target_var, data):
    data_without_SA = data.drop(sensitive_att, 1)
    cols = data.columns.tolist()
    if isinstance(sensitive_att,list):
        for elem in sensitive_att:
            cols.remove(elem)
    else:
        cols.remove(sensitive_att)
    cols.remove(target_var)
    x_0 = data_without_SA.iloc[:, :-1]
    y_0 = data_without_SA.iloc[:, -1]
    sc = StandardScaler()
    x_0 = sc.fit_transform(x_0)
    data_std = pd.DataFrame(data=x_0, columns=cols, index=data.index)
    data_std[sensitive_att] = data[sensitive_att]
    data_std[target_var] = y_0
    return data_std


def standarize_data(sensitive_att, target_var, data, name ):
    if (name == 'OPSD') | (name == 'AI4I'):

        data_SA = data.iloc[:, -2:]
        data_SA = data_SA.iloc[:, 0:1]

        data_without_SA = data.iloc[:, :-2]
        cols = data.columns.tolist()

        cols = cols[:-2]

        x_0 = data_without_SA.iloc[:, :]
        y_0 = data.iloc[:, -1]
        sc = StandardScaler()

        x_0 = sc.fit_transform(x_0)
        data_std = pd.DataFrame(data=x_0, columns=cols, index=data.index)
        data_std = pd.concat([data_std, data_SA], axis=1)
        data_std[target_var] = y_0


    elif (name == 'DC') | (name == 'MEPS') | (name == 'MobiAct'):
        data_SA = data.iloc[:, -3:]
        data_SA = data_SA.iloc[:, 0:2]

        data_without_SA = data.iloc[:, :-3]
        cols = data.columns.tolist()

        cols = cols[:-3]

        x_0 = data_without_SA.iloc[:, :]
        y_0 = data.iloc[:, -1]
        sc = StandardScaler()

        x_0 = sc.fit_transform(x_0)
        data_std = pd.DataFrame(data=x_0, columns=cols, index=data.index)
        data_std = pd.concat([data_std, data_SA], axis=1)
        data_std[target_var] = y_0

    elif (name == "Adult") | (name == 'KDD'):
        data_SA = data.iloc[:, -4:]
        data_SA = data_SA.iloc[:, 0:3]

        data_without_SA = data.iloc[:, :-4]
        cols = data.columns.tolist()

        cols = cols[:-4]

        x_0 = data_without_SA.iloc[:, :]
        y_0 = data.iloc[:, -1]
        sc = StandardScaler()

        x_0 = sc.fit_transform(x_0)
        data_std = pd.DataFrame(data=x_0, columns=cols, index=data.index)
        data_std = pd.concat([data_std, data_SA], axis=1)
        data_std[target_var] = y_0
        print(data_std.iloc[:,-6:])

    else:
        print("Standarization not supported for the dataset.")
        return data
    print(data_std)
    return(data_std)

def find_idx(ligne, index):
    for idx, e in enumerate(ligne):
        if index <= sum(ligne[0:int(idx) + 1]):
            break
    return idx

def somme(matrix, idx):
    tab = 0
    for i in matrix[0:idx]:
        tab = tab + i
    return tab

def get_distribution_index(alpha, dataset, nb_user, sensitive_att, nb_class):
    indiv_list = []
    for goal in range(0, nb_class):
        list_1 = [idx for idx, x in enumerate(dataset[sensitive_att]) if x == goal]
        indiv_list.append(list_1)
    np.random.seed(9)
    s = normalize(np.random.dirichlet([alpha] * nb_class, nb_user).transpose(), sensitive_att, dataset)
    data_list_transfer = []
    for user in range(0, nb_user):
        if user == 0:
            bound_1 = 0
            bound_2 = int(s[0][user])
            tmp = indiv_list[0][bound_1:bound_2]
            data_list_transfer.append(tmp)
        else:
            bound_1 = int(s[0][user - 1])
            bound_2 = int(s[0][user])
            tmp = indiv_list[0][bound_1:bound_2]
            data_list_transfer.append(tmp)
        for class_ in range(1, 2):
            if user == 0:
                bound_1 = 0
                bound_2 = int(s[class_][user])
                tmp = indiv_list[class_][bound_1:bound_2]
                data_list_transfer[user] = data_list_transfer[user] + tmp
            else:
                bound_1 = int(s[class_][user - 1])
                bound_2 = int(s[class_][user])
                tmp = indiv_list[class_][bound_1:bound_2]
                data_list_transfer[user] = data_list_transfer[user] + tmp
    dict_users = {}
    for i in range(nb_user):
        dict_users[i] = set(data_list_transfer[i])
    return dict_users

def generate_client_data_dirichelet(alpha, df, nb_clients, sensitive_att, nb_class):
    dfs = []
    dist = get_distribution_index(alpha, df, nb_clients, sensitive_att, nb_class)
    for i in range(len(dist)):
        dfs.append(df.iloc[list(dist[i]), :])
    for i in range(len(dfs)):
        if dfs[i].size == 0:
            print("the number of generated datasets is less than the number of clients")
    return dfs

def get_warmup_data(nb_users, data_preparation_info, data_info):
    warm_up_path = data_info["warmup_train_sets_path"]
    sensitive_att_name = data_info["sensitive_att_name"]
    target_var_name = data_info["target_var_name"]
    nb_classes = data_info["nb_classes"]
    nb_features = data_info["nb_features"]
    standarize = data_info["standardization"]
    name = data_info["name"]
    train_sets = []
    df = get_train_files(warm_up_path, nb_users)
    for i in range(len(df)):
        train = pd.read_csv(df[i])
        train_sets.append(train)
    if standarize:
        for j in range(len(train_sets)):
            train_sets[j] = standarize_data(sensitive_att_name, target_var_name, train_sets[j], name)
    training_sets = [MyDataset(train_sets[i], sensitive_att_name, target_var_name, nb_features, nb_classes) for i in range(nb_users)]
    
    return training_sets



def filter_celeba_by_indices(dataset,indices):
    filtered_dataset = copy.deepcopy(dataset)
    filtered_dataset.attr = torch.stack([filtered_dataset.attr[i] for i in indices])
    filtered_dataset.bbox = torch.stack([filtered_dataset.bbox[i] for i in indices])
    filtered_dataset.filename = [filtered_dataset.filename[i] for i in indices]
    filtered_dataset.identity = torch.stack([filtered_dataset.identity[i] for i in indices])
    filtered_dataset.landmarks_align = torch.stack([filtered_dataset.landmarks_align[i] for i in indices])

    return filtered_dataset

def distribute_celeba_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(25)
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def normalize_col(s, sa):
    # Calculate the count of each unique value in the list
    counts = Counter(sa)

    # Convert the counts to a list
    size = list(counts.values())
    print(size)
    print(len(size))

    for idx, line in enumerate(s):
        print(idx, line)

    for idx, line in enumerate(s):
        s[idx] = line / sum(line) * size[idx]
    return np.cumsum(np.around(s, 0), axis=1)


def distribute_celeba_non_iid(alpha, target_, sa_, nb_user=10):
    target = [1 if item == 1 else 0 for item in target_]
    sa = [1 if item == 1 else 0 for item in sa_]

    
    indiv_list = []

    for goal in range(0, 2):
        list_1 = [idx for idx, x in enumerate(sa) if x == goal]
        indiv_list.append(list_1)

    nb_class = 2

    s = normalize_col(np.random.dirichlet([alpha] * nb_class, nb_user).transpose(), sa)
    data_list_transfer = []
    for user in range(0, nb_user):
        if user == 0:
            bound_1 = 0
            bound_2 = int(s[0][user])
            tmp = indiv_list[0][bound_1:bound_2]
            data_list_transfer.append(tmp)
        else:
            bound_1 = int(s[0][user - 1])
            bound_2 = int(s[0][user])
            tmp = indiv_list[0][bound_1:bound_2]
            data_list_transfer.append(tmp)

        for class_ in range(1,2): #range(1, 10):
            if user == 0:
                bound_1 = 0
                bound_2 = int(s[class_][user])
                tmp = indiv_list[class_][bound_1:bound_2]
                data_list_transfer[user] = data_list_transfer[user] + tmp
            else:
                bound_1 = int(s[class_][user - 1])
                bound_2 = int(s[class_][user])
                tmp = indiv_list[class_][bound_1:bound_2]
                data_list_transfer[user] = data_list_transfer[user] + tmp

    dict_users = {}
    for i in range(nb_user):
        dict_users[i] = set(data_list_transfer[i])
    return dict_users



def distribute_celeba_non_iid_4_classes(alpha, target_, sa_, nb_user=10):
    print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    target = [1 if item == 1 else 0 for item in target_]
    sa = [1 if item == 1 else 0 for item in sa_]

    sa_target = [str(a) + str(b) for a, b in zip(sa, target)]

    # Define the mapping dictionary
    mapping_dict = {
        '01': 1,
        '10': 2,
        '11': 3,
        '00': 0
    }

    # Use list comprehension to apply the mapping
    mapped_list = [mapping_dict[s] for s in sa_target]

    #print(mapped_list)
    indiv_list = []

    for goal in range(0, 4):
        list_1 = [idx for idx, x in enumerate(mapped_list) if x == goal]
        indiv_list.append(list_1)

    nb_class = 4

    s = normalize_col(np.random.dirichlet([alpha] * nb_class, nb_user).transpose(), mapped_list)
    data_list_transfer = []
    for user in range(0, nb_user):
        if user == 0:
            bound_1 = 0
            bound_2 = int(s[0][user])
            tmp = indiv_list[0][bound_1:bound_2]
            data_list_transfer.append(tmp)
        else:
            bound_1 = int(s[0][user - 1])
            bound_2 = int(s[0][user])
            tmp = indiv_list[0][bound_1:bound_2]
            data_list_transfer.append(tmp)

        for class_ in range(1,4): #range(1, 10):
            if user == 0:
                bound_1 = 0
                bound_2 = int(s[class_][user])
                tmp = indiv_list[class_][bound_1:bound_2]
                data_list_transfer[user] = data_list_transfer[user] + tmp
            else:
                bound_1 = int(s[class_][user - 1])
                bound_2 = int(s[class_][user])
                tmp = indiv_list[class_][bound_1:bound_2]
                data_list_transfer[user] = data_list_transfer[user] + tmp

    dict_users = {}
    for i in range(nb_user):
        dict_users[i] = set(data_list_transfer[i])
    return dict_users



def distribute_celeba_non_iid_personalized(dataset, num_users):
    """
    Sample non-I.I.D. client data from CelebA dataset based on gender and smile attributes
    :param dataset: CelebA dataset
    :param num_users: Number of clients (should be 4 in this case)
    :return: dict of image index for each user
    """
    target_gender = dataset.attr[:, 20].numpy()  # Assuming the 21st attribute represents gender
    target_smile = dataset.attr[:, 31].numpy()  # Assuming the 32nd attribute represents smile

    # Indices of male and female samples
    male_idxs = np.where(target_gender == 1)[0]
    female_idxs = np.where(target_gender == -1)[0]

    # Indices of smiling and not smiling samples
    smile_idxs = np.where(target_smile == 1)[0]  # Assuming 1 represents smiling
    not_smile_idxs = np.where(target_smile == -1)[0]  # Assuming -1 represents not smiling

    # Create non-IID distribution among clients
    user_indices = {}
    np.random.seed(25)

    # Two clients with majority male and smiling samples, and majority female and not smiling samples
    male_smile_user = np.random.choice(male_idxs, len(male_idxs) // 2, replace=False)
    female_not_smile_user = np.random.choice(female_idxs, len(female_idxs) // 2, replace=False)
    user_indices[0] = set(np.concatenate((male_smile_user, female_not_smile_user)))

    # Two clients with majority female and smiling samples, and majority male and not smiling samples
    female_smile_user = np.random.choice(female_idxs, len(female_idxs) // 2, replace=False)
    male_not_smile_user = np.random.choice(male_idxs, len(male_idxs) // 2, replace=False)
    user_indices[1] = set(np.concatenate((female_smile_user, male_not_smile_user)))

    # Exclude the chosen samples from the available pool for the next clients
    male_remaining = np.setdiff1d(male_idxs, list(user_indices[0].union(user_indices[1])))
    female_remaining = np.setdiff1d(female_idxs, list(user_indices[0].union(user_indices[1])))
    smile_remaining = np.setdiff1d(smile_idxs, list(user_indices[0].union(user_indices[1])))
    not_smile_remaining = np.setdiff1d(not_smile_idxs, list(user_indices[0].union(user_indices[1])))

    # Two clients with majority male and not smiling samples, and majority female and smiling samples
    male_not_smile_user = np.random.choice(male_remaining, len(male_remaining) // 2, replace=False)
    female_smile_user = np.random.choice(female_remaining, len(female_remaining) // 2, replace=False)
    user_indices[2] = set(np.concatenate((male_not_smile_user, female_smile_user)))

    # The remaining samples are assigned to the last client
    user_indices[3] = set(np.arange(len(dataset))) - user_indices[0].union(user_indices[1], user_indices[2])

    return user_indices

def get_celeba_data(data_info, nb_clients):

    
    
    class DatasetSplit(Dataset):
        """An abstract Dataset class wrapped around Pytorch Dataset class.
        """

        def __init__(self, dataset, idxs):
            self.dataset = dataset
            self.idxs = [int(i) for i in idxs]

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, item):
        
            image, label = self.dataset[self.idxs[item]]
            return torch.tensor(image), torch.tensor(label)

    random.seed(25)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Step 1: Read the data
    train_dataset = CelebA(data_info['celeba_path'],
                                                    download=False,
                                                    split='train',
                                                    transform=transform,
                                                    target_type="attr")


#    train_dataset = filter_celeba_by_indices(train_dataset, random.sample(list(range(len(train_dataset))), 8000))

    # get stratified sample of celeba
    # Get the attribute labels (targets) from the CelebA dataset
    #print(dir(train_dataset))
    #targets = train_dataset.attr[:,31]

    # Define the number of samples you want in the subset
    #subset_size = 8000

    # Create a StratifiedShuffleSplit to perform stratified sampling based on the attribute labels
    #sss = StratifiedShuffleSplit(n_splits=1, test_size=subset_size, random_state=42)

    # Get the indices of the samples in the stratified subset
    #_, subset_indices = next(sss.split(range(len(train_dataset)), targets))

    # Create a Subset dataset for the stratified subset
    #train_dataset = Subset(train_dataset, subset_indices)


    # distribute data on clients based on age
    distrib = data_info['distribution']


    if distrib == 'iid':
        clients_idx = distribute_celeba_iid(train_dataset, nb_clients)
    elif distrib == 'non-iid':
        column, sa = [], []
        for i in range(len(train_dataset)):
            sample, target = train_dataset[i]
            column.append(int(target[31]))
            sa.append(int(target[20]))

        alpha = data_info['alpha']
        if alpha ==1:
            f = open(data_info['celeba_path']+'/non_iid_gender_alpha1.pkl', 'rb')
            clients_idx = pickle.load(f)
            print('read config from ',data_info['celeba_path']+'/non_iid_gender_alpha1.pkl')
            f.close()
        else:
            print('Call Dirichlet process')
            clients_idx = distribute_celeba_non_iid_4_classes(alpha, column, sa, nb_clients)
            with open(data_info['celeba_path']+'/non_iid_gender_alpha'+str(alpha)+'.pkl', 'wb') as f:
                pickle.dump(clients_idx, f)
            print('saved config at ',data_info['celeba_path']+'/non_iid_gender_alpha'+str(alpha)+'.json')
            f.close()
        #clients_idx = distribute_celeba_non_iid_personalized(train_dataset, nb_clients)
    else:
        print('not implemented distribution')


    print('start')
    for k in clients_idx.keys():
        print(k,len(clients_idx[k]))



    clients_datasets = [DatasetSplit(train_dataset, clients_idx[c]) for c in range(nb_clients)]



    for c in range(nb_clients):
        data = []
        images_list = []
        client_dataset = clients_datasets[c]
        print(len(client_dataset))
        for i in range(len(client_dataset)):
            sample, target = client_dataset[i]
            sensitive_attribute_male = int(target[20] == 1)  # Sensitive attribute 'Male' from CelebA dataset
            sensitive_attribute_young = int(target[39] == 1)  # Sensitive attribute 'Young' from CelebA dataset
            target_attribute = int(target[31] == 1)  # Attribute 'Smiling' from CelebA dataset
            data.append([sensitive_attribute_male, sensitive_attribute_young, target_attribute])
            images_list.append(sample)


        df = pd.DataFrame(data, columns=['gender', 'age', 'smiling'])
        print(df)
        # Step 3: Instantiate the MyDataset class
        sensitive_att_name = ['gender', 'age']
        target_var_name = 'smiling'
        nb_features = 32*32*3+2
        nb_classes = 2

        images = torch.stack(images_list)

        print('=======================clients stats '+str(c)+' =====================================')
        nb_smile, nb_grin, nb_old, nb_young, nb_female, nb_male =len(df[df.smiling==1]),len(df[df.smiling!=1]),len(df[df.age!=1]),len(df[df.age==1]), len(df[df.gender!=1]),len(df[df.gender==1])
        print('size: {}  nb similing: {} nb_grin: {}  nb young: {} nb old: {}  nb female: {}  nb male: {}'.format(len(df),nb_smile, nb_grin, nb_old, nb_young, nb_female, nb_male))

#        df.to_csv("astral-datasets/CELEBA/FL_train/train/trainClient"+str(c)+".csv", index=False)

        clients_datasets[c] = MyImageDataset(images, df, sensitive_att_name, target_var_name, nb_features, nb_classes)

    print('nb clients '+str(len(clients_datasets)))

    valid_dataset = CelebA(data_info['celeba_path'],
                                                    download=False,
                                                    split='test',
                                                    transform=transform,
                                                    target_type="attr")


#    test_dataset = filter_celeba_by_indices(test_dataset, random.sample(list(range(len(test_dataset))), 2000))

#    idxs = list(range(len(test_dataset)))
#    print('idxs')

#    random.shuffle(idxs)
#    idx_valid = idxs[int(0.5 * len(idxs)):]
#    idx_test = idxs[:int(0.5 * len(idxs))]
    


#    valid_dataset = filter_celeba_by_indices(test_dataset, idx_valid)
#    test_dataset = filter_celeba_by_indices(test_dataset, idx_test)


    data = []
    images_list = []
    for i in range(len(valid_dataset)):
        sample, target = valid_dataset[i]
        sensitive_attribute_male = int(target[20] == 1)  # Sensitive attribute 'Male' from CelebA dataset
        sensitive_attribute_young = int(target[39] == 1)  # Sensitive attribute 'Young' from CelebA dataset
        target_attribute = int(target[31] == 1)  # Attribute 'Smiling' from CelebA dataset
        data.append([sensitive_attribute_male, sensitive_attribute_young, target_attribute])
        images_list.append(sample)


    df = pd.DataFrame(data, columns=['gender', 'age', 'smiling'])
    print(df)
    nb_smile, nb_grin, nb_old, nb_young, nb_female, nb_male =len(df[df.smiling==1]),len(df[df.smiling!=1]),len(df[df.age!=1]),len(df[df.age==1]), len(df[df.gender!=1]),len(df[df.gender==1])
    print('valid size: {}  nb similing: {} nb_grin: {}  nb young: {} nb old: {}  nb female: {}  nb male: {}'.format(len(df),nb_smile, nb_grin, nb_old, nb_young, nb_female, nb_male))
#    df.to_csv("astral-datasets/CELEBA/FL_train/valid.csv", index=False)
    images = torch.stack(images_list)
    valid_dataset = MyImageDataset(images, df, sensitive_att_name, target_var_name, nb_features, nb_classes)


    test_dataset = CelebA(data_info['celeba_path'],
                                                    download=False,
                                                    split='valid',
                                                    transform=transform,
                                                    target_type="attr")



    data = []
    images_list = []
    for i in range(len(test_dataset)):
        sample, target = test_dataset[i]
        sensitive_attribute_male = int(target[20] == 1)  # Sensitive attribute 'Male' from CelebA dataset
        sensitive_attribute_young = int(target[39] == 1)  # Sensitive attribute 'Young' from CelebA dataset
        target_attribute = int(target[31] == 1)  # Attribute 'Smiling' from CelebA dataset
        data.append([sensitive_attribute_male, sensitive_attribute_young, target_attribute])
        images_list.append(sample)


    df = pd.DataFrame(data, columns=['gender', 'age', 'smiling'])
    print(df)
    nb_smile, nb_grin, nb_old, nb_young, nb_female, nb_male =len(df[df.smiling==1]),len(df[df.smiling!=1]),len(df[df.age!=1]),len(df[df.age==1]), len(df[df.gender!=1]),len(df[df.gender==1])
    print('test size: {}  nb similing: {} nb_grin: {}  nb young: {} nb old: {}  nb female: {}  nb male: {}'.format(len(df),nb_smile, nb_grin, nb_old, nb_young, nb_female, nb_male))
#    df.to_csv("astral-datasets/CELEBA/FL_train/test.csv", index=False)
    images = torch.stack(images_list)
    test_dataset = MyImageDataset(images, df, sensitive_att_name, target_var_name, nb_features, nb_classes)

    return clients_datasets, test_dataset, valid_dataset

def get_train_data(nb_users, data_preparation_info, data_info):
    initial_preparation = data_preparation_info["initial_data_preparation"]
    try:
        data_dist_function = data_preparation_info["data_distribution_function"]["name"]
        if data_dist_function == "dirichlet":
            alpha = data_preparation_info["data_distribution_function"]["parameter"]
        else:
            alpha = None
        save_generated_data = data_preparation_info["save_generated_datasets"]
        if save_generated_data:
            saving_path = data_preparation_info["saving_path"]
        else:
            saving_path = None
    except:
        print('No data preparation required.')
    train_sets_path = data_info["clients_train_sets_path"]
    test_set_path = data_info["test_set_path"]
    sensitive_att_name = data_info["sensitive_att_name"]
    target_var_name = data_info["target_var_name"]
    nb_classes = data_info["nb_classes"]
    nb_features = data_info["nb_features"]
    standarize = data_info["standardization"]
    name = data_info["name"]
    train_sets = []
    if not initial_preparation:
        df = get_train_files(train_sets_path, nb_users)
        for i in range(len(df)):
            train = pd.read_csv(df[i])
            train_sets.append(train)
    else:
        if data_dist_function == "dirichlet":
            df = pd.read_csv(train_sets_path)
            train_sets = generate_client_data_dirichelet(alpha, df, nb_users, sensitive_att_name, nb_classes)
            if save_generated_data:
                for j in range(len(train_sets)):
                    filename = "data_party_" + str(j) + ".csv"
                    train_sets[j].to_csv(join(saving_path, filename), index=False)
    if standarize:
        for j in range(len(train_sets)):
            if len(train_sets[j]) != 0:
                train_sets[j] = standarize_data(sensitive_att_name, target_var_name, train_sets[j], name)
    training_sets = [MyDataset(train_sets[i], sensitive_att_name, target_var_name, nb_features, nb_classes) if len(train_sets[i]) else -1 for i in range(nb_users)]
    return training_sets

def get_validation(data_info):
    sensitive_att_name = data_info["sensitive_att_name"]
    target_var_name = data_info["target_var_name"]
    nb_classes = data_info["nb_classes"]
    nb_features = data_info["nb_features"]
    standarize = data_info["standardization"]
    name = data_info["name"]
    try:
        valid_set_path = data_info["validation_set_path"]
        validset = pd.read_csv(valid_set_path)
        if (standarize) :
            validset = standarize_data(sensitive_att_name, target_var_name, validset, name)
        validation_dataset = MyDataset(validset, sensitive_att_name, target_var_name, nb_features, nb_classes)
    except:
        validation_dataset = None
    return validation_dataset

def get_test(data_info):
    test_set_path = data_info["test_set_path"]
    sensitive_att_name = data_info["sensitive_att_name"]
    target_var_name = data_info["target_var_name"]
    nb_classes = data_info["nb_classes"]
    nb_features = data_info["nb_features"]
    standarize = data_info["standardization"]
    testset = pd.read_csv(test_set_path)
    name = data_info["name"]
    if (standarize) :
        testset = standarize_data(sensitive_att_name, target_var_name, testset, name)
    testing_dataset = MyDataset(testset, sensitive_att_name, target_var_name, nb_features, nb_classes)
    return testing_dataset

def get_bias_test(data_info):
    bias_test_set_path = data_info["bias_test_set_path"]
    sensitive_att_name = data_info["sensitive_att_name"]
    target_var_name = data_info["target_var_name"]
    nb_classes = data_info["nb_classes"]
    nb_features = data_info["nb_features"]
    standarize = data_info["standardization"]
    biastestset = pd.read_csv(bias_test_set_path)
    name = data_info["name"]
    if standarize:
        biastestset = standarize_data(sensitive_att_name, target_var_name, biastestset, name)
    bias_testing_dataset = MyDataset(biastestset, sensitive_att_name, target_var_name, nb_features, nb_classes)
    return bias_testing_dataset

def prepare_data(nb_users, data_preparation_info, data_info, FL_parameters):
    if data_info['name'] == 'CELEBA':
        training_sets, validation_dataset, testing_dataset = get_celeba_data(data_info, nb_users)
        warmup_sets = None
    else:
        if FL_parameters["warmup"] == 2:
            training_sets = get_train_data(nb_users-FL_parameters["nb_clients_warmup"], data_preparation_info, data_info)
            warmup_sets= get_warmup_data(FL_parameters["nb_clients_warmup"], data_preparation_info, data_info)
        else:
            training_sets = get_train_data(nb_users, data_preparation_info, data_info)
            warmup_sets = None
        validation_dataset = get_validation(data_info)
        testing_dataset = get_test(data_info)
    return training_sets, validation_dataset, testing_dataset, warmup_sets

def prepare_model(learning_info, FL_parameters):
    model_class = learning_info['model_class']
    model_class = getattr(models, model_class)
    model = model_class(learning_info, FL_parameters["warmup"])

    model = model.model
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    print('model nb params', sum(p.numel() for p in model.parameters()))
    return model

def get_config(repository):
    configs = []
    if os.path.isfile(repository):
        config_files_paths = [repository]
    else :
        config_files_paths = [join(repository, f) for f in listdir(repository) if isfile(join(repository, f))]
    np.random.seed(42)
    for i in range(len(config_files_paths)):
        file = config_files_paths[i]
        print(file)
        with open(file) as f:
            parameters = json.load(f)
            FL_parameters = parameters['fl_parameters']
            data_preparation_info = parameters['fl_data_preparation_info']
            learning_info = parameters['learning_info']
            data_info = parameters['data_info']
            remark = parameters['remark']
            try:
                bias_mitigation_info = parameters['bias_mitigation_info']
                bias_mitigation_info['apply'] = 1
            except:
                print('No bias mitigation required')
                bias_mitigation_info = {'apply': 0}
            print(bias_mitigation_info)
            configs.append((FL_parameters, data_preparation_info, learning_info, data_info, bias_mitigation_info, remark))
            print('Configs has been read')
    return configs

def run_FL(FL_parameters, train_sets, validation_set, test_set, FL_model, learning_info, data_preparation_info, data_info, remark):
    nb_runs = FL_parameters["nb_runs"]
    aggregation = FL_parameters["aggregation_method"]
    nb_users = FL_parameters["nb_clients"]
    nb_rounds = FL_parameters["nb_rounds"]
    ratio_clients = FL_parameters["ratio_of_participating_clients"]
    client_selection_method = FL_parameters["participants_selection_method"]
    metrics_path = FL_parameters["metrics_collection_path"]
    seed = 42
    print(nb_rounds)
    for run in range(nb_runs):
        train_loss, local_test_accuracy, local_test_loss, local_test_metrics, global_test_accuracy, global_test_loss, global_test_metrics = [], [], [], [], [], [], []
        weights_list = []
        metrics_list = []
        clients = None
        server = Server(validation_set, test_set,  FL_model)
        global_model = copy.deepcopy(FL_model)
        for r_ound in range(nb_rounds):
            round_strt_time = time.time()
            if r_ound == 0:
                clients = [Client(train_sets[i], FL_model, learning_info) for i in range(nb_users)]
            else:
                # send global model of last round to the clients
                for i in range(nb_users):
                    clients[i].set_model_weights(global_model.get_weights())
            print('===================================================')
            print('Round ' + str(r_ound) + ' global model')
            print('===================================================')
            try:
                print(global_model.get_weights())
            except:
                print('Model non initialized')

            local_weights_list = []
            # running local training on all clients
            clients_idx = []
            for i in range(nb_users):
                # getting size of client (i) training sets to use it later in the averaging
                clients_idx.append(clients[i].train_dataset.__len__())
                # local training
                print('===================================================')
                print('client ' + str(i))
                print('Before local training')
                try:
                    print(clients[i].get_model_weights())
                except:
                    print('Model non initialized')
                local_model, loss = clients[i].local_training(global_model)
                print('After local training')
                print(local_model.get_weights())
                print('===================================================')
                # saving local model weight and train loss of client (i)               
                local_weights_list.append(local_model.get_weights())
                weights_list.append([run, r_ound, i, 'local model', local_model.get_weights()])
                metrics_list.append([run, r_ound, i, 'local model training loss', -1, loss, -1, -1])
            print(clients_idx)
            # update global weights using weighted averaging (currently only weighted average is supported)
            global_model = server.weighted_average_weights(local_weights_list, clients_idx, aggregation)
            round_end_time = (time.time() - round_strt_time)
            print('After aggregation')
            print('===================================================')
            print(global_model.get_weights())
            weights_list.append([run, r_ound, i, 'global model', global_model.get_weights()])
            for i in range(nb_users):
                test_data = server.get_testing_dataset()
                local_test_acc, local_test_loss, local_test_metrics = clients[i].test_inference(test_data)
                print(i)
                print(local_test_acc, local_test_loss, local_test_metrics)
                metrics_list.append([run, r_ound, i, 'local model tested on test set', local_test_acc, local_test_loss, local_test_metrics, -1])
            print(global_model.get_weights())
            try:
                for i in range(nb_users):
                    validation_data = server.get_validation_dataset()
                    local_valid_acc, local_valid_loss, local_valid_metrics = clients[i].test_inference(validation_data)
                    print(i)
                    print(local_valid_acc, local_valid_loss, local_valid_metrics)
                    metrics_list.append(
                        [run, r_ound, i, 'local model tested on validation set', local_valid_acc, local_valid_loss, local_valid_metrics, -1])
                global_valid_acc, global_valid_loss, global_valid_metrics = server.validation()
                metrics_list.append([run, r_ound, -1, 'global model tested on validation set', global_valid_acc, global_valid_loss, global_valid_metrics, round_end_time])
                print('global')
                print(global_valid_acc, global_valid_loss, global_valid_metrics)
            except:
                print('No validation set provided.')
            print(global_model.get_weights())
            global_test_acc, global_test_loss, global_test_metrics = server.test_inference()
            metrics_list.append([run, r_ound, -1, 'global model tested on test set', global_test_acc, global_test_loss, global_test_metrics, round_end_time])
            print('global')
            print(global_test_acc, global_test_loss, global_test_metrics)
        print("| Run {} ---- Test Accuracy: {:.2f}% --- Test Loss: {:.2f} --- Metrics {}".format(run, 100 * global_test_acc, global_test_loss, global_test_metrics))
        columns = ['rund_id', 'round_id', 'client_id', 'info', 'accuracy', 'loss', 'metrics', 'round_train_time']
        metrics_df = pd.DataFrame(data=metrics_list, columns=columns)
        for param in FL_parameters, learning_info, data_preparation_info, data_info:
            for k in param.keys():
                try:
                    metrics_df[k] = param[k]
                except:
                    metrics_df[k] = json.dumps(param[k])
        metrics_df['remark'] = remark
        columns = ['rund_id', 'round_id', 'client_id', 'info', 'weights']
        weights_df = pd.DataFrame(data=weights_list, columns=columns)
        timestamp = time.strftime("%b_%d_%Y_%H_%M_%S")
        filename = "collected_metrics-" + "run" + str(run) + "-" + timestamp + ".csv"
        metrics_df.to_csv(join(metrics_path, filename), index=False)
        filename = "learned_weights-" + "run" + str(run) + "-" + timestamp + ".csv"
        weights_df.to_csv(join(metrics_path, filename), index=False)
    return 0

def prepare_data_ML(data_info):
    sensitive_att_name = data_info["sensitive_att_name"]
    target_var_name = data_info["target_var_name"]
    nb_classes = data_info["nb_classes"]
    nb_features = data_info["nb_features"]
    standarize = data_info["standardization"]
    train_set_path = data_info["train_set_path"]
    name = data_info["name"]
    train_set = pd.read_csv(train_set_path, header = 0)
    if standarize:
        train_set = standarize_data(sensitive_att_name, target_var_name, train_set, name)
    training_set = MyDataset(train_set, sensitive_att_name, target_var_name, nb_features, nb_classes)
    try:
        valid_set_path = data_info["validation_set_path"]
        validset = pd.read_csv(valid_set_path)
        if standarize:
            validset = standarize_data(sensitive_att_name, target_var_name, validset, name)
        validation_dataset = MyDataset(validset, sensitive_att_name, target_var_name, nb_features, nb_classes)
    except:
        validation_dataset = None
    test_set_path = data_info["test_set_path"]
    testset = pd.read_csv(test_set_path)
    if standarize:
        testset = standarize_data(sensitive_att_name, target_var_name, testset, name)
    testing_dataset = MyDataset(testset, sensitive_att_name, target_var_name, nb_features, nb_classes)

    return training_set, validation_dataset, testing_dataset

def get_config_ML(repository):
    configs = []
    config_files_paths = [f for f in listdir(repository) if isfile(join(repository, f))]
    np.random.seed(42)
    for i in range(len(config_files_paths)):
        info = []
        file = os.path.join(repository, config_files_paths[i])
        print(file)
        with open(file) as f:
            parameters = json.load(f)
            ML_parameters = parameters['ml_parameters']
            learning_info = parameters['learning_info']
            data_info = parameters['data_info']
            remark = parameters['remark']
            configs.append((ML_parameters, learning_info, data_info, remark))
            print('Configs has been read')
    return configs
