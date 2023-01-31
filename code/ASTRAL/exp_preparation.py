import pandas as pd
import time
import json
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import models
from models import *
from datasets import *
from server_client import *

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
    if (name == 'DC') | (name == 'MEPS'):
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
        return data_std
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
        return data_std
    else:
        print("Standarization not supported for the dataset.")
        return data
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
