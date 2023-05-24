import os
import pandas as pd
import time
import sys
import json
import warnings
warnings.filterwarnings('ignore')
from datasets import MyDataset
from typing import List, Tuple
from server_client import *
import models

class FLBase:
    FL_parameters: dict = {}
    validation_set: dict = {}
    test_set: dict = {}
    warmup_sets: dict = {}
    FL_model: FLModel = None
    learning_info: dict = {}
    data_preparation_info: dict = {}
    data_info: dict = {}
    BiasMitigation_info: dict = {}
    remark: list = []
    clients: List[Client] = []
    server: Server = None
    metrics_list: List = []
    global_model: FLModel = None
    weights_list: List = []
    global_test_acc: float = 0.00
    train_sets: List[MyDataset] = []

    def __init__(self, data: dict, *a, **kw):
        self.__dict__.update(data)
        self.metrics_list: List = []
        self.weights_list: List = []

    def run(self):
        nb_clients = self.FL_parameters["nb_clients"]
        nb_runs = self.FL_parameters["nb_runs"]
        try:
            self.cdp = bool(self.FL_parameters["cdp"])
        except:
            self.cdp = False
        if self.cdp:
            self.cdp_eps = self.FL_parameters["cdp_eps"]
            self.cdp_clip = self.FL_parameters["cdp_clip"]
        if self.FL_parameters["warmup"] == 2:
            nb_warmups = self.FL_parameters["nb_clients_warmup"]
        for i in range(nb_runs):
            self.global_model: FLModel = copy.deepcopy(self.FL_model)
            self.server = Server(nb_clients, self.validation_set, self.test_set, self.FL_model)
            if self.FL_parameters["warmup"] == 2:
                self.clients = [
                    Client(self.warmup_sets[i], self.FL_model, self.learning_info, )
                    for i in range(nb_warmups)
                ]
                self.run_rounds_warmup(run=i)
                self.clients.extend( [Client(self.train_sets[i], self.FL_model, self.learning_info, ) for i in range(nb_clients-nb_warmups)])
                self.run_rounds(run=i)
            else :
                self.clients = [Client(self.train_sets[i], self.FL_model, self.learning_info, ) for i in range(nb_clients)]
                self.run_rounds(run=i)
        self.write_metrics_df(i)

    def run_rounds(self, run: int, *a, **kw):
        nb_rounds = self.FL_parameters["nb_rounds"]
        global_test_acc, global_test_loss, global_test_metrics = self.run_global_evaluation(run, -1, 0, 0,0, 0)
        for round_ in range(nb_rounds):
            round_strt_time = time.time()
            downstream = 0
            upstream = 0
            for i in range(len(self.clients)):
                self.clients[i].set_model_weights(self.global_model.get_weights())
                # measuring downstream data
                downstream = downstream + len(self.global_model.get_weights())
            print('===================================================')
            print('Round ' + str(round_) + ' global model')
            print('===================================================')
            try:
                print(self.global_model.get_weights())
            except:
                print('Model non initialized')
            # running local training on all clients
            local_weights_list: List = []
            _, _, clients_idx = self.run_local_training(local_weights_list, round_=round_, run=run)
            _, _, _ = self.run_local_evaluation(run, round_)
            #print(clients_idx)
            #measuring upstream data
            upstream = upstream + len(local_weights_list) * len(local_weights_list[0])
            #aggregating the global model
            self.global_model, server_time = self.server.weighted_average_weights(local_weights_list, clients_idx, self.FL_parameters["aggregation_method"], self.learning_info['byzantine'] )
            if self.cdp:
                print('================Before CDP===================================')
                print(self.global_model.get_weights())
                self.global_model = self.server.cdp(self.global_model, self.cdp_eps, sum(clients_idx),self.learning_info['batch_size'],self.learning_info['nb_epochs'], nb_rounds, clip = self.cdp_clip)
                print('================After CDP ===================================')
                print(self.global_model.get_weights())
            round_end_time = (time.time() - round_strt_time)
            print('After aggregation')
            print('===================================================')
            print(self.global_model.get_weights())
            self.weights_list.append([run, round_, -1, 'global model', self.global_model.get_weights()])
            # getting global test accuracies, global test losses, global test metrics
            global_test_acc, global_test_loss, global_test_metrics = self.run_global_evaluation(run, round_, round_end_time, server_time, upstream, downstream)
            print(f"| Run {run} ---- Test Accuracy: {100 * global_test_acc:.2f}% --- Test Loss: \ {global_test_loss:.2f} --- Metrics {global_test_metrics}")

    def run_rounds_warmup(self, run: int, *a, **kw):
        nb_rounds = self.FL_parameters["nb_rounds_warmup"]
        for round_ in range(-nb_rounds, 0):
            round_strt_time = time.time()
            upstream = 0
            downstream = 0
            for i in range(len(self.clients)):
                self.clients[i].set_model_weights(self.global_model.get_weights())
                downstream = downstream + len(
                    self.global_model.get_weights())
            print('===================================================')
            print('Round ' + str(round_) + ' global model')
            print('===================================================')
            try:
                print(self.global_model.get_weights())
            except:
                print('Model non initialized')
            # running local training on all clients
            local_weights_list: List = []
            _, _, clients_idx = self.run_local_training(local_weights_list, round_=round_, run=run)
            _, _, _ = self.run_local_evaluation(run, round_)
            print(clients_idx)
            self.global_model, server_time = self.server.weighted_average_weights(local_weights_list, clients_idx, self.FL_parameters["aggregation_method"])
            round_end_time = (time.time() - round_strt_time)
            upstream = upstream + len(local_weights_list) * len(local_weights_list[0])
            print('After aggregation')
            print('===================================================')
            print(self.global_model.get_weights())
            self.weights_list.append([run, round_, -1, 'global model', self.global_model.get_weights()])
            # getting global test accuracies, global test losses, global test metrics
            global_test_acc, global_test_loss, global_test_metrics = self.run_global_evaluation(run, round_, round_end_time, server_time, upstream, downstream)
        print(f"| Warmup ---- Test Accuracy: {100 * global_test_acc:.2f}% --- Test Loss: \ {global_test_loss:.2f} --- Metrics {global_test_metrics}")

    def run_local_training(self, local_weights_list: List, *a, **kw) -> Tuple:
        """
            run local training on all clients
        """
        local_models: List = []
        clients_idx: List = []
        for i in range(len(self.clients)):
            # getting size of client (i) training sets to use it later in the averaging
            if (self.clients[i].train_dataset != -1 ) :
               clients_idx.append(self.clients[i].train_dataset.__len__())
            else :
               clients_idx.append(0)
            # local training
            print('===================================================')
            print(f'client {i}')
            print('Before local training')
            try:
                '''print(self.clients[i].get_model_weights())'''
            except Exception as err:
                print(err, 'Model non initialized')
            if (self.clients[i].train_dataset != -1 ) :
               local_model, loss = self.clients[i].local_training(self.global_model)
               local_models.append(local_model)
               print('After local training')
               print(local_model.get_weights())
               print('===================================================')
               # saving local model weight and train loss of client (i)
               local_weights_list.append(local_model.get_weights())
               self.weights_list.append([kw["run"], kw["round_"], i, 'local model', local_model.get_weights()])
               self.metrics_list.append([kw["run"], kw["round_"], i, 'local model training loss', -1, loss, -1, -1, -1, -1])
            else :
               clients_idx.append(0)
               local_weights_list.append([0 for i in range(100)])
        return local_models, loss, clients_idx
    
    
    def run_global_evaluation(self, run, round_, round_end_time, server_time, upstream, downstream):
        # getting global test accuracies, global test losses, global test metrics
        try:
            global_valid_acc, global_valid_loss, global_valid_metrics = self.server.validation()
            self.metrics_list.append(
                [run, round_, -1, 'global model tested on validation set', global_valid_acc,
                 global_valid_loss,
                 global_valid_metrics, round_end_time, server_time, upstream, downstream])
            #print('server_time in s'+ str(server_time))
            print('Global validation')
            print(global_valid_acc, global_valid_loss, global_valid_metrics)
        except:
            print('No validation set provided.')
        global_test_acc, global_test_loss, global_test_metrics= self.server.test_inference()
        self.metrics_list.append([run, round_, -1, 'global model tested on test set', global_test_acc, global_test_loss, global_test_metrics, round_end_time, server_time, upstream, downstream])
        print('Global test')
        print(global_test_acc, global_test_loss, global_test_metrics)
        return global_test_acc, global_test_loss, global_test_metrics

    def run_local_evaluation(self, run, round_):
        clients_train_metrics = []
        clients_valid_metrics = []
        clients_test_metrics = []
        for i in range(len(self.clients)):
#            if  self.clients[i].train_dataset != -1:
#                local_train_acc, local_train_loss, local_train_metrics = self.clients[i].test_inference(self.clients[i].train_dataset)
#                print(i)
#                print(local_train_acc, local_train_loss, local_train_metrics)
#                self.metrics_list.append(
#                    [run, round_, i, 'local model tested on train set', local_train_acc, local_train_loss, local_train_metrics, -1, -1, -1])
#                clients_train_metrics.append(local_train_metrics)
            try:
                print("getting validation data")
                validation_data = self.server.get_validation_dataset()
                local_valid_acc, local_valid_loss, local_valid_metrics = self.clients[i].test_inference(validation_data)
                print(i)
                print(local_valid_acc, local_valid_loss, local_valid_metrics)
                self.metrics_list.append([run, round_, i, 'local model tested on validation set', local_valid_acc, local_valid_loss, local_valid_metrics, -1, -1, -1])
                clients_valid_metrics.append(local_valid_metrics)
            except:
                print('No validation set provided.')
#            test_data = self.server.get_testing_dataset()
#            local_test_acc, local_test_loss, local_test_metrics = self.clients[i].test_inference(test_data)
#            print(i)
#            print(local_test_acc, local_test_loss, local_test_metrics)
#            self.metrics_list.append([run, round_, i, 'local model tested on test set', local_test_acc, local_test_loss, local_test_metrics, -1, -1, -1])
#            clients_test_metrics.append(local_test_metrics)
        return clients_train_metrics, clients_valid_metrics, clients_test_metrics
        
    def write_metrics_df(self, i):
        metrics_path = self.FL_parameters["metrics_collection_path"]
        columns = ['runs_id', 'round_id', 'client_id', 'info', 'accuracy', 'loss', 'metrics', 'round_train_time','server_time', 'size_upstream', 'size_downstream']
        metrics_df = pd.DataFrame(data=self.metrics_list, columns=columns)
        for param in self.FL_parameters, self.learning_info, self.data_preparation_info, self.data_info, self.BiasMitigation_info:
            for k in param.keys():
                print(k)
                try:
                    metrics_df[k] = param[k]
                except:
                    metrics_df[k] = json.dumps(param[k])
        metrics_df['remark'] = self.remark
        columns = ['runs_id', 'round_id', 'client_id', 'info', 'weights']
        weights_df = pd.DataFrame(data=self.weights_list, columns=columns)
        timestamp = time.strftime("%b_%d_%Y_%H_%M_%S")
        filename = f"collected_metrics-{timestamp}.csv"
        metrics_df.to_csv(os.path.join(metrics_path, filename), index=False)
        filename = f"learned_weights-{timestamp}.csv"
        weights_df.to_csv(os.path.join(metrics_path, filename), index=False)

class BiasMitigationBase(FLBase):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

class PW_FL(BiasMitigationBase):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.e_min = self.BiasMitigation_info["e_min"]
        self.e_max = self.BiasMitigation_info["e_max"]

    def run_rounds(self, run: int, *a, **kw):
        global_test_acc, global_test_loss, global_test_metrics = self.run_global_evaluation(run, -1, 0, 0,0, 0)
        for round_ in range(self.FL_parameters["nb_rounds"]):
            round_strt_time = time.time()
            upstream = 0
            downstream = 0
            # send global model of last round to the clients
            for i in range(len(self.clients)):
                self.clients[i].set_model_weights(self.global_model.get_weights())
                downstream = downstream + len(
                    self.global_model.get_weights())
            print('===========================================global_test_acc, global_test_loss, global_test_metrics========')
            print(f'Round {round_} global model')
            print('===================================================')
            try:
                print(self.global_model.get_weights())
            except NotImplementedError:
                print('Model non initialized')
            local_weights_list: List[List] = []
            _, _, clients_idx = self.run_local_training(local_weights_list, round_=round_, run=run)
            (pessemistic_local_weights_list, pessemistic_client_idx) \
                = self.select_clients(clients_idx, local_weights_list, round_, run, self.BiasMitigation_info["fairness_metric_name"])
            print(pessemistic_client_idx)
            if len(pessemistic_client_idx) > 0:
                upstream = upstream + len(local_weights_list) * len(local_weights_list[0])
                self.global_model, server_time = self.server.weighted_average_weights(pessemistic_local_weights_list, pessemistic_client_idx, self.FL_parameters["aggregation_method"], self.learning_info['byzantine'])
                round_end_time = (time.time() - round_strt_time)
                print('After aggregation')
                print('===================================================')
                print(self.global_model.get_weights())
                self.weights_list.append([run, round_, -1, 'global model', self.global_model.get_weights()])
                global_test_acc, global_test_loss, global_test_metrics = self.run_global_evaluation(run, round_, round_end_time, server_time, upstream, downstream)
            else:
                print('No client selected by Pessemistic Weighted Aggregation')

        print(f"| Run {run} ---- Test Accuracy: {100 * global_test_acc:.2f}% --- Test Loss: \ {global_test_loss:.2f} --- Metrics {global_test_metrics}")
        self.e_min =  self.BiasMitigation_info["e_min"]

    def select_clients(self, clients_idx, local_weights_list, round_, run, PW_metric_name) -> Tuple[List, List]:
        """
            method: pessimistic
        """

        def condition_selection(metrics , e_min, e_max):
            if isinstance(metrics, list):
                i = 0
                print(metrics)
                while (i < (len(metrics))):
                    if ( metrics[i] < e_min[i]) or (metrics[i] > e_max[i]):
                        return False
                    i = i +1
                return True
            else:
                if (metrics >= e_min) and (metrics <= e_max):
                    return True
                else:
                    return False
        pessemistic_local_weights_list: List = []
        pessemistic_client_idx: List = []
        clients_train_metrics, clients_valid_metrics, clients_test_metrics = self.run_local_evaluation(run, round_)
        for i in range(len(self.clients)):
            local_test_metrics = clients_valid_metrics[i]
            print("===================e_min {}".format(self.e_min))
            if condition_selection(local_test_metrics[PW_metric_name] , self.e_min, self.e_max):
                pessemistic_local_weights_list.append(local_weights_list[i])
                pessemistic_client_idx.append(clients_idx[i])
                print(i)
                print('selected')
        if ((self.BiasMitigation_info["dynamic"] == 1)):
            self.e_min = self.e_min +  self.BiasMitigation_info["step"]   
        return pessemistic_local_weights_list, pessemistic_client_idx

class IBW_FL(BiasMitigationBase):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def prelec_function_DI(self, p, alpha, beta):
        if p <= 1:
            return np.exp(-beta * (pow(-np.log(p), alpha)))
        else :
            return np.exp(-beta * (pow(-np.log(1 / p), alpha)))

    def prelec_function_SPD(self, p, alpha, beta):
        if p <= 0:
            return np.exp(-beta * (pow(-np.log(p+1), alpha)))
        else :
            return np.exp(-beta * (pow(-np.log(-p+1), alpha)))

    def WTC_function(self, p, beta):
        if p <= 1:
            return pow(p, beta) / pow(pow(p, beta) + pow(1 - p, beta), 1 / beta)
        else:
            return pow(1 / p, beta) / pow(pow(1 / p, beta) + pow(1 - (1 / p), beta), 1 / beta)

    def Linear_odds_log_function(self, p, sigma, gamma):
        if p <= 1:
            return sigma * pow(p, gamma) / (sigma * pow(p, gamma) + pow(1 - p, gamma))
        else :
            return sigma * pow(1 / p, gamma) / (sigma * pow(1 / p, gamma) + pow(1 - (1 / p), gamma))

    def run_rounds(self, run: int, *a, **kw):
        IBW_metric_name = self.BiasMitigation_info['fairness_metric_name']
        global_test_acc, global_test_loss, global_test_metrics = self.run_global_evaluation(run, -1, 0, 0,0, 0)
        for round_ in range(self.FL_parameters["nb_rounds"]):
            round_strt_time = time.time()
            upstream = 0
            downstream = 0
            for i in range(len(self.clients)):
                self.clients[i].set_model_weights(self.global_model.get_weights())
                downstream = downstream + len(
                    self.global_model.get_weights())
            print('===================================================')
            print('Round ' + str(round_) + ' global model')
            print('===================================================')
            try:
                print(self.global_model.get_weights())
            except:
                print('Model non initialized')
            IBW_agg_weights_list = []
            # running local training on all clients
            local_weights_list: List = []
            _, _, clients_idx = self.run_local_training(local_weights_list, round_=round_, run=run)
            clients_train_metrics, clients_valid_metrics, clients_test_metrics = self.run_local_evaluation(run, round_)
            print(clients_idx)
            for i in range(len(self.clients)):
                local_test_metrics = clients_valid_metrics[i]
                print(local_test_metrics[IBW_metric_name])
                local_bias = local_test_metrics[IBW_metric_name]
                if isinstance(local_bias, list):
                    if IBW_metric_name == 'Disparate Impact':
                        alpha = self.BiasMitigation_info['alpha']
                        beta = self.BiasMitigation_info['beta']
                        agg_weight = clients_idx[i] * self.prelec_function_DI(max([abs(ele) for ele in local_bias]), alpha, beta)
                    elif IBW_metric_name == 'Statistical Parity Difference':
                        alpha = self.BiasMitigation_info['alpha']
                        beta = self.BiasMitigation_info['beta']
                        agg_weight = clients_idx[i] * self.prelec_function_SPD(max([abs(ele) for ele in local_bias]), alpha, beta)
                        print('SPD {} weight {}'.format(local_bias, agg_weight))
                    elif IBW_metric_name == 'Equal Opportunity Difference':
                        alpha = self.BiasMitigation_info['alpha']
                        beta = self.BiasMitigation_info['beta']
                        agg_weight = clients_idx[i] * self.prelec_function_SPD(max([abs(ele) for ele in local_bias]), alpha, beta)
                        print('EOD {} weight {}'.format(local_bias, agg_weight))
                    elif IBW_metric_name == 'Discrimination Index':
                        alpha = self.BiasMitigation_info['alpha']
                        beta = self.BiasMitigation_info['beta']
                        agg_weight = clients_idx[i] * self.prelec_function_SPD(max([abs(ele) for ele in local_bias]), alpha, beta)
                        print('DI {} weight {}'.format(local_bias, agg_weight))
                else:
                    if IBW_metric_name == 'Disparate Impact':
                        alpha = self.BiasMitigation_info['alpha']
                        beta = self.BiasMitigation_info['beta']
                        agg_weight = clients_idx[i] * self.prelec_function_DI(local_bias, alpha, beta)
                    elif IBW_metric_name == 'Statistical Parity Difference':
                        alpha = self.BiasMitigation_info['alpha']
                        beta = self.BiasMitigation_info['beta']
                        agg_weight = clients_idx[i] * self.prelec_function_SPD(local_bias, alpha, beta)
                        print('SPD {} weight {}'.format(local_bias, agg_weight))
                    elif IBW_metric_name == 'Equal Opportunity Difference':
                        alpha = self.BiasMitigation_info['alpha']
                        beta = self.BiasMitigation_info['beta']
                        agg_weight = clients_idx[i] * self.prelec_function_SPD(local_bias, alpha, beta)
                        print('EOD {} weight {}'.format(local_bias, agg_weight))
                    elif IBW_metric_name == 'Discrimination Index':
                        alpha = self.BiasMitigation_info['alpha']
                        beta = self.BiasMitigation_info['beta']
                        agg_weight = clients_idx[i] * self.prelec_function_SPD(local_bias, alpha, beta)
                        print('DI {} weight {}'.format(local_bias, agg_weight))
                IBW_agg_weights_list.append(agg_weight)
            print('Aggregation weights for round {} : {} '.format(round_, IBW_agg_weights_list))
            upstream = upstream + len(local_weights_list) * len(local_weights_list[0])
            self.global_model, server_time = self.server.weighted_average_weights(local_weights_list, IBW_agg_weights_list, self.FL_parameters["aggregation_method"], self.learning_info['byzantine'])
            round_end_time = (time.time() - round_strt_time)
            print('After aggregation')
            print('===================================================')
            print(self.global_model.get_weights())
            self.weights_list.append([run, round_, -1, 'global model', self.global_model.get_weights()])
            # getting global test accuracies, global test losses, global test metrics
            global_test_acc, global_test_loss, global_test_metrics = self.run_global_evaluation(run, round_, round_end_time, server_time, upstream, downstream)
            print(f"| Run {run} ---- Test Accuracy: {100 * global_test_acc:.2f}% --- Test Loss: \ {global_test_loss:.2f} --- Metrics {global_test_metrics}")

    def run_rounds_not_prelec(self, run: int, *a, **kw):
        IBW_metric_name = self.BiasMitigation_info['fairness_metric_name']
        PR_function = self.BiasMitigation_info['probability_reweighing_function']
        for round_ in range(self.FL_parameters["nb_rounds"]):
            round_strt_time = time.time()
            for i in range(len(self.clients)):
                self.clients[i].set_model_weights(self.global_model.get_weights())
            print('===================================================')
            print('Round ' + str(round_) + ' global model')
            print('===================================================')
            try:
                print(self.global_model.get_weights())
            except:
                print('Model non initialized')
            IBW_agg_weights_list = []
            # running local training on all clients
            local_weights_list: List = []
            _, _, clients_idx = self.run_local_training(local_weights_list, round_=round_, run=run)
            clients_train_metrics, clients_valid_metrics, clients_test_metrics = self.run_local_evaluation(run, round_)
            print(clients_idx)
            for i in range(len(self.clients)):
                local_test_metrics = clients_valid_metrics[i]
                print(local_test_metrics[IBW_metric_name])
                local_bias = local_test_metrics[IBW_metric_name]
                if PR_function == 'PrelecW':
                    alpha = self.BiasMitigation_info['alpha']
                    beta = self.BiasMitigation_info['beta']
                    agg_weight = clients_idx[i] * self.prelec_function(local_bias, alpha, beta)
                elif PR_function == 'Wu_GonzW':
                    agg_weight = clients_idx[i] * self.WTC_function(local_bias, 0.71)
                elif PR_function == 'Tve_KahW':
                    agg_weight = clients_idx[i] * self.WTC_function(local_bias, 0.61)
                elif PR_function == 'Cam_HoW':
                    agg_weight = clients_idx[i] * self.WTC_function(local_bias, 0.56)
                elif PR_function == 'Linear_odds_logW':
                    sigma = self.BiasMitigation_info['sigma']
                    gamma = self.BiasMitigation_info['gamma']
                    agg_weight = clients_idx[i] * self.Linear_odds_log_function(local_bias, sigma, gamma)
                else:
                    agg_weight = -1
                    print('Function not supported')
                IBW_agg_weights_list.append(agg_weight)
            self.global_model, server_time = self.server.weighted_average_weights(local_weights_list, IBW_agg_weights_list, self.FL_parameters["aggregation_method"], self.learning_info['byzantine'])
            print('Aggregation weights for round {} : {} '.format(round_, IBW_agg_weights_list))
            round_end_time = (time.time() - round_strt_time)
            print('After aggregation')
            print('===================================================')
            print(self.global_model.get_weights())
            self.weights_list.append([run, round_, -1, 'global model', self.global_model.get_weights()])
            # getting global test accuracies, global test losses, global test metrics
            global_test_acc, global_test_loss, global_test_metrics = self.run_global_evaluation(run, round_, round_end_time)
            print(f"| Run {run} ---- Test Accuracy: {100 * global_test_acc:.2f}% --- Test Loss: \ {global_test_loss:.2f} --- Metrics {global_test_metrics}")

class FairFed(BiasMitigationBase):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def compute_f1_score(self, predictions_dict):
        tp, fp, fn = 0, 0, 0
        for key, value in predictions_dict.items():
            pred, true = key
            if pred == true:
                tp += value
            elif pred == 1 and true == 0:
                fp += value
            elif pred == 0 and true == 1:
                fn += value
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def DI(self, n_yz, absolute = True):
        """
        Given a dictionary of number of samples in different groups, compute the discrimination index.
        |F1(Group1) - F1(Group2)|
        """
        f1_scores = {}
        for key, value in n_yz.items():
            pred, true, sensitive = key
            if sensitive not in f1_scores:
                f1_scores[sensitive] = {'tp': 0, 'fp': 0, 'fn': 0}
            if pred == true == 1:
                f1_scores[sensitive]['tp'] += value
            elif pred == 1 and true == 0:
                f1_scores[sensitive]['fp'] += value
            elif pred == 0 and true == 1:
                f1_scores[sensitive]['fn'] += value
        for sensitive, scores in f1_scores.items():
            tp, fp, fn = scores['tp'], scores['fp'], scores['fn']
            
            precision = 0
            recall = 0
            if(tp + fp !=0):
                precision = tp / (tp + fp)
            if(tp + fn!=0):
                recall = tp / (tp + fn)
            if (precision + recall !=0):
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0
            f1_scores[sensitive] = f1_score
        
        print("F1 Score priv",f1_scores[1])
        print("F1 Score unpriv",f1_scores[0])
        if absolute:
            return abs(f1_scores[1] - f1_scores[0])
        else:
            return f1_scores[1] - f1_scores[0]
        
    def SPD(self, n_yz, absolute = True):
        """
        Given a dictionary of number of samples in different groups, compute the risk difference.
        |P(Group1, pos) - P(Group2, pos)| = |N(Group1, pos)/N(Group1) - N(Group2, pos)/N(Group2)|
        """
        n_z1 = max(n_yz[(1,1)] + n_yz[(0,1)], 1)
        n_z0 = max(n_yz[(0,0)] + n_yz[(1,0)], 1)
        if absolute:
            return abs(n_yz[(1,1)]/n_z1 - n_yz[(1,0)]/n_z0)
        else:
            return n_yz[(1,1)]/n_z1 - n_yz[(1,0)]/n_z0
    
    def EOD(self, n_yz, absolute = True):
        """
        Given a dictionary of number of samples in different groups, compute the risk difference.
        |P(Group1, pos) - P(Group2, pos)| = |N(Group1, pos, pospred)/N(Group1) - N(Group2, pos,pospred )/N(Group2)|
        """
        n_z1 = max(n_yz[(1,0,1)]+ n_yz[(1,1,1)] + n_yz[(0,0,1)]+n_yz[(0,1,1)], 1)
        n_z0 = max(n_yz[(1,0,0)]+ n_yz[(1,1,0)] + n_yz[(0,0,0)]+n_yz[(0,1,0)], 1)
        if absolute:
            return abs(n_yz[(1,1,1)]/n_z1 - n_yz[(1,1,0)]/n_z0)
        else:
            return n_yz[(1,1,1)]/n_z1 - n_yz[(1,1,0)]/n_z0

    def prelec_function_DI(self, p, alpha, beta):
        if p <= 1:
            return np.exp(-beta * (pow(-np.log(p), alpha)))
        else :
            return np.exp(-beta * (pow(-np.log(1 / p), alpha)))

    def prelec_function_SPD(self, local_bias, beta, n_yz):
        return np.exp(-beta * abs(self.SPD(local_bias) - self.SPD(n_yz)))
    
    def prelec_function_EOD(self, local_bias, beta, n_yz):
        return np.exp(-beta * abs(self.EOD(local_bias) - self.EOD(n_yz)))
    
    def prelec_function_DI(self, local_bias, beta, n_yz):
        return np.exp(-beta * abs(self.DI(local_bias) - self.DI(n_yz))) 

    def run_rounds(self, run: int, *a, **kw):
        fairfed_metric_name = self.BiasMitigation_info['fairness_metric_name']
        global_test_acc, global_test_loss, global_test_metrics= self.run_global_evaluation(run, -1, 0, 0,0, 0)
        for round_ in range(self.FL_parameters["nb_rounds"]):
            round_strt_time = time.time()
            upstream = 0
            downstream = 0
            for i in range(len(self.clients)):
                self.clients[i].set_model_weights(self.global_model.get_weights())
                downstream = downstream + len(
                    self.global_model.get_weights())
            print('===================================================')
            print('Round ' + str(round_) + ' global model')
            print('===================================================')
            try:
                print(self.global_model.get_weights())
            except:
                print('Model non initialized')
            IBW_agg_weights_list = []
            # running local training on all clients
            local_weights_list: List = []
            _, _, clients_idx = self.run_local_training(local_weights_list, round_=round_, run=run)
            n_yz, loss_yz, m_yz, f_z = {}, {}, {}, {}

            if fairfed_metric_name == 'Statistical Parity Difference':
                for y in [0,1]:
                    for z in range(2):
                        n_yz[(y,z)] = 0
            elif fairfed_metric_name == 'Equal Opportunity Difference':
                for y in [0,1]:
                    for y_true in [0,1]:
                        for z in range(2):
                            n_yz[(y,y_true,z)] = 0
            elif fairfed_metric_name == 'Discrimination Index':
                for y in [0,1]:
                    for y_true in [0,1]:
                        for z in range(2):
                            n_yz[(y,y_true,z)] = 0

            clients_train_metrics, _, _, n_yz_c = self.run_local_evaluation_fairfed(run, round_, fairfed_metric_name)
            for yz in n_yz:
                for i in range(len(self.clients)):
                    n_yz[yz] += n_yz_c[i][yz]
            beta = self.BiasMitigation_info['beta']
            print(clients_idx)
            if fairfed_metric_name == 'Statistical Parity Difference':
                print('Aprox global SPD {} distirbution {}'.format(self.SPD(n_yz), n_yz))
            elif fairfed_metric_name == 'Equal Opportunity Difference':
                print('Aprox global EOD {} distirbution {}'.format(self.EOD(n_yz), n_yz))
            elif fairfed_metric_name == 'Discrimination Index':
                print('Aprox global DI {} distirbution {}'.format(self.DI(n_yz), n_yz))
            for i in range(len(self.clients)):
                local_test_metrics = clients_train_metrics[i]
                # print(local_test_metrics[IBW_metric_name])
                local_bias = local_test_metrics[fairfed_metric_name]
                if fairfed_metric_name == 'Statistical Parity Difference':
                    agg_weight = self.prelec_function_SPD(n_yz_c[i], beta,n_yz)* clients_idx[i] / sum(clients_idx)
                    # print('SPD {} weight {}'.format(local_bias, agg_weight))
                elif fairfed_metric_name == 'Equal Opportunity Difference':
                    agg_weight = self.prelec_function_EOD(n_yz_c[i], beta,n_yz)* clients_idx[i] / sum(clients_idx)
                    # print('EOD {} weight {}'.format(local_bias, agg_weight))
                elif fairfed_metric_name == 'Discrimination Index':
                    agg_weight = (clients_idx[i] *self.prelec_function_DI(n_yz_c[i],beta,n_yz)) /sum(clients_idx)
                    print('DI {} fairfed weight {} weight {}'.format(self.DI(n_yz_c[i]),self.prelec_function_DI(n_yz_c[i],beta,n_yz), agg_weight))
                IBW_agg_weights_list.append(agg_weight)
            print('Aggregation weights for round {} : {} '.format(round_, IBW_agg_weights_list))
            upstream = upstream + len(local_weights_list) * len(local_weights_list[0])
            self.global_model, server_time = self.server.weighted_average_weights(local_weights_list, IBW_agg_weights_list, self.FL_parameters["aggregation_method"], self.learning_info['byzantine'])
            round_end_time = (time.time() - round_strt_time)
            print('After aggregation')
            print('===================================================')
            print(self.global_model.get_weights())
            self.weights_list.append([run, round_, -1, 'global model', self.global_model.get_weights()])
            # getting global test accuracies, global test losses, global test metrics
            global_test_acc, global_test_loss, global_test_metrics, n_yz_valid,n_yz_test = self.run_global_evaluation_fairfed(run, round_, round_end_time, server_time, upstream, downstream)
            print(f"| Run {run} ---- Valid DI: \ {self.DI(n_yz_valid):.5f} --- Test DI: \ {self.DI(n_yz_test):.5f}")
            print(f"| Run {run} ---- Test Accuracy: {100 * global_test_acc:.2f}% --- Test Loss: \ {global_test_loss:.2f} --- Metrics {global_test_metrics}")

    def run_local_evaluation_fairfed(self, run, round_, fairfed_metric_name):
            clients_train_metrics = []
            clients_valid_metrics = []
            clients_test_metrics = []
            n_yz_c = []
            for i in range(len(self.clients)):
                if fairfed_metric_name == 'Statistical Parity Difference':
                    local_train_acc, local_train_loss, local_train_metrics, n_yz = self.clients[i].test_inference_SPD_fairfed(self.clients[i].train_dataset)
                elif fairfed_metric_name == 'Equal Opportunity Difference':
                    local_train_acc, local_train_loss, local_train_metrics, n_yz = self.clients[i].test_inference_EOD_fairfed(self.clients[i].train_dataset)
                elif fairfed_metric_name == 'Discrimination Index':
                    local_train_acc, local_train_loss, local_train_metrics, n_yz = self.clients[i].test_inference_EOD_fairfed(self.clients[i].train_dataset)
                
                print(i)
                print(local_train_acc, local_train_loss, local_train_metrics)
                self.metrics_list.append(
                    [run, round_, i, 'local model tested on train set', local_train_acc, local_train_loss, local_train_metrics, -1, -1, -1])
                clients_train_metrics.append(local_train_metrics)
                n_yz_c.append(n_yz)
                print(n_yz)
            return clients_train_metrics, clients_valid_metrics, clients_test_metrics, n_yz_c
    
    def run_global_evaluation_fairfed(self, run, round_, round_end_time, server_time, upstream, downstream):
        # getting global test accuracies, global test losses, global test metrics
        try:
            global_valid_acc, global_valid_loss, global_valid_metrics,n_yz_valid = self.server.validation_fairfed()
            self.metrics_list.append(
                [run, round_, -1, 'global model tested on validation set', global_valid_acc,
                 global_valid_loss,
                 global_valid_metrics, round_end_time, server_time, upstream, downstream])
            #print('server_time in s'+ str(server_time))
            print('Global validation')
            print(global_valid_acc, global_valid_loss, global_valid_metrics)
        except:
            print('No validation set provided.')
        global_test_acc, global_test_loss, global_test_metrics,n_yz_test= self.server.test_inference_fairfed()
        self.metrics_list.append([run, round_, -1, 'global model tested on test set', global_test_acc, global_test_loss, global_test_metrics, round_end_time, server_time, upstream, downstream])
        print('Global test')
        print(global_test_acc, global_test_loss, global_test_metrics)
        return global_test_acc, global_test_loss, global_test_metrics,n_yz_valid,n_yz_test

class ASTRAL_OPT_FL(BiasMitigationBase):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.metric_name = self.BiasMitigation_info['fairness_metric_name']
        self.variant = self.BiasMitigation_info['variant']
        self.logs = bool(self.BiasMitigation_info["logs"])
    def run_rounds(self, run: int, *a, **kw):
        global_test_acc, global_test_loss, global_test_metrics = self.run_global_evaluation(run, -1, 0, 0, 0, 0)
        for round_ in range(self.FL_parameters["nb_rounds"]):
            round_strt_time = time.time()
            upstream = 0
            downstream = 0
            for i in range(len(self.clients)):
                self.clients[i].set_model_weights(self.global_model.get_weights())
                downstream = downstream + len(
                    self.global_model.get_weights())
            print('===================================================')
            print('Round ' + str(round_) + ' global model')
            print('===================================================')
            try:
                print(self.global_model.get_weights())
            except:
                print('Model non initialized')
            IBW_agg_weights_list = []
            # running local training on all clients
            local_weights_list: List = []
            local_models, _, clients_idx = self.run_local_training(local_weights_list, round_=round_, run=run)
            clients_train_metrics, clients_valid_metrics, clients_test_metrics = self.run_local_evaluation(run, round_)
            upstream = upstream + len(local_weights_list) * len(local_weights_list[0])
            self.global_model, server_time = self.server.Astral_optim_aggregation(self.metric_name, local_models, local_weights_list, clients_idx,round_, self.learning_info['byzantine'], self.BiasMitigation_info,clients_valid_metrics)
            round_end_time = (time.time() - round_strt_time)
            print('After aggregation')
            print('===================================================')
            print(self.global_model.get_weights())
            self.weights_list.append([run, round_, -1, 'global model', self.global_model.get_weights()])
            # getting global test accuracies, global test losses, global test metrics
            global_test_acc, global_test_loss, global_test_metrics = self.run_global_evaluation(run, round_, round_end_time, server_time, upstream, downstream)
            print(f"| Run {run} ---- Test Accuracy: {100 * global_test_acc:.2f}% --- Test Loss: \ {global_test_loss:.2f} --- Metrics {global_test_metrics}")


def PesonalizedW_FL(FL_parameters, train_sets, validation_set, test_set, FL_model, learning_info, data_preparation_info, data_info, PersonalizedW_info, remark):
    nb_runs = FL_parameters["nb_runs"]
    aggregation = FL_parameters["aggregation_method"]
    nb_users = FL_parameters["nb_clients"]
    nb_rounds = FL_parameters["nb_rounds"]
    ratio_clients = FL_parameters["ratio_of_participating_clients"]
    client_selection_method = FL_parameters["participants_selection_method"]
    metrics_path = FL_parameters["metrics_collection_path"]
    PersonalizedW_metric_name = PersonalizedW_info['fairness_metric_name']
    PR_function = PersonalizedW_info['probability_reweighing_function']
    seed = 42
    print(nb_rounds)
    for run in range(nb_runs):
        train_loss, local_test_accuracy, local_test_loss, local_test_metrics, global_test_accuracy, global_test_loss, global_test_metrics = [], [], [], [], [], [], []
        weights_list = []
        metrics_list = []
        # setting server with test data and global model
        server = Server(validation_set, test_set, FL_model)
        global_model = copy.deepcopy(FL_model)
        for r_ound in range(nb_rounds):
            round_strt_time = time.time()
            if r_ound == 0:
                # setting clients with training sets data and global model
                clients = [Client(train_sets[i], FL_model, learning_info,) for i in range(nb_users)]
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
            PersonalizedW_models_list = []
            PersonalizedW_agg_weights_list = []
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
                PersonalizedW_models_list.append(local_model.get_weights())
                weights_list.append([run, r_ound, i, 'local model', local_model.get_weights()])
                metrics_list.append([run, r_ound, i, 'local model training loss', -1, loss, -1, -1])
            print(clients_idx)
            # getting local models metrics and selecting clients based on pessemistic method
            # getting local test accuracies, local test losses, local test metrics
            start = timer()
            for i in range(nb_users):
                test_data = server.get_testing_dataset()
                local_test_acc, local_test_loss, local_test_metrics = clients[i].test_inference(test_data)
                print(i)
                print(local_test_acc, local_test_loss, local_test_metrics)
                metrics_list.append([run, r_ound, i, 'local model tested on test set', local_test_acc, local_test_loss, local_test_metrics, -1])
                # computing aggregation weights based on clients bias
                print(local_test_metrics[PersonalizedW_metric_name])
                local_bias = local_test_metrics[PersonalizedW_metric_name]
                if PR_function == 'PrelecW':
                    alpha = PersonalizedW_info['alpha']
                    beta = PersonalizedW_info['beta']
                    agg_weight = clients_idx[i] * prelec_function(local_bias, alpha, beta)
                elif PR_function == 'Wu_GonzW':
                    agg_weight = clients_idx[i] * WTC_function(local_bias, 0.71)
                elif PR_function == 'Tve_KahW':
                    agg_weight = clients_idx[i] * WTC_function(local_bias, 0.61)
                elif PR_function == 'Cam_HoW':
                    agg_weight = clients_idx[i] * WTC_function(local_bias, 0.56)
                elif PR_function == 'Linear_odds_logW':
                    sigma = PersonalizedW_info['sigma']
                    gamma = PersonalizedW_info['gamma']
                    agg_weight = clients_idx[i] * Linear_odds_log_function(local_bias, sigma, gamma)
                else:
                    print('Function not supported')
                PersonalizedW_agg_weights_list.append(agg_weight)
            try:
                for i in range(nb_users):
                    validation_data = server.get_validation_dataset()
                    local_valid_acc, local_valid_loss, local_valid_metrics = clients[i].test_inference(validation_data)
                    print(i)
                    print(local_valid_acc, local_valid_loss, local_valid_metrics)
                    metrics_list.append([run, r_ound, i, 'local model tested on validation set', local_valid_acc, local_valid_loss, local_valid_metrics, -1])
            except:
                print('No validation set provided.')
            global_model, server_time = server.weighted_average_weights(PersonalizedW_models_list, PersonalizedW_agg_weights_list)
            end = timer()
            round_end_time = (time.time() - round_strt_time)
            print('After aggregation')
            print('===================================================')
            print(global_model.get_weights())
            weights_list.append([run, r_ound, -1, 'global model', global_model.get_weights()])
            # getting global test accuracies, global test losses, global test metrics
            try:
                global_valid_acc, global_valid_loss, global_valid_metrics = server.validation()
                metrics_list.append([run, r_ound, -1, 'global model tested on validation set', global_valid_acc, global_valid_loss, global_valid_metrics, round_end_time])
                print('global')
                print(global_valid_acc, global_valid_loss, global_valid_metrics)
            except:
                print('No validation set provided.')
            global_test_acc, global_test_loss, global_test_metrics = server.test_inference()
            metrics_list.append([run, r_ound, -1, 'global model tested on test set', global_test_acc, global_test_loss, global_test_metrics, round_end_time])
            print('global')
            print(global_test_acc, global_test_loss, global_test_metrics)
        print("| Run {} ---- Test Accuracy: {:.2f}% --- Test Loss: {:.2f} --- Metrics {}".format(run, 100 * global_test_acc, global_test_loss, global_test_metrics))
        columns = ['rund_id', 'round_id', 'client_id', 'info', 'accuracy', 'loss', 'metrics', 'round_train_time']
        metrics_df = pd.DataFrame(data=metrics_list, columns=columns)
        for param in FL_parameters, learning_info, data_preparation_info, data_info, PersonalizedW_info:
            for k in param.keys():
                print(k)
                try:
                    metrics_df[k] = param[k]
                except:
                    metrics_df[k] = json.dumps(param[k])
        metrics_df['remark'] = remark
        columns = ['rund_id', 'round_id', 'client_id', 'info', 'weights']
        weights_df = pd.DataFrame(data=weights_list, columns=columns)
        timestamp = time.strftime("%b_%d_%Y_%H_%M_%S")
        filename = "collected_metrics-" + "run" + str(run) + "-" + timestamp + ".csv"
        metrics_df.to_csv(os.path.join(metrics_path, filename), index=False)
        filename = "learned_weights-" + "run" + str(run) + "-" + timestamp + ".csv"
        weights_df.to_csv(os.path.join(metrics_path, filename), index=False)
    return 0

def prelec_function(p, alpha, beta):
    if p <= 1:
        return np.exp(-beta * (pow(-np.log(p), alpha)))
    else:
        return np.exp(-beta * (pow(-np.log( 1 / p), alpha)))

def WTC_function(p, beta):
    if p <= 1:
        return pow(p, beta) / pow(pow(p, beta) + pow(1 - p, beta), 1 / beta)
    else :
        return pow(1/ p, beta) / pow(pow(1 / p, beta) + pow(1 - (1 / p), beta), 1 / beta)

def Linear_odds_log_function(p, sigma, gamma):
    if p <= 1:
        return sigma * pow(p, gamma) / (sigma * pow(p, gamma) + pow(1 - p, gamma))
    else :
        return sigma * pow(1 / p, gamma) / (sigma * pow(1 / p, gamma) + pow(1 - (1 / p), gamma))
