import numpy as np
import copy
import time
from multiprocessing.sharedctypes import Value
import torch
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from torch.autograd import Variable
from functools import reduce
from numpy.linalg import norm
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore')
from models import *

class Client(object):
    """docstring for Client"""

    def __init__(
            self, local_training_dataset, global_model, model_spec
    ):
        super(Client, self).__init__()
        self.device = "cpu"
        self.local_model = copy.deepcopy(global_model)
        self.train_dataset = local_training_dataset  # (local_training_dataset.x, local_training_dataset.y)
        try:
            self.local_random = bool(model_spec["local_random"])
            self.rand_proportion = model_spec["random_proportion"]
        except:
            self.local_random = False

    def local_training(self, global_model, seed=42):
        self.local_model = copy.deepcopy(global_model)
        if self.train_dataset != -1:
            epoch_loss = self.local_model.train(self.train_dataset)
        if self.local_random == True:
            self.local_model.set_weights_rand(self.rand_proportion)
        return self.local_model, epoch_loss

    def set_model(self, model):
        self.local_model = copy.deepcopy(model)

    def set_model_weights(self, model_weights):
        self.local_model.set_weights(model_weights)

    def get_model_weights(self):
        return self.local_model.get_weights()

    def get_model(self):
        return self.local_model

    def test_inference(self, testing_dataset):
        """Returns the inference accuracy and loss."""
        accuracy, loss, metrics = self.local_model.test_inference(testing_dataset)
        return accuracy, loss, metrics

class Server(object):
    """docstring for Server"""

    def __init__(self, nb_clients, validation_dataset, testing_dataset, global_model, test_batch_size=64):
        super(Server, self).__init__()
        self.device = "cpu"
        self.global_model = copy.deepcopy(global_model)
        self.global_model_variable = copy.deepcopy(global_model)
        self.nb_clients = nb_clients
        self.validation_dataset = validation_dataset
        self.testing_dataset = testing_dataset  # (testing_dataset.x, testing_dataset.y)
        self.test_batch_size = test_batch_size
        self.fed_avg_weights = [1 / self.nb_clients for i in range(self.nb_clients)]

    def set_model(self, model):
        self.global_model = copy.deepcopy(model)

    def set_model_weights(self, model_weights):
        self.global_model.set_weights(model_weights)

    def get_model_weights(self):
        return self.global_model.get_weights()

    def get_model(self):
        return self.global_model

    def get_validation_dataset(self):
        return self.validation_dataset

    def get_testing_dataset(self):
        return self.testing_dataset

    def update_global_model(self, global_weights):
        self.set_model_weights(global_weights)



    def gaussian_noise(self, data_shape, s, sigma):
        """
        Gaussian noise
        """
        return torch.normal(0, sigma * s, data_shape).to('cpu')

    def cdp(self, global_model, cdp_eps, N, bath_size, nb_epochs, nb_rounds, clip =1):

        clipped_grads = {name: torch.zeros_like(param) for name, param in global_model.model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(global_model.model.parameters(), max_norm=clip)

        for name, param in global_model.model.named_parameters():
            sigma = compute_noise(N, bath_size, cdp_eps, nb_epochs * nb_rounds, 0.00001, 0.1)

            clipped_grads[name] += self.gaussian_noise(clipped_grads[name].shape, clip, sigma)

            # scale back
        for name, param in global_model.model.named_parameters():
            clipped_grads[name] /= (N * 0.5)

        for name, param in global_model.model.named_parameters():
            param.data = clipped_grads[name]
        self.global_model = global_model

        return global_model
    def weighted_average_weights(self, w, nc, method, byzantine = None):
        start = timer()
        #performs aggregation
        if type(byzantine) != int:
            if byzantine != 0 or sum(byzantine) > 0:
                for i, j in enumerate(byzantine):
                    if j > 0:
                        w[i] = np.random.multivariate_normal(np.zeros(100), j ** 2 * np.identity(100), size=1).tolist()[0]
        print("performing aggregation")
        if method == 'fed_avg':
            '''classical average aggregation'''
            print("classical average aggregation")
            w_avg = copy.deepcopy(w)
            n = sum(nc)

            for i in range(len(w)):
                for j in range(len(w[i])):
                    w_avg[i][j] = w_avg[i][j] * nc[i]
                    w_avg[i][j] = w_avg[i][j] / n

            w_avg = np.sum(w_avg, axis=0)
            self.update_global_model(w_avg)
            end = timer()
            return copy.deepcopy(self.global_model), end-start

        if method == 'loss_obj_fair_const+':
            #classical average aggregation
            print("classical average aggregation")
            w_avg = copy.deepcopy(w)

            for i in range(len(w)):
                for j in range(len(w[i])):
                    w_avg[i][j] = w_avg[i][j] * nc[i]

            w_avg = np.sum(w_avg, axis=0)
            self.update_global_model(w_avg)
            end = timer()
            return copy.deepcopy(self.global_model), end-start

        if method[:4] == 'NDC_':
            #norm tresholding aggregation             
            print("norm tresholding aggregation")
            try:
                M = float(method[4:])
            except:
                raise TypeError("ill-defined NDC treshold")
            print(f"NDC; M = {M}")
            w_avg = w
            n = sum(nc)
            for i in range(len(w)):
                print(f"client {i}: norm={norm(w[i])}")
                norm_client = norm(w[i])
                for j in range(len(w[i])):
                    w_avg[i][j] = w[i][j] / max(1, norm_client / M)
                    w_avg[i][j] = w[i][j] * nc[i]
                    w_avg[i][j] = w_avg[i][j] / n
                print(f"client {i}: norm={norm(w[i])}")
            w_avg = np.sum(w_avg, axis=0)
            print(w_avg)
            self.update_global_model(w_avg)
            end = timer()
            print(self.global_model)
            print(self.global_model.get_weights())
            return copy.deepcopy(self.global_model), end-start
     
        elif method[:11] == 'multi_krum_':
            '''multi-krum aggregation'''
            print("multi-krum aggregation")
            try:
                f = int(method[11:])
            except:
                raise TypeError("ill-defined byzantine worker count")
            if method[11:].isdigit() == False:
                raise TypeError("f should be an int")
            if 2*f+2 > len(w):
                raise ValueError("f should be smaller")
            print(f"Krum; f = {f}")
            score = []
            for v1 in w:
                dist = 0
                tracking = []
                for v2 in w:
                    dist += norm(np.subtract(np.array(v1),np.array(v2)))**2
                    tracking.append(norm(np.subtract(np.array(v1),np.array(v2)))**2)
                dist -= sum(sorted(tracking, reverse=True)[:f+1])
                score.append(dist)
            print(score)
            idx = list(np.argpartition(np.array(score), len(w)-f))
            w_avg = np.average([w[i] for i in idx[:len(w)-f]], axis=0, weights=[nc[i] for i in idx[:len(w)-f]])         
            print(w_avg)
            self.update_global_model(w_avg)
            end = timer()
            print(self.global_model)
            print(self.global_model.get_weights())
            return copy.deepcopy(self.global_model), end-start
        elif method == 'RFA':
            '''median aggregation using smoothed Weiszfeld algorithm '''
            print("median aggregation")

            v = 0.00001
            R = 100
            w_avg = copy.deepcopy(w)
            n = sum(nc)

            for i in range(len(w)):
                for j in range(len(w[i])):
                    w_avg[i][j] = w_avg[i][j] * nc[i]
                    w_avg[i][j] = w_avg[i][j] / n

            w_avg = np.sum(w_avg, axis=0)

            print(f"MA; v = {v}")

            for r in range(R):
                nc2 = [nc[i] / max(v, norm(np.subtract(np.array(w[i]), np.array(w_avg)))) for i in range(len(w))]
                w_avg = np.average(w, axis=0, weights=nc2)
            print(w_avg)
            self.update_global_model(w_avg)
            end = timer()
            print(self.global_model)
            print(self.global_model.get_weights())
            return copy.deepcopy(self.global_model), end-start
        elif method[:14] == 'trimmed_means_':
            '''trimmed-means aggregation'''
            print("trimmed-means aggregation")
            try:
                b = float(method[14:])
            except:
                raise TypeError("ill-defined trimmed-means treshold")
            if b >= 0.5:
                raise ValueError("b should be less than 0.5")
            print(f"TM; b = {b}")

            nb_to_remove = int(len(w)*b)
            print(2*nb_to_remove)

            w_temp = []
            nc_trimmed = []
            w_avg = []

            for i in range(len(w[0])):
                w_column = list(np.array(w)[:,i])
                nc_temp = list(nc)
                #print(w_column)
                for b in range(nb_to_remove):

                    imax = w_column.index(np.max(w_column)) 
                    w_column.pop(imax)
                    nc_temp.pop(imax)

                    
                    imin = w_column.index(np.min(w_column))
                    w_column.pop(imin)
                    nc_temp.pop(imin)

                w_temp.append(w_column)
                nc_trimmed.append(nc_temp)
                w_avg.append(np.average(w_temp[i], axis=0, weights=nc_trimmed[i]))

            print(w_avg)
            self.update_global_model(w_avg)
            end = timer()
            print(self.global_model)
            print(self.global_model.get_weights())
            return copy.deepcopy(self.global_model), end-start
            
        else:
            raise ValueError("unknown aggregation method")

    def test_inference(self):
        # Returns the test accuracy and loss.
        accuracy, loss, metrics = self.global_model.test_inference(self.testing_dataset, self.test_batch_size)
        return accuracy, loss, metrics

    def validation(self):
        # Returns the test accuracy and loss. """
        accuracy, loss, metrics = self.global_model.test_inference(self.validation_dataset, self.test_batch_size)
        return accuracy, loss, metrics


