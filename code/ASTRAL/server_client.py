import numpy as np
import copy
import time
from multiprocessing.sharedctypes import Value
import torch
# from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from torch.autograd import Variable
from functools import reduce
from numpy.linalg import norm
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore')
from models import *
from Astral_optim import *

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
    
    def test_inference_SPD_fairfed(self, testing_dataset):
        """Returns the inference accuracy and loss."""
        accuracy, loss, metrics, n_yz = self.local_model.test_inference_SPD_fairfed(testing_dataset)
        return accuracy, loss, metrics, n_yz
    
    def test_inference_EOD_fairfed(self, testing_dataset):
        """Returns the inference accuracy and loss."""
        accuracy, loss, metrics, n_yz = self.local_model.test_inference_EOD_fairfed(testing_dataset)
        return accuracy, loss, metrics, n_yz

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

    def update_global_model_by_state(self, state):
        self.global_model.model.load_state_dict(state)

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
        

        if method == 'fed_avg' and ( isinstance(self.global_model, ResnetPytorch) or  isinstance(self.global_model, MLPPytorch)):

            print("classical average aggregation")
           
            n = sum(nc)
            print(nc,n)
            w_avg = {}
            w_avg = copy.deepcopy(w[0])

            for key in w_avg.keys():
                w_avg[key] = w_avg[key] * nc[0]

            for key in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[key] += (nc[i] * w[i][key])
                w_avg[key] = torch.div(w_avg[key], sum(nc))
            self.update_global_model_by_state(w_avg)
            end = timer()
            return copy.deepcopy(self.global_model), end-start

        elif method == 'fed_avg':
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
        print('teeeeeeeeeest')
        print(len(self.testing_dataset))
        accuracy, loss, metrics= self.global_model.test_inference(self.testing_dataset, self.test_batch_size)
        return accuracy, loss, metrics

    def validation(self):
        # Returns the test accuracy and loss. """
        print('vaaaaaaaaaalid')
        print(len(self.validation_dataset))
        accuracy, loss, metrics = self.global_model.test_inference(self.validation_dataset, self.test_batch_size)
        return accuracy, loss, metrics
    
    def test_inference_fairfed(self):
        # Returns the test accuracy and loss.
        accuracy, loss, metrics, n_yz_test = self.global_model.test_inference_fairfed_DI(self.testing_dataset, self.test_batch_size)
        return accuracy, loss, metrics, n_yz_test

    def validation_fairfed(self):
        # Returns the test accuracy and loss. """
        accuracy, loss, metrics,n_yz_valid = self.global_model.test_inference_fairfed_DI(self.validation_dataset, self.test_batch_size)
        return accuracy, loss, metrics,n_yz_valid

    def Astral_optim_aggregation2(self, local_models, clients_weights, nc, nb_iterations, eps, FL_rounds):
        hco_lp = HCO_LP(len(clients_weights[0]), len(clients_weights), eps)
        hco_lp2 = HCO_LP2(len(clients_weights[0]), len(clients_weights), eps)
        losses_data = []
        disparities_data = []
        pred_disparities_data = []
        accs_data = []
        aucs_data = []
        client_losses = []

        X = self.get_validation_dataset().data.float()
        Y = self.get_validation_dataset().target.float()
        A = self.get_validation_dataset().sa.float()
        if torch.cuda.is_available():
            X = X.cuda()
            Y = Y.cuda()
            A = A.cuda()

        m = len(clients_weights)
        alphas = [1/m for i in range(m)]

        for i in range(nb_iterations):
            grads_performance = []
            grads_disparity = []
            client_disparities = []
            self.weighted_average_weights(clients_weights, alphas, 'fed_avg')

            A = A.view(len(A), -1)
            Y = Y.view(len(Y), -1)

            loss, acc, auc, pred_dis, dis, pred_y, _ = self.global_model.model(X, Y, A)
            ############################################################## GPU version
            loss.backward(retain_graph=True)
            grad_ = []
            for param in self.global_model.model.parameters():
                if param.grad is not None:
                    grad_.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            grad_ = torch.stack(grad_)
            grads = []
            for j in range(len(local_models)):
                grads.extend(Variable(torch.mul(torch.tensor(grad_[0]), clients_weights[j][0] ).clone().flatten(), requires_grad = False))
            grads_per = torch.stack(grads)
            grads_performance.append(grads_per)
            grads_performance = torch.stack(grads_performance)
            self.global_model.optimizer.zero_grad()


            torch.abs(pred_dis).backward(retain_graph=True)
            grad_ = []
            for param in self.global_model.model.parameters():
                if param.grad is not None:
                    grad_.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            grads = []
            for j in range(len(local_models)):
                grads.extend(Variable(torch.mul(torch.tensor(grad_[0]), clients_weights[j][0] ).clone().flatten(), requires_grad = False))
            grads_dis = torch.stack(grads)
            grads_disparity.append(grads_dis)
            grads_disparity = torch.stack(grads_disparity)
            client_disparities.append(torch.abs(pred_dis))
            self.global_model.optimizer.zero_grad()
            client_losses = loss
            losses_data = loss.item()
            alphas_l = Variable(loss.data.clone(), requires_grad=False)
            loss_max_performance = loss
            alphas_g = Variable(torch.tensor(dis).data.clone(), requires_grad=False)
            loss_max_disparity = torch.tensor(dis)
            losses = np.array(losses_data)
            # a batch of [loss_c1, loss_c2, ... loss_cn], [grad_c1, grad_c2, grad_cn]
            # Calculate the alphas from the LP solver
            alphas_l = alphas_l.view(1, -1) # global loss
            # ! aggregation methods
            grad_l = alphas_l @ grads_performance
            alphas_g = alphas_g.view(1, -1) # global bias
            grad_g = alphas_g @ grads_disparity
            delta = torch.tensor([grads_per.numpy(), grads_dis.numpy()])
            a = hco_lp2.get_alpha([loss_max_performance, loss_max_disparity], grad_g, grad_l)
            a = hco_lp2.get_alpha([loss_max_performance, loss_max_disparity], grads_disparity, grads_performance)
            if torch.cuda.is_available():
                a = torch.from_numpy(a.reshape(-1)).cuda()
            else:
                a = torch.from_numpy(a.reshape(-1))
            a = a.view(-1)
            ############################################################## GPU version
        # 2. Optimization step
            print("D : {}".format(a))
            alphas = (torch.tensor(alphas) + 100000000* a).view(-1).numpy()
            print("ALPHA : {}".format(alphas))
            print(np.sum(alphas))
            print(self.global_model.get_weights())
            print('ITERATION {}'.format(i))
        return(copy.deepcopy(self.global_model))

    def Astral_optim_aggregation(self, metric_name, local_models, clients_weights, nc, FL_round, byzantine, BiasMitigation_info,clientvalid_metrics):
        start=timer()

        if ((BiasMitigation_info['variant'] == "differential_evolution")): 
            # CURRENT ASTRAL-Hrt optimization method
            try:
                workers = BiasMitigation_info['worker']
            except:
                workers = -1
            try:
                maxiter = BiasMitigation_info['maxiter']
            except:
                maxiter = 1000
            try:
                popsize = BiasMitigation_info['popsize']
            except:
                popsize = 15
            try:
                tol = BiasMitigation_info['tol']
            except:
                tol = 0.01
            try:
                mutation = tuple(BiasMitigation_info['mutation'])
            except:
                mutation = (0.5,1)
            try:
                recombination = BiasMitigation_info['recombination']
            except:
                recombination = 0.7
            opt = DE_alpha_search(metric_name, copy.deepcopy(clients_weights), self.global_model_variable,self.validation_dataset,nc,BiasMitigation_info['eps'],mutation,recombination, workers,maxiter,popsize,tol)
            a = opt.get_alpha()
            print(a)
            self.weighted_average_weights(clients_weights, a, 'fed_avg', byzantine)
            end=timer()
            return (copy.deepcopy(self.global_model)), end-start
        elif BiasMitigation_info['variant'] == "differential_evolution_rwgt":
            # other explored method using differential evolution to find parameters of fedavg+rwgt (not studied)
            opt = DE_alpha_Rwgt_search(metric_name, copy.deepcopy(clients_weights), self.global_model_variable,self.validation_dataset,nc,BiasMitigation_info['eps'],clientvalid_metrics,nc)
            a = opt.get_alpha()
            if a == 0 :
                a=nc
            print(a)
            end=timer()
            return (copy.deepcopy(self.global_model)), end-start
        elif ((BiasMitigation_info['variant'] == "CMA")):
            try:
                workers = BiasMitigation_info['worker']
            except:
                workers = -1
            try:
                maxiter = BiasMitigation_info['maxiter']
            except:
                maxiter = 1000
            opt = CMAES_alpha_search(metric_name, copy.deepcopy(clients_weights), self.global_model_variable,self.validation_dataset,nc,BiasMitigation_info['eps'],workers,maxiter)
            a = opt.get_alpha()
            if a == 0 :
                end=timer()
                return (copy.deepcopy(self.global_model)), end-start



        X = self.get_validation_dataset().data.float()
        Y = self.get_validation_dataset().target.float()
        A = self.get_validation_dataset().sa.float()
        X_without_sa = self.get_validation_dataset().data_without_sa.float()
        nb_sa = self.get_validation_dataset().nb_sa
        Theta = torch.tensor(clients_weights)
        A_ = []
        if nb_sa == 1:
            A_.append(A.view(len(A), -1))
        else:
            for i in range(nb_sa):
                A_.append(A[i].view(len(A[i]), -1))
        Y_ = Y.view(len(Y), -1)

        if  BiasMitigation_info['variant'] == "fair_obj_loss_const":
            _, _, _, _, _, Y_pred, _ = self.global_model.model(X, Y_, A_)
            opt = fair_obj_loss_const_OPT(len(X), len(clients_weights[0]), len(clients_weights), X, Y, Y_pred, A, Theta, self.fed_avg_weights, BiasMitigation_info['eps'], bool(BiasMitigation_info['logs']))
            a = opt.get_alpha()
        elif BiasMitigation_info['variant'] == "loss_obj_fair_const":
            _, _, _, _, _, Y_pred, _ = self.global_model.model(X, Y_, A_)
            opt = loss_obj_fair_const_OPT(len(X), len(clients_weights[0]), len(clients_weights), X, Y, Y_pred, A, Theta, self.fed_avg_weights, BiasMitigation_info['eps'], bool(BiasMitigation_info['logs']), nb_sa)
            a = opt.get_alpha()
        elif BiasMitigation_info['variant'] == "loss_obj_fair_const+":
            _, _, _, _, _, Y_pred, _ = self.global_model.model(X, Y_, A_)
            opt = loss_obj_fair_const_OPT_PLUS(len(X), len(clients_weights[0]), len(clients_weights), X, Y, Y_pred, A, Theta, self.fed_avg_weights, BiasMitigation_info['eps'], bool(BiasMitigation_info['logs']))
            a = opt.get_alpha()
        elif BiasMitigation_info['variant'] == "wu_surrogate_risk_difference":
            _, _, _, _, _, Y_pred, _ = self.global_model.model(X, Y_, A_)
            opt = wu_surrogate_risk_difference(len(X), len(clients_weights[0]), len(clients_weights), X, Y, Y_pred, A,  X_without_sa, Theta, y_var = BinaryVariable(name= self.validation_dataset.target_var_name, pos=1, neg=0), s_var = BinaryVariable(name =self.validation_dataset.sensitive_att, pos=1, neg=0), logs =  bool(BiasMitigation_info['logs']), kappa_name=BiasMitigation_info['kappa_name'], delta_name=BiasMitigation_info['delta_name'], phi_name=BiasMitigation_info['phi_name'])
            a = opt.get_alpha()
        elif BiasMitigation_info['variant'] == "bayesian_grid_search":
            # Explored derivative-free optimization method 
            opt = Bayesian_alpha_search(copy.deepcopy(clients_weights), self.global_model_variable,self.validation_dataset,self.fed_avg_weights,BiasMitigation_info['eps'])
            a = opt.get_alpha()

        else:
            print("Variant {} not supported by ASTRAL_OPT.".format(BiasMitigation_info.variant))
            end=timer()
            return(copy.deepcopy(self.global_model)), end-start
        print(a)

        if BiasMitigation_info['variant'] == "loss_obj_fair_const+":
            self.weighted_average_weights(clients_weights, a, 'loss_obj_fair_const+', byzantine)
        else :
            self.weighted_average_weights(clients_weights, a, 'fed_avg', byzantine)
        end=timer()
        return (copy.deepcopy(self.global_model)), end-start
