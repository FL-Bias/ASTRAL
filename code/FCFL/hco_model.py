# lenet base model for Pareto MTL
import os
import numpy as np
import argparse
import json
import pdb
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
torch.set_printoptions(profile="full")
from ctypes import sizeof
from lib2to3.pytree import Leaf
from turtle import shape
from sklearn.metrics import roc_auc_score, classification_report
from timeit import default_timer as timer
from hco_lp import HCO_LP
from dataload import LoadDataset
from po_lp import PO_LP
from bias_metrics import *
import robustness_methods as rob


class RegressionTrain(torch.nn.Module):

    def __init__(self, model, disparity_type = "DP", dataset  = "adult", warmup= "no"):
        super(RegressionTrain, self).__init__()
        self.model = model
        self.loss = nn.BCELoss()
        self.disparity_type = disparity_type
        self.dataset = dataset
        self.warmup = warmup
        
        #* warmup
        if warmup == "Adult-":
            #with open('../../astral-experiments/warmup_models_ASTRAL/logisticPytorch_Adult', "rb") as f:  # Unpickling
            with open('../../astral-experiments/warmup_models_ASTRAL/LogisticPytorch_Adult_sex_race_age_new_privileged', "rb") as f:  # Unpickling
                weights = pickle.load(f)
            dict={'layers.0.weight': torch.tensor([weights[:-1]]), 'layers.0.bias': torch.tensor([weights[-1]]) }
            model.load_state_dict(dict, strict=True)
        if warmup == "Adult-race":
            #with open('../../astral-experiments/warmup_models_ASTRAL/LogisticPytorch_Adult_race_new_privileged', "rb") as f:  # Unpickling
            with open('astral-experiments/warmup_models/race/logisticPytorch_Adult_race', "rb") as f:  # Unpickling
                weights = pickle.load(f)
            dict={'layers.0.weight': torch.tensor([weights[:-1]]), 'layers.0.bias': torch.tensor([weights[-1]]) }
            model.load_state_dict(dict, strict=True)
        if warmup == "KDD":
            #with open('../../astral-experiments/warmup_models_ASTRAL/LogisticPytorch_Adult_race_new_privileged', "rb") as f:  # Unpickling
            with open('../../astral-datasets/Configurations/logisticPytorch_KDD_age_gender_race', "rb") as f:  # Unpickling
                weights = pickle.load(f)
            dict={'layers.0.weight': torch.tensor([weights[:-1]]), 'layers.0.bias': torch.tensor([weights[-1]]) }
            model.load_state_dict(dict, strict=True)
        elif warmup == "MEPS":
            #with open('../../astral-experiments/warmup_models_ASTRAL/logisticPytorch_MEPS_fcfl', "rb") as f:  # Unpickling
            with open('../../astral-experiments/warmup_models_ASTRAL/LogisticPytorch_MEPS_sex_race_privileged', "rb") as f:  # Unpickling
                weights = pickle.load(f)
            dict={'layers.0.weight': torch.tensor([weights[:-1]]), 'layers.0.bias': torch.tensor([weights[-1]]) }
            model.load_state_dict(dict, strict=True)
        elif warmup == "Dutch":
            with open('../../astral-experiments/warmup_models_ASTRAL/LogisticPytorch_Dutch_sex_standard2attr',
                      "rb") as f:  # Unpickling
                weights = pickle.load(f)
            dict = {'layers.0.weight': torch.tensor([weights[:-1]]), 'layers.0.bias': torch.tensor([weights[-1]])}
            model.load_state_dict(dict, strict=True)
        else:
            pass

    def forward(self, x, y, A):
        ys_pre = self.model(x).flatten()
        ys = torch.sigmoid(ys_pre)
        hat_ys1 = F.relu(ys - 0.5) / torch.max(torch.tensor(0.00001).cuda(),(ys - 0.5))
        hat_ys = (ys >=0.5).float()
        task_loss = self.loss(ys, y)
        accs = torch.mean((hat_ys == y).float()).item()
        aucs = roc_auc_score(y.cpu(), ys.clone().detach().cpu())
        if True:
            if self.disparity_type == "DP":
                if torch.sum(A).float() == 0:
                    pred_dis = - torch.sum(torch.sigmoid(10 * ys_pre) * (1-A))/torch.sum(1-A)
                    disparitys = - torch.sum(hat_ys * (1-A))/torch.sum(1-A)
                    Astral_disp = ((torch.sum(hat_ys1 * A)) / (torch.sum(hat_ys1 * (1-A))/torch.sum(1-A))).detach().cpu().numpy()

                elif torch.sum(1-A).float() == 0:
                    pred_dis = torch.sum(torch.sigmoid(10 * ys_pre) * A)/torch.sum(A)
                    disparitys = torch.sum(hat_ys * A)/torch.sum(A)
                    Astral_disp = (torch.sum(hat_ys1 * A)/torch.sum(A)).detach().cpu().numpy()
                else:
                    pred_dis = torch.sum(torch.sigmoid(10 * ys_pre) * A)/torch.sum(
                        A) - torch.sum(torch.sigmoid(10 * ys_pre) * (1-A))/torch.sum(1-A)
                    disparitys = torch.sum(hat_ys * A)/torch.sum(A) - \
                        torch.sum(hat_ys * (1-A))/torch.sum(1-A)
                    Astral_disp = ((torch.sum(hat_ys1 * A)/torch.sum(A)) / (torch.sum(hat_ys1 * (1-A))/torch.sum(1-A))).detach().cpu().numpy()

            return task_loss, accs, aucs, pred_dis, disparitys, ys, Astral_disp

        else:
            print("error model in forward")
            exit()


class RegressionModel(torch.nn.Module):
    def __init__(self, n_feats, n_hidden):
        super(RegressionModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_feats, 1))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y_temp = self.layers[i](y)
            if i < len(self.layers) - 1:
                y = torch.tanh(y_temp)
            else:
                y = y_temp
        return y

class MODEL(object):
    def __init__(self, args, logger, writer):
        super(MODEL, self).__init__()

        self.dataset = args.dataset
        self.uniform_eps = bool(args.uniform_eps)
        if self.uniform_eps:
            self.eps = args.eps
        else:
            args.eps[0] = 0.0
            self.eps = [0.0, args.eps[1], args.eps[2]]

        self.max_epoch1 = args.max_epoch_stage1
        self.max_epoch2 = args.max_epoch_stage2
        self.ckpt_dir = args.ckpt_dir
        self.global_epoch = args.global_epoch
        self.log_pickle_dir = args.log_dir
        self.per_epoches = args.per_epoches
        self.factor_delta = args.factor_delta
        self.lr_delta = args.lr_delta
        self.deltas = np.array([0.,0.])
        self.deltas[0] = args.delta_l
        self.deltas[1] = args.delta_g
        self.eval_epoch = args.eval_epoch
        self.data_load(args)
        self.logger = logger
        self.logger.info(str(args))
        self.n_linscalar_adjusts = 0
        self.done_dir = args.done_dir
        self.FedAve = args.FedAve
        self.writer = writer
        self.uniform = args.uniform
        self.performence_only = args.uniform
        self.policy = args.policy
        self.disparity_type = args.disparity_type
        self.model = RegressionTrain(RegressionModel(args.n_feats, args.n_hiddens), args.disparity_type, args.dataset, args.warmup)
        self.log_train = dict()
        self.log_test = dict()
        self.log_validation = dict()
        self.baseline_type = args.baseline_type
        self.weight_fair = args.weight_fair
        self.sensitive_attr = args.sensitive_attr
        self.weight_eps = args.weight_eps
        self.aggregation_method = args.aggregation_method
        self.total_comm_upstream = 0
        self.total_comm_downstream = 0
        self.traces = args.traces
                
        if self.aggregation_method == 'fed_avg':
            '''classical average aggregation'''
            self.method_value = 0
        elif self.aggregation_method[:4] == 'NDC_':
            '''norm tresholding aggregation'''
            try:
                M = float(self.aggregation_method[4:])
            except:
                raise TypeError("ill-defined NDC treshold")
            self.method_value = 0
        elif self.aggregation_method[:11] == 'multi_krum_':
            '''multi-krum aggregation'''
            try:
                f = int(self.aggregation_method[11:])
            except:
                raise TypeError("ill-defined byzantine worker count")
            if self.aggregation_method[11:].isdigit() == False:
                raise TypeError("f should be an int")
            self.method_value =  f
 
        if torch.cuda.is_available():
            self.model.cuda()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=args.step_size, momentum=0., weight_decay=1e-4)

        _, n_params = self.getNumParams(self.model.parameters())
        self.hco_lp = HCO_LP(n=n_params, eps = self.eps)
        self.po_lp = PO_LP(n_theta=n_params, n_alpha = 1+ self.n_clients-self.method_value,  eps = self.eps[0])
        if int(args.load_epoch) != 0:
            self.model_load(str(args.load_epoch))
        self.commandline_save(args)
    def get_weights(self):
        model_weights = []
        i = 0
        for layer in self.model.modules():
            i = i+1
            if isinstance(layer, nn.Linear):
                for x in layer.weight[0] :
                    model_weights.append(x.item())
                model_weights.append(layer.bias[0].item())
        return model_weights
        
    def commandline_save(self, args):
        with open(args.commandline_file, "w") as f:
            json.dump(args.__dict__, f, indent =2)

    def getNumParams(self, params):
        numParams, numTrainable = 0, 0
        for param in params:
            npParamCount = np.prod(param.data.shape)
            numParams += npParamCount
            if param.requires_grad:
                numTrainable += npParamCount
        return numParams, numTrainable

    def model_load(self, ckptname='last'):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                self.logger.info("=> no checkpoint found")
                exit()
            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])
        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['model'])
            self.optim.load_state_dict(checkpoint['optim'])
            self.logger.info("=> loaded checkpoint '{} (epoch {})'".format(filepath, self.global_epoch))        
        else:
            self.logger.info("=> no checkpoint found at '{}'".format(filepath))

    def model_save(self, ckptname = None):
        states = {'epoch':self.global_epoch,
                  'model':self.model.state_dict(),
                  'optim':self.optim.state_dict()}
        if ckptname == None:
            ckptname = str(self.global_epoch)
        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        os.makedirs(self.ckpt_dir, exist_ok = True)
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        self.logger.info("=> saved checkpoint '{}' (epoch {})".format(filepath, self.global_epoch))

    def data_load(self, args):
        self.client_train_loaders, self.client_test_loaders, self.client_validation_loaders = LoadDataset(args)
        self.n_clients = len(self.client_train_loaders)
        self.iter_train_clients = [enumerate(i) for i in self.client_train_loaders]
        self.iter_test_clients = [enumerate(i) for i in self.client_test_loaders]
        self.iter_validation_clients = [enumerate(i) for i in self.client_validation_loaders]

    def valid_stage1(self,  if_train = False, epoch = -1):
        with torch.no_grad():
            losses = []
            accs = []
            diss = []
            pred_diss = []
            aucs = []
            Astral_disparity = []
            fairness_rep, fairness_rep1, fairness_rep2, fairness_rep3 =[], [], [], []
            if if_train:
                loader = self.client_train_loaders
            if (self.traces == 'test' or self.traces == 'both'):
                loader = self.client_test_loaders
                for client_idx, client_test_loader in enumerate(loader):
                    valid_loss = []
                    valid_accs = []
                    valid_diss = []
                    valid_pred_dis = []
                    valid_auc = []
                    astral_disparity = []
                    features_fairness = np.array([])
                    labels_fairness = np.array([])
                    pred_labels_fairness = np.array([])
                    for it, (X, Y, A) in enumerate(client_test_loader):
                        X = X.float()
                        Y = Y.float()
                        A = A.float()

                        (features, labels, SA) = ( X.cpu(), Y.cpu(), A.cpu())
                        features_fairness = np.concatenate((features_fairness, features), axis=0) if features_fairness.size else features
                        labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels

                        if torch.cuda.is_available():
                            X = X.cuda()
                            Y = Y.cuda()
                            A = A.cuda()
                        loss, acc, auc, pred_dis, disparity, pred_y, Astral_disp = self.model(X, Y, A)
                        pred_labels = (pred_y >=0.5).float().cpu()
                        pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1)), axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1)
                        pred_labels = pred_labels.view(len(pred_labels), -1)

                        valid_loss.append(loss.item())
                        valid_accs.append(acc)
                        valid_diss.append(disparity.item())
                        astral_disparity.append(Astral_disp)
                        valid_pred_dis.append(pred_dis.item())
                        valid_auc.append(auc)
                    assert len(valid_auc)==1
                    losses.append(np.mean(valid_loss))
                    accs.append(np.mean(valid_accs))
                    diss.append(np.mean(valid_diss))
                    Astral_disparity.append(np.mean(astral_disparity))
                    pred_diss.append(np.mean(valid_pred_dis))
                    aucs.append(np.mean(valid_auc))

                    if type(labels_fairness) == torch.Tensor and type(pred_labels_fairness) == torch.Tensor and type(features_fairness) == torch.Tensor:
                        labels_fairness = labels_fairness.numpy()
                        pred_labels_fairness = pred_labels_fairness.numpy()
                        features_fairness = features_fairness.numpy()

#                    fairness_rep.append(fairness_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), np.array(self.get_weights()), self.sensitive_attr))
#                    fairness_rep1.append(fairness_report_indice(features_fairness, labels_fairness.astype(int),
#                                                                pred_labels_fairness.astype(int),
#                                                                np.array(self.get_weights()),
#                                                                -1))
#                    fairness_rep2.append(fairness_report_indice(features_fairness, labels_fairness.astype(int),
#                                                        pred_labels_fairness.astype(int), np.array(self.get_weights()),
#                                                        -2))
#                    fairness_rep3.append(fairness_report_indice(features_fairness, labels_fairness.astype(int),
#                                                        pred_labels_fairness.astype(int), np.array(self.get_weights()),
#                                                        -3))

                self.log_test[str(epoch)] = { "client_losses_t": losses, "pred_client_disparities_t": pred_diss, "client_accs_t": accs, "client_aucs_t": aucs, "client_disparities_t": diss, "max_losses_t": [max(losses), max(diss)], "Astral_disparity_t": Astral_disparity, "Fairness report": fairness_rep,
                                              "Fairness report 1": fairness_rep1,      "Fairness report 2": fairness_rep2, "Fairness report 3": fairness_rep3}
            
            if if_train:
                for i, item in enumerate(losses):
                    self.writer.add_scalar("valid_train/loss_:"+str(i),  item , epoch)
                    self.writer.add_scalar("valid_trains/acc_:"+str(i),  accs[i], epoch)
                    self.writer.add_scalar("valid_trains/auc_:"+str(i),  aucs[i], epoch)
                    self.writer.add_scalar("valid_trains/disparity_:"+str(i),  diss[i], epoch)
                    self.writer.add_scalar("valid_trains/pred_disparity_:"+str(i),  pred_diss[i], epoch)

            elif (self.traces == 'test' or self.traces == 'both'):
                for i, item in enumerate(losses):
                    self.writer.add_scalar("test/loss_:"+str(i),  item , epoch)
                    self.writer.add_scalar("test/acc_:"+str(i),  accs[i], epoch)
                    self.writer.add_scalar("test/auc_:"+str(i),  aucs[i], epoch)
                    self.writer.add_scalar("test/disparity_:"+str(i),  diss[i], epoch)
                    self.writer.add_scalar("test/pred_disparity_:"+str(i),  pred_diss[i], epoch)
                    self.writer.add_scalar("test/Astral_disparity_:"+str(i),  Astral_disparity[i], epoch)
#                    self.writer.add_text("test/fairness_report_:" + str(i),json.dumps(fairness_rep[i]), epoch)
#                    self.writer.add_text("test/fairness_report1_:" + str(i),json.dumps(fairness_rep1[i]), epoch)
#                    self.writer.add_text("test/fairness_report2_:" + str(i),json.dumps(fairness_rep2[i]), epoch)
#                    self.writer.add_text("test/fairness_report3_:" + str(i),json.dumps(fairness_rep3[i]), epoch)
            if (self.traces == 'valid' or self.traces == 'both'):
                losses = []
                accs = []
                diss = []
                pred_diss = []
                aucs = []
                Astral_disparity = []
                fairness_rep, fairness_rep1, fairness_rep2, fairness_rep3 = [], [], [], []
                if if_train:
                    loader = self.client_train_loaders
                else:
                    loader = self.client_validation_loaders
                for client_idx, client_valid_loader in enumerate(loader):
                    valid_loss = []
                    valid_accs = []
                    valid_diss = []
                    valid_pred_dis = []
                    valid_auc = []
                    astral_disparity = []
                    features_fairness = np.array([])
                    labels_fairness = np.array([])
                    pred_labels_fairness = np.array([])
                    for it, (X, Y, A) in enumerate(client_valid_loader):
                        X = X.float()
                        Y = Y.float()
                        A = A.float()

                        (features, labels, SA) = (X.cpu(), Y.cpu(), A.cpu())
                        features_fairness = np.concatenate((features_fairness, features), axis=0) if features_fairness.size else features
                        labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels

                        if torch.cuda.is_available():
                            X = X.cuda()
                            Y = Y.cuda()
                            A = A.cuda()
                        loss, acc, auc, pred_dis, disparity, pred_y, Astral_disp = self.model(X, Y, A)

                        pred_labels = (pred_y >=0.5).float().cpu()
                        pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1)), axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1)
                        pred_labels = pred_labels.view(len(pred_labels), -1)

                        valid_loss.append(loss.item())
                        valid_accs.append(acc)
                        valid_diss.append(disparity.item())
                        astral_disparity.append(Astral_disp)
                        valid_pred_dis.append(pred_dis.item())
                        valid_auc.append(auc)
                    assert len(valid_auc)==1
                    losses.append(np.mean(valid_loss))
                    accs.append(np.mean(valid_accs))
                    diss.append(np.mean(valid_diss))
                    Astral_disparity.append(np.mean(astral_disparity))
                    pred_diss.append(np.mean(valid_pred_dis))
                    aucs.append(np.mean(valid_auc))

                    if type(labels_fairness) == torch.Tensor and type(pred_labels_fairness) == torch.Tensor and type(features_fairness) == torch.Tensor:
                        labels_fairness = labels_fairness.numpy()
                        pred_labels_fairness = pred_labels_fairness.numpy()
                        features_fairness = features_fairness.numpy()

#                    fairness_rep.append(fairness_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), np.array(self.get_weights()), self.sensitive_attr))
#                    fairness_rep1.append(fairness_report_indice(features_fairness, labels_fairness.astype(int),
#                                                        pred_labels_fairness.astype(int), np.array(self.get_weights()),
#                                                        -1))
#                    fairness_rep2.append(fairness_report_indice(features_fairness, labels_fairness.astype(int),
#                                                        pred_labels_fairness.astype(int), np.array(self.get_weights()),
#                                                        -2))
#                    fairness_rep3.append(fairness_report_indice(features_fairness, labels_fairness.astype(int),
#                                                        pred_labels_fairness.astype(int), np.array(self.get_weights()),
#                                                        -3))

                self.log_validation[str(epoch)] = { "client_losses_t": losses, "pred_client_disparities_t": pred_diss, "client_accs_t": accs, "client_aucs_t": aucs, "client_disparities_t": diss, "max_losses_t": [max(losses), max(diss)], "Astral_disparity_t": Astral_disparity, "Fairness report": fairness_rep,
                                                    "Fairness report 1": fairness_rep1 ,   "Fairness report 2": fairness_rep2, "Fairness report 3": fairness_rep3}
                self.logger.info(self.log_validation[str(epoch)])
                for i, item in enumerate(losses):
                    self.writer.add_scalar("valid/loss_:"+str(i),  item , epoch)
                    self.writer.add_scalar("valid/acc_:"+str(i),  accs[i], epoch)
                    self.writer.add_scalar("valid/auc_:"+str(i),  aucs[i], epoch)
                    self.writer.add_scalar("valid/disparity_:"+str(i),  diss[i], epoch)
                    self.writer.add_scalar("valid/pred_disparity_:"+str(i),  pred_diss[i], epoch)
                    self.writer.add_scalar("valid/Astral_disparity_:"+str(i),  Astral_disparity[i], epoch)
#                    self.writer.add_text("valid/fairness_report_:" + str(i), json.dumps(fairness_rep[i]), epoch)
#                    self.writer.add_text("valid/fairness_report1_:" + str(i), json.dumps(fairness_rep1[i]), epoch)
#                    self.writer.add_text("valid/fairness_report2_:" + str(i), json.dumps(fairness_rep2[i]), epoch)
#                    self.writer.add_text("valid/fairness_report3_:" + str(i), json.dumps(fairness_rep3[i]), epoch)
            return losses, accs, diss, pred_diss, aucs
   
    def soften_losses(self, losses, delta):
        '''if id == None:
            id = [i for i in range(len(losses))]'''
        losses_list = torch.stack(losses)
        loss = torch.max(losses_list)
        alphas = F.softmax((losses_list - loss)/delta)
        alpha_without_grad = (Variable(alphas.data.clone(), requires_grad=False)) 
        return alpha_without_grad, loss

    def train(self):     
        if self.baseline_type == "none":
            if self.policy == "alternating":
                start_epoch = self.global_epoch
                for epoch in range(start_epoch , self.max_epoch1 + self.max_epoch2):
                    if int(epoch/self.per_epoches) %2 == 0:
                        self.train_stage1(epoch)
                    else:
                        self.train_stage2(epoch)

            elif self.policy == "two_stage":
                if self.uniform:
                    self.performence_only  = True
                else:
                    self.performence_only  = False
                start_epoch = self.global_epoch
                for epoch in range(start_epoch, self.max_epoch1):
                    self.train_stage1(epoch)

                for epoch in range(self.max_epoch1, self.max_epoch2 + self.max_epoch1):
                    self.train_stage2(epoch)

    def save_log(self):
        with open(os.path.join(self.log_pickle_dir, "train_log.pkl"), "wb") as f:
            pickle.dump(self.log_train, f)
        if (self.traces == 'test' or self.traces=='both'):
            with open(os.path.join(self.log_pickle_dir, "test_log.pkl"), "wb") as f:
                pickle.dump(self.log_test, f)    
        if (self.traces == 'valid' or self.traces=='both'):
            with open(os.path.join(self.log_pickle_dir, "valid_log.pkl"), "wb") as f:
                pickle.dump(self.log_validation, f)
        os.makedirs(self.done_dir, exist_ok = True)
        self.logger.info("logs have been saved")   

    def train_stage1(self, epoch):
        self.model.train()
        self.optim.zero_grad()
        grads_performance = []
        grads_disparity = []
        losses_data = []
        disparities_data = []
        pred_disparities_data = []
        accs_data = []
        aucs_data = []
        client_losses = []
        client_disparities = []
       
        comm_upstream = 0 #Initialize to 0 each start of one FL round
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement()
     
        comm_downstream= param_size*self.n_clients
        for client_idx in range(self.n_clients):
            try:
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            except StopIteration:
                self.iter_train_clients[client_idx] = enumerate(self.client_train_loaders[client_idx])
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            X = X.float()
            Y = Y.float()
            A = A.float()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
                A = A.cuda()
            loss, acc, auc, pred_dis, dis, pred_y, Astral_disp = self.model(X, Y, A)
            self.iter_train_clients[client_idx]
############################################################## GPU version
            loss.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
            grad = torch.stack(grad)
            comm_upstream+= grad.nelement()
            grads_performance.append(grad)
            self.optim.zero_grad()
            torch.abs(pred_dis).backward(retain_graph=True)
            if self.performence_only:
                self.optim.zero_grad()
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
            grad = torch.stack(grad)
            comm_upstream+= grad.nelement()
            grads_disparity.append(grad)
            self.optim.zero_grad()   
            if self.uniform_eps:
                client_disparities.append(torch.abs(pred_dis))
                specific_eps = 0

            client_losses.append(loss)
            losses_data.append(loss.item())
            disparities_data.append(dis)
            pred_disparities_data.append(pred_dis.item())
            accs_data.append(acc)
            aucs_data.append(auc)
        
        #Measure how much data is sent to the server
        comm_upstream+= len(client_losses)
        comm_upstream+= len(client_disparities)
        start = timer()
        #* compatibility with mKrum
        updt_grads, ids = rob.select_grads(grads_performance, method=self.aggregation_method)
        try:
            for i in range(len(ids)):
                client_losses.pop(ids[i])
                losses_data.pop(ids[i])
                disparities_data.pop(ids[i])
                pred_disparities_data.pop(ids[i])
                accs_data.pop(ids[i])
                aucs_data.pop(ids[i])
                client_disparities.pop(ids[i])
                grads_performance.pop(ids[i])
                grads_disparity.pop(ids[i])
        except:
            pass
        try:
            if updt_grads != None:
                grads_performance = updt_grads
        except:
            pass  
        
        alphas_l, loss_max_performance = self.soften_losses(client_losses, self.deltas[0])
        loss_max_performance = loss_max_performance.item()
        alphas_g, loss_max_disparity = self.soften_losses(client_disparities, self.deltas[1])
        loss_max_disparity = loss_max_disparity.item()
        losses = np.array(losses_data)
            # a batch of [loss_c1, loss_c2, ... loss_cn], [grad_c1, grad_c2, grad_cn]
        if self.FedAve:
            preference = np.array([1 for i in range(self.n_clients)])
            alpha = preference / preference.sum()
            self.n_linscalar_adjusts += 1          
        else:
            try:
                    # Calculate the alphas from the LP solver                
                alphas_l = alphas_l.view(1, -1)
                grad_l = alphas_l @ torch.stack(grads_performance) 

                alphas_g = alphas_g.view(1, -1)
                grad_g = alphas_g @  torch.stack(grads_disparity)  
                alpha, deltas = self.hco_lp.get_alpha([loss_max_performance, loss_max_disparity], grad_l, grad_g, self.deltas, self.factor_delta, self.lr_delta) 
                if torch.cuda.is_available():
                    alpha = torch.from_numpy(alpha.reshape(-1)).cuda()
                else:
                    alpha = torch.from_numpy(alpha.reshape(-1))
                self.deltas = deltas
                alpha = alpha.view(-1)
            except Exception as e:
                print(e)
                exit()
############################################################## GPU version
        # 2. Optimization step 
              
        self.optim.zero_grad() 
        weighted_loss1 = torch.sum(torch.stack(client_losses)*alphas_l)     
        weighted_loss2 = torch.sum(torch.stack(client_disparities)*alphas_g)
        weighted_loss = torch.sum(torch.stack([weighted_loss1, weighted_loss2]) * alpha)
        weighted_loss.backward()
        self.optim.step()
        end = timer()
        
        # 2. apply gradient dierctly
        ############################
        self.writer.add_scalar("server_time",  end - start , epoch)
        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            self.model.eval()
            losses, accs, client_disparities, pred_dis, aucs = self.valid_stage1(if_train = False, epoch = epoch)
            if epoch != 0:
                self.model_save()   
        self.global_epoch+=1
        self.log_train[str(epoch)]={ "server_time": end - start, "comm_downstream": comm_downstream, "comm_upstream": comm_upstream}
        self.writer.add_scalar("comm_downstream",  comm_downstream , epoch)
        self.writer.add_scalar("comm_upstream",  comm_upstream , epoch)
        self.total_comm_downstream+= comm_downstream
        self.total_comm_upstream+= comm_upstream
        
    def train_stage2(self, epoch):
        self.model.train()
        grads_performance = []
        grads_disparity = []
        disparities_data = []
        client_losses = []
        client_disparities = []
        losses_data = []
        accs_data = []
        pred_diss_data = []
        aucs_data = []
        comm_upstream = 0 #Initialize to 0 each start of one FL round
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement()
        
        comm_downstream= param_size*self.n_clients
        
        for client_idx in range(self.n_clients):
            try:
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            except StopIteration:
                self.iter_train_clients[client_idx] = enumerate(self.client_train_loaders[client_idx])
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            X = X.float()
            Y = Y.float()
            A = A.float()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
                A = A.cuda()

            loss, acc, auc, pred_dis, dis, pred_y, Astral_disp = self.model(X, Y, A)
            loss.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
            grad = torch.stack(grad)
            comm_upstream+= grad.nelement()
            grads_performance.append(grad)
            self.optim.zero_grad()
            torch.abs(pred_dis).backward(retain_graph=True)
            if self.performence_only:
                self.optim.zero_grad() 
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
            grad = torch.stack(grad)
            comm_upstream+= grad.nelement()
            grads_disparity.append(grad)
            self.optim.zero_grad()  

            client_losses.append(loss)
            client_disparities.append(torch.abs(pred_dis))
            disparities_data.append(dis) 
            accs_data.append(acc)
            losses_data.append(loss.item())
            pred_diss_data.append(pred_dis.item())
            aucs_data.append(auc)
        
        #Measure how much data is sent to the server
        comm_upstream+= len(client_disparities)
        comm_upstream+= len(client_losses)
        start = timer()
        #* compatibility with mKrum
        updt_grads, ids = rob.select_grads(grads_performance, method=self.aggregation_method)
        try:
            for i in range(len(ids)):
                client_losses.pop(ids[i])
                losses_data.pop(ids[i])
                disparities_data.pop(ids[i])
                pred_diss_data.pop(ids[i])
                accs_data.pop(ids[i])
                aucs_data.pop(ids[i])
                client_disparities.pop(ids[i])
                grads_performance.pop(ids[i])
                grads_disparity.pop(ids[i])
        except:
            pass
        try:
            if updt_grads != None:
                grads_performance = updt_grads
        except:
            pass

        alpha_disparity, max_disparity = self.soften_losses(client_disparities, self.deltas[1])
        client_pred_disparity = torch.sum(alpha_disparity * torch.stack(client_disparities))
        grad_disparity = alpha_disparity.view(1, -1) @ torch.stack(grads_disparity)
        grads_performance = torch.stack(grads_performance)

        if max_disparity.item() < self.eps[0]:
            grad_disparity = torch.zeros_like(grad_disparity, requires_grad= False)
        grad_performance = torch.mean(grads_performance, dim = 0, keepdim=True)
        grads = torch.cat((grads_performance, grad_disparity), dim = 0)

##########################################GPU()
        grad_performance = grad_performance.t()
        alpha, gamma = self.po_lp.get_alpha(grads, grad_performance, grads.t())
        if torch.cuda.is_available():
            alpha = torch.from_numpy(alpha.reshape(-1)).cuda()
        else:
            alpha = torch.from_numpy(alpha.reshape(-1))

        client_losses.append(client_pred_disparity)
        weighted_loss = torch.sum(torch.stack(client_losses) * alpha)
        weighted_loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        end = timer()
        self.writer.add_scalar("server_time",  end - start , epoch)

        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            self.model.eval()
            losses, accs, client_disparities, pred_dis, aucs = self.valid_stage1(if_train = False, epoch = epoch)
            if epoch != 0:
                self.model_save()   
        self.global_epoch+=1
        self.log_train[str(epoch)]={ "server_time": end - start, "comm_downstream": comm_downstream, "comm_upstream": comm_upstream}
        self.writer.add_scalar("comm_downstream",  comm_downstream , epoch)
        self.writer.add_scalar("comm_upstream",  comm_upstream , epoch)
        self.total_comm_downstream+= comm_downstream
        self.total_comm_upstream+= comm_upstream
