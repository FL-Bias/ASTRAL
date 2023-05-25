import copy
import numpy as np
import cvxpy as cp
import cvxopt
import math
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import pandas as pd
import argparse
import json
import pdb
import pickle
from lib2to3.pytree import Leaf
from turtle import shape
from sklearn.metrics import roc_auc_score, classification_report
import cma
# import black_box as bb
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from scipy.optimize import NonlinearConstraint, differential_evolution
from timeit import default_timer as timer
from Utils.attribute import *
#from Utils.classifiers import *
from Utils.functions import *


class DE_alpha_search(object):
    """ASTRAL-Hrt optimization method implementation using Differential Evolution"""
    def __init__(self,
                metric_name,
                Theta,
                global_model_variable,
                validation_dataset,
                fedavg_w,
                threshold,
                mutation = (0.5,1),
                recombination = 0.7,
                workers=-1,
                maxiter=1000,
                popsize=15,
                tol=0.01,
                test_batch_size=64):
        self.metric_name = metric_name
        self.Theta = Theta
        self.global_model_variable = copy.deepcopy(global_model_variable)
        self.validation_dataset = validation_dataset
        self.fedavg_w = fedavg_w
        self.threshold = threshold
        self.old_is_fair = False
        self.test_batch_size = test_batch_size
        self.workers=workers
        self.maxiter=maxiter
        self.popsize=popsize
        self.tol=tol
        self.mutation = mutation
        self.recombination = recombination
        self.calls = 0
        if isinstance(self.validation_dataset.sensitive_att,list):
            self.constraints="multiple"
        else:
            self.constraints="single"
        if self.metric_name == "Statistical Parity Difference":
            self.test_bias_quick = "test_inference_quick"
        elif self.metric_name == "Equal Opportunity Difference":
            self.test_bias_quick = "test_inference_quick_EOD"
        elif self.metric_name == "Discrimination Index":
            self.test_bias_quick = "test_inference_quick_discr_idx"
    def compute_bias_computed_global_model(self,vars):
        nc = vars
        w_avg = copy.deepcopy(self.Theta)
        n = sum(nc)
        for i in range(len(self.Theta)):
            for j in range(len(self.Theta[i])):
                w_avg[i][j] = w_avg[i][j] * nc[i]
                w_avg[i][j] = w_avg[i][j] / n
        w_avg = np.sum(w_avg, axis=0)
        global_model= copy.deepcopy(self.global_model_variable)
        global_model.set_weights(w_avg)
        try:
            #accuracy,metrics = global_model.test_inference_quick(self.validation_dataset, self.test_batch_size)
            accuracy,metrics = getattr(global_model, self.test_bias_quick)(self.validation_dataset, self.test_batch_size)
            total_constraints_num = len(metrics[self.metric_name])
            unrespected_constraint = total_constraints_num
            spd_not_respect=[]
            for i in metrics[self.metric_name]:
                if abs(i) < self.threshold:
                    unrespected_constraint-=1
                else:
                    spd_not_respect.append(abs(i))    
            if unrespected_constraint != 0:
                return -accuracy + sum(spd_not_respect)
            return -accuracy
        except Exception as e:
            return np.inf
  
    def compute_bias_computed_global_model_single_constraint(self,vars):
        nc = vars
        w_avg = copy.deepcopy(self.Theta)
        n = sum(nc)
        
        for i in range(len(self.Theta)):
            for j in range(len(self.Theta[i])):
                w_avg[i][j] = w_avg[i][j] * nc[i]
                w_avg[i][j] = w_avg[i][j] / n
        w_avg = np.sum(w_avg, axis=0)
        global_model= copy.deepcopy(self.global_model_variable)
        global_model.set_weights(w_avg)
        try:
            #accuracy,metrics = global_model.test_inference_quick(self.validation_dataset, self.test_batch_size)
            accuracy, metrics = getattr(global_model, self.test_bias_quick)(self.validation_dataset,
                                                                            self.test_batch_size)
            if abs(metrics[self.metric_name]) > self.threshold:
#                if self.calls == 0:
#                    self.calls = 1
                return -accuracy + abs(metrics[self.metric_name])
            else :
#                if self.calls == 0:
#                    self.old_is_fair = True
#                    self.calls = 1
                return -accuracy

        except Exception as e:
            print("The error raised is: ", e)
            return np.inf

    def compute_bias_final_single(self,vars):
        nc = vars
        w_avg = copy.deepcopy(self.Theta)
        n = sum(nc)     
        for i in range(len(self.Theta)):
            for j in range(len(self.Theta[i])):
                w_avg[i][j] = w_avg[i][j] * nc[i]
                w_avg[i][j] = w_avg[i][j] / n
        w_avg = np.sum(w_avg, axis=0)
        global_model= copy.deepcopy(self.global_model_variable)
        global_model.set_weights(w_avg)
        try:
            #_,metrics = global_model.test_inference_quick(self.validation_dataset, self.test_batch_size)
            _, metrics = getattr(global_model, self.test_bias_quick)(self.validation_dataset,
                                                                            self.test_batch_size)
            return [abs(metrics[self.metric_name])]
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf
    def compute_bias_final(self,vars):
        nc = vars
        w_avg = copy.deepcopy(self.Theta)
        n = sum(nc)
        for i in range(len(self.Theta)):
            for j in range(len(self.Theta[i])):
                w_avg[i][j] = w_avg[i][j] * nc[i]
                w_avg[i][j] = w_avg[i][j] / n
        w_avg = np.sum(w_avg, axis=0)
        global_model= copy.deepcopy(self.global_model_variable)
        global_model.set_weights(w_avg)
        try:
            #_,metrics = global_model.test_inference_quick(self.validation_dataset, self.test_batch_size)
            _, metrics = getattr(global_model, self.test_bias_quick)(self.validation_dataset,
                                                                            self.test_batch_size)
            return [abs(metrics[self.metric_name][i]) for i in range(len(metrics[self.metric_name]))]
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf

    def get_alpha(self, ):
        r_min, r_max = -1, 1
        bounds = [[r_min, r_max] for i in range(len(self.fedavg_w))]
        if self.constraints=="multiple":
            result = differential_evolution(self.compute_bias_computed_global_model,tol=self.tol,popsize=self.popsize,maxiter=self.maxiter,strategy='best1bin', bounds=bounds,workers=self.workers,disp=True)
            print('Status : %s' % result['message'])
            print('Total Evaluations: %d' % result['nfev'])
            # evaluate solution
            solution = list(result['x'])
            spds=self.compute_bias_final(solution)
            print(spds)
            for i in spds:
                if abs(i) > self.threshold:
                    return 0
        else :
            print("getting alpha")
            result = differential_evolution(self.compute_bias_computed_global_model_single_constraint,tol=self.tol,popsize=self.popsize,maxiter=self.maxiter, mutation = self.mutation, recombination=self.recombination, strategy='best1bin', bounds=bounds,workers=self.workers,seed=1,disp=True)
            print('Status : %s' % result['message'])
            print('Total Evaluations: %d' % result['nfev'])
            # evaluate solution
            solution = list(result['x'])
            if ((self.compute_bias_final_single(solution)[0] > self.threshold)): # & (self.old_is_fair == True)):
                return 0
        return solution


