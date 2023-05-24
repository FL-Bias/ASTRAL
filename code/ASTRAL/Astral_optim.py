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

class HCO_LP(object): # hard-constrained optimization

    def __init__(self, n, m, eps, lr = 1):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        # self.objs = objs # the two objs [l, g].
        self.n = n # the dimension of \theta
        self.m = m
        self.lr = lr
        self.eps = [eps] # the error bar of the optimization process [eps1 < g, eps2 < delta1, eps3 < delta2]
        self.Ca1 = cp.Parameter((2,1))       # [d_l, d_g] * d_l or [d_l, d_g] * d_g.
        self.Ca2 = cp.Parameter((2,1))
        self.delta = cp.Parameter((2, m))
        self.alpha = cp.Variable((1,2))     # Variable to optimize
        # disparities has been satisfies, in this case we only maximize the performance
        obj_dom = cp.Maximize(self.alpha @ self.Ca1)
        obj_fair = cp.Maximize(self.alpha @ self.Ca2)
        self.alphas = torch.tensor((1,m))
        constraints_dom = [self.alpha >= 0, cp.sum(self.alpha) == 1, cp.sum(self.alpha @ self.delta) == 0]
        constraints_fair = [self.alpha >= 0, cp.sum(self.alpha) == 1, self.alpha @ self.Ca1 >= 0, cp.sum(self.alpha @ self.delta) == 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_dom)  # LP balance
        self.prob_fair = cp.Problem(obj_fair, constraints_fair)
        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.disparity = 0     # Stores the latest maximum of selected K disparities

    def get_alpha(self, dis_max, d_l1, d_l2, alphas, delta, lr = 0.01):
        d_ls = torch.cat((d_l1, d_l2))
        self.delta.value = d_ls.cpu().numpy()
        self.alphas = alphas
        print(dis_max[1])
        if dis_max[1]<= self.eps[0]: # [l, g] disparities < eps0
            print('dom')
            self.Ca1.value = (d_ls @ d_l1.t()).cpu().numpy()
            self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"
            return (self.alpha @ delta).value
        else:
            print('fair')
            self.Ca1.value = (d_ls @ d_l1.t()).cpu().numpy()
            self.Ca2.value = (d_ls @ d_l2.t()).cpu().numpy()
            self.gamma = self.prob_fair.solve(solver=cp.GLPK, verbose=False)
            print(self.alpha.value)
            print(cp.sum(self.alpha).value)
            print(cp.sum(self.alpha @ delta).value)
            print(cp.sum(self.alpha @ d_ls).value)
            print('end fair')
            self.last_move = "fair"
            return (self.alpha @ d_ls).value

class HCO_LP2(object): # hard-constrained optimization

    def __init__(self, n, m, eps, lr = 1):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.n = n # the dimension of \theta
        self.m = m
        self.lr = lr
        self.eps = [eps] # the error bar of the optimization process [eps1 < g, eps2 < delta1, eps3 < delta2]
        self.Ca1 = cp.Parameter((2,1))
        self.Ca2 = cp.Parameter((2,1))
        self.delta = cp.Parameter((2, m))
        self.alpha = cp.Variable((1,2))     # Variable to optimize
        # disparities has been satisfies, in this case we only maximize the performance
        obj_dom = cp.Minimize(self.alpha @ self.Ca1)
        obj_fair = cp.Minimize(self.alpha @ self.Ca2)
        constraints_fair = [self.alpha >= 0, cp.sum(self.alpha) == 1, cp.sum(self.alpha @ self.delta) == 0]
        constraints_dom = [self.alpha >= 0, cp.sum(self.alpha) == 1, self.alpha @ self.Ca1 >= 0, cp.sum(self.alpha @ self.delta) == 0]
        self.prob_fair = cp.Problem(obj_fair, constraints_fair)
        self.prob_dom = cp.Problem(obj_dom, constraints_dom)  # LP balance
        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.disparity = 0     # Stores the latest maximum of selected K disparities

    def get_alpha(self, dis_max, d_l1, d_l2, lr = 0.01):
        d_ls = torch.cat((d_l1, d_l2))
        self.delta.value = d_ls.cpu().numpy()
        print(dis_max[1])
        if dis_max[1] > self.eps[0]: # [l, g] disparities < eps0
            print('fair')
            self.Ca1.value = (d_ls @ d_l1.t()).cpu().numpy()
            self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "fair"
            return (self.alpha @ self.delta).value
        else:
            print('dom')
            self.Ca1.value = (d_ls @ d_l1.t()).cpu().numpy()
            self.Ca2.value = (d_ls @ d_l2.t()).cpu().numpy()
            self.gamma = self.prob_fair.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"
            return (self.alpha @ self.delta).value

class fair_obj_loss_const_OPT(object): # hard-constrained optimization

    def __init__(self, n_x, n, m, X, Y, Y_pred, S, Theta, old_alpha, eps, logs):
        self.n = n # the dimension of \theta
        self.m = m # the dimension of alpha
        self.n_x = n_x
        self.X = cp.Parameter((n_x, n))
        self.Y = cp.Parameter((n_x, 1))
        self.y = cp.Parameter((n_x, 1))
        self.sum_x = cp.Parameter((n,1))
        self.Theta = cp.Parameter((m,n))
        self.Theta_t = cp.Parameter((n,m))
        self.alpha = cp.Variable((1,m))     # Variable to optimize
        self.old_alpha = cp.Parameter((1, m))
        x = X.cpu().numpy()
        x_np = []
        s_bar = S.cpu().numpy().mean()
        s_np = S.cpu().numpy() - s_bar
        for i in range(len(x)):
            x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
            x_np.append(x_tmp * s_np[i])
        x_np = np.array(x_np)
        X_ = []
        for i in range(len(x)):
            x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
            X_.append(x_tmp)
        self.X.value = np.array(X_)
        self.sum_x.value = x_np.sum(axis=0).transpose().reshape(self.n, 1)
        self.Theta.value = Theta.cpu().numpy()
        self.Theta_t.value = Theta.cpu().numpy().transpose()
        self.eps = eps
        self.Y.value = Y.cpu().numpy().reshape(self.n_x, 1)
        self.y.value = Y_pred.cpu().detach().numpy().reshape(self.n_x, 1)
        self.old_alpha.value = np.array(old_alpha).reshape(1, self.m)
        self.loss = -cp.sum(cp.multiply(self.Y, self.X @ self.Theta_t @ self.old_alpha.T) - cp.logistic(self.X @ self.Theta_t @ self.old_alpha.T)).value
        self.eps = eps
        self.logs = logs
        # disparities has been satisfies, in this case we only maximize the performance
        self.log_likelihood = -cp.sum(cp.multiply(self.Y, self.X @ self.Theta_t @ self.alpha.T) - cp.logistic(self.X @ self.Theta_t @ self.alpha.T))
        if self.eps >= 0:
            obj_fair = cp.Minimize(cp.abs(self.alpha @ self.Theta @ self.sum_x))
            constraints_fair = [cp.sum(self.alpha) == 1,
                              self.log_likelihood <= (1+self.eps) * self.loss]
        else:
            obj_fair = cp.Minimize(cp.abs(self.alpha @ self.Theta @ self.sum_x))
            constraints_fair = [cp.sum(self.alpha) == 1]
        self.prob_fair = cp.Problem(obj_fair, constraints_fair)
        print('loss')
        print(self.loss)
        print('loss2')
        print((1+self.eps)*self.loss)

    def get_alpha(self, ):
        self.gamma = self.prob_fair.solve(verbose = self.logs)
        print("status:", self.prob_fair.status)
        print("optimal value", self.prob_fair.value)
        print('end')
        return (self.alpha).value.reshape(-1)

class loss_obj_fair_const_OPT(object): # hard-constrained optimization

    def __init2__(self, n_x, n, m, X, Y, Y_pred, S, Theta, old_alpha, eps, logs):
        self.n = n # the dimension of \theta
        self.m = m # the dimension of alpha
        self.n_x = n_x
        self.X = cp.Parameter((n_x, n))
        self.Y = cp.Parameter((n_x, 1))
        self.y = cp.Parameter((n_x, 1))
        self.sum_x = cp.Parameter((n,1))
        self.Theta = cp.Parameter((m,n))
        self.Theta_t = cp.Parameter((n,m))
        self.alpha = cp.Variable((1,m))     # Variable to optimize
        self.old_alpha = cp.Parameter((1, m))
        x = X.cpu().numpy()
        x_np = []
        s_bar = S.cpu().numpy().mean()
        s_np = S.cpu().numpy() - s_bar
        for i in range(len(x)):
            x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
            x_np.append(x_tmp * s_np[i])
        x_np = np.array(x_np)
        X_ = []
        for i in range(len(x)):
            x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
            X_.append(x_tmp)
        self.X.value = np.array(X_)
        self.sum_x.value = x_np.sum(axis=0).transpose().reshape(self.n, 1)
        self.Theta.value = Theta.cpu().numpy()
        self.Theta_t.value = Theta.cpu().numpy().transpose()
        self.eps = eps
        self.Y.value = Y.cpu().numpy().reshape(self.n_x, 1)
        self.y.value = Y_pred.cpu().detach().numpy().reshape(self.n_x, 1)
        self.old_alpha.value = np.array(old_alpha).reshape(1, self.m)
        self.loss = -cp.sum(cp.multiply(self.Y, self.X @ self.Theta_t @ self.old_alpha.T) - cp.logistic(self.X @ self.Theta_t @ self.old_alpha.T)).value
        self.eps = eps
        self.logs = logs
        # disparities has been satisfies, in this case we only maximize the performance
        self.log_likelihood = -cp.sum(cp.multiply(self.Y, self.X @ self.Theta_t @ self.alpha.T) - cp.logistic(self.X @ self.Theta_t @ self.alpha.T))
        obj_fair = cp.Minimize(self.log_likelihood)
        constraints_fair = [cp.sum(self.alpha) == 1, 1/self.n_x *  cp.abs(self.alpha @ self.Theta @ self.sum_x) <= self.eps]
        self.prob_fair = cp.Problem(obj_fair, constraints_fair)

    def __init__(self, n_x, n, m, X, Y, Y_pred, S, Theta, old_alpha, eps, logs, nb_sa):
        self.n = n # the dimension of \theta
        self.m = m # the dimension of alpha
        self.n_x = n_x
        self.X = cp.Parameter((n_x, n))
        self.Y = cp.Parameter((n_x, 1))
        self.y = cp.Parameter((n_x, 1))
        self.sum_x = []
        self.Theta = cp.Parameter((m,n))
        self.Theta_t = cp.Parameter((n,m))
        self.alpha = cp.Variable((1,m))     # Variable to optimize
        self.old_alpha = cp.Parameter((1, m))
        x = X.cpu().numpy()
        X_ = []
        for i in range(len(x)):
            x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
            X_.append(x_tmp)
        self.X.value = np.array(X_)
        self.Theta.value = Theta.cpu().numpy()
        self.Theta_t.value = Theta.cpu().numpy().transpose()
        self.eps = eps
        self.Y.value = Y.cpu().numpy().reshape(self.n_x, 1)
        self.y.value = Y_pred.cpu().detach().numpy().reshape(self.n_x, 1)
        self.old_alpha.value = np.array(old_alpha).reshape(1, self.m)
        self.loss = -cp.sum(cp.multiply(self.Y, self.X @ self.Theta_t @ self.old_alpha.T) - cp.logistic(self.X @ self.Theta_t @ self.old_alpha.T)).value
        self.eps = eps
        self.logs = logs
        # disparities has been satisfies, in this case we only maximize the performance
        self.log_likelihood = -cp.sum(cp.multiply(self.Y, self.X @ self.Theta_t @ self.alpha.T) - cp.logistic(self.X @ self.Theta_t @ self.alpha.T))
        constraints_fair = []
        if nb_sa == 1:
            x_np = []
            s_bar = S.cpu().numpy().mean()
            s_np = S.cpu().numpy() - s_bar
            for i in range(len(x)):
                x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
                x_np.append(x_tmp * s_np[i])
            x_np = np.array(x_np)
            self.sum_x.append(cp.Parameter((n,1)))
            self.sum_x[0].value = x_np.sum(axis=0).transpose().reshape(self.n, 1)
            constraints_fair.append(1/self.n_x *  cp.abs(self.alpha @ self.Theta @ self.sum_x[0]) <= self.eps)
        else:
            for k in range(nb_sa):
                element = S[:, k]
                x_np = []
                s_bar = element.cpu().numpy().mean()
                s_np = element.cpu().numpy() - s_bar
                for i in range(len(x)):
                    x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
                    x_np.append(x_tmp * s_np[i])
                x_np = np.array(x_np)
                self.sum_x.append(cp.Parameter((n,1)))
                self.sum_x[k].value = x_np.sum(axis=0).transpose().reshape(self.n, 1)
                constraints_fair.append(1/self.n_x *  cp.abs(self.alpha @ self.Theta @ self.sum_x[k]) <= self.eps[k])
        obj_fair = cp.Minimize(self.log_likelihood)
        constraints_fair.append(cp.sum(self.alpha) == 1)
        self.prob_fair = cp.Problem(obj_fair, constraints_fair)

    def get_alpha(self, ):
        self.gamma = self.prob_fair.solve(verbose=self.logs)
        print("status:", self.prob_fair.status)
        print("optimal value", self.prob_fair.value)
        return (self.alpha).value.reshape(-1)

class loss_obj_fair_const_OPT_PLUS(object): # hard-constrained optimization

    def __init__(self, n_x, n, m, X, Y, Y_pred, S, Theta, old_alpha, eps, logs):
        self.n = n # the dimension of \theta
        self.m = m # the dimension of alpha
        self.n_x = n_x
        self.X = cp.Parameter((n_x, n))
        self.Y = cp.Parameter((n_x, 1))
        self.y = cp.Parameter((n_x, 1))
        self.sum_x = cp.Parameter((n,1))
        self.Theta = cp.Parameter((m,n))
        self.Theta_t = cp.Parameter((n,m))
        self.alpha = cp.Variable((1,m))     # Variable to optimize
        self.old_alpha = cp.Parameter((1, m))
        x = X.cpu().numpy()
        x_np = []
        s_bar = S.cpu().numpy().mean()
        s_np = S.cpu().numpy() - s_bar
        for i in range(len(x)):
            x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
            x_np.append(x_tmp * s_np[i])
        x_np = np.array(x_np)
        X_ = []
        for i in range(len(x)):
            x_tmp = np.insert(x[i], len(x[i]), 1, axis=0)
            X_.append(x_tmp)
        self.X.value = np.array(X_)
        self.sum_x.value = x_np.sum(axis=0).transpose().reshape(self.n, 1)
        self.Theta.value = Theta.cpu().numpy()
        self.Theta_t.value = Theta.cpu().numpy().transpose()
        self.eps = eps
        self.Y.value = Y.cpu().numpy().reshape(self.n_x, 1)
        self.y.value = Y_pred.cpu().detach().numpy().reshape(self.n_x, 1)
        self.old_alpha.value = np.array(old_alpha).reshape(1, self.m)
        self.loss = -cp.sum(cp.multiply(self.Y, self.X @ self.Theta_t @ self.old_alpha.T) - cp.logistic(self.X @ self.Theta_t @ self.old_alpha.T)).value
        self.eps = eps
        self.logs = logs
        # disparities has been satisfies, in this case we only maximize the performance
        self.log_likelihood = -cp.sum(cp.multiply(self.Y, self.X @ self.Theta_t @ self.alpha.T) - cp.logistic(self.X @ self.Theta_t @ self.alpha.T))
        obj_fair = cp.Minimize(self.log_likelihood)
        constraints_fair = [1/self.n_x *  cp.abs(self.alpha @ self.Theta @ self.sum_x) <= self.eps]
        self.prob_fair = cp.Problem(obj_fair, constraints_fair)
        print('loss')
        print(self.loss)
        print('loss2')
        print((1+self.eps)*self.loss)

    def get_alpha(self, ):
        self.gamma = self.prob_fair.solve(verbose=self.logs)
        print("status:", self.prob_fair.status)
        print("optimal value", self.prob_fair.value)
        print('end')
        return (self.alpha).value.reshape(-1)

class wu_surrogate_risk_difference(object):
    def __init__(self, n_x, n, m, X, Y, Y_pred, S,  X_without_sa, Theta, y_var, s_var, logs, kappa_name='zero_one', delta_name='zero_one', phi_name='zero_one'):
        self.y_var = y_var
        self.s_var = s_var
        self.kappa_name = kappa_name
        self.delta_name = delta_name
        self.phi_name = phi_name
        self.logs = logs
        self.n = n # the dimension of \theta
        self.m = m # the dimension of alpha
        self.n_x = n_x # number of data instances
        self.X = cp.Parameter((n_x, n))
        self.X_without_sa = cp.Parameter((n_x, n-1))
        self.Y = cp.Parameter((n_x, 1))
        self.y = cp.Parameter((n_x, 1))
        self.Theta = cp.Parameter((m,n))
        self.Theta_t = cp.Parameter((n,m))
        self.alpha = cp.Variable((1,m))
        x = X.cpu().numpy()
        X_ = []
        for i in range(len(x)):
            x_tmp = np.insert(x[i], len(x[i]) , 1, axis=0)
            X_.append(x_tmp)
        self.X_ = np.array(X_)
        self.X.value = np.array(X_)
        x_without_sa = X_without_sa.cpu().numpy()
        X_ = []
        for i in range(len(x_without_sa)):
            x_tmp = np.insert(x_without_sa[i], len(x_without_sa[i]) - 1, 1, axis=0)
            X_.append(x_tmp)
        self.X_without_sa_ = np.array(X_)
        self.Theta.value = Theta.cpu().numpy()
        self.Theta_t.value = Theta.cpu().numpy().transpose()
        self.Y.value = Y.cpu().numpy().reshape(self.n_x, 1)
        y = Y.cpu().numpy()
        s = S.cpu().numpy()
        x_without_sa = pd.DataFrame(x_without_sa)
        s = pd.Series(s, name=s_var.name)
        y = pd.Series(y, name=y_var.name)
        self.switch()
        self.count(x_without_sa, y, s)
        self.preprocess(x_without_sa)
        # objective function
        self.log_likelihood = -cp.sum(cp.multiply(self.Y, self.X @ self.Theta_t @ self.alpha.T) - cp.logistic(self.X @ self.Theta_t @ self.alpha.T))
        objective = cp.Minimize(self.log_likelihood)
        # constraints
        self.S = cp.Parameter((n_x, 1))
        self.S.value = s.to_numpy().reshape(n_x,1)
        print(self.S.shape)
        print(self.X.shape)
        print(self.Theta_t.shape)
        print(self.alpha.T.shape)
        print((self.X @ self.Theta_t @ self.alpha.T).shape)
        self.weight = cp.Parameter((n_x, 1))
        self.weight.value = s.apply(lambda val: 1.0 / self.pos_num if val == self.s_var.pos else 1.0 / self.neg_num).to_numpy().reshape(n_x,1)
        self.sz = cp.Parameter()
        sz = self.X @ self.Theta_t @ self.alpha.T
        print(self.weight.shape)
        rd_min, rd_max, rd_k_min, rd_d_max = self.get_rd_max_min(s_var, y_var, x_without_sa, s, y)
        self.tau1 = 0.05 - rd_min + rd_k_min
        self.tau2 = 0.05 + rd_max - rd_d_max
        print(self.tau1)
        print(self.tau2)
        print(self.X.value)
        print(self.Theta_t.value)
        print(self.alpha.T.value)
        constraints = [1 / n_x * ((cp.sum(cp.multiply(self.weight, sz))) - 1) <= self.tau1, cp.sum(self.alpha) == 1]
        self.prob = cp.Problem(objective, constraints)

    def switch(self):
        if self.phi_name == 'logistic':
            self.cvx_phi = lambda z: cp.logistic(-z)  
        elif self.phi_name == 'hinge':
            self.cvx_phi = lambda z: cp.pos(1 - z)
        elif self.phi_name == 'squared':
            self.cvx_phi = lambda z: cp.square(-z)
        elif self.phi_name == 'exponential':
            self.cvx_phi = lambda z: cp.exp(-z)
        else:
            print('%s is not include' % self.phi_name)
            print('Logistic is the default setting')
            self.cvx_phi = lambda z: cp.logistic(-z)  
        if self.kappa_name == 'logistic':
            self.cvx_kappa = lambda z: cp.logistic(z) 
            self.psi_kappa = lambda mu: ((1 + mu) * math.log(1 + mu) + (1 - mu) * math.log(3 - mu)) / 2
        elif self.kappa_name == 'hinge':
            self.cvx_kappa = lambda z: cp.pos(1 + z)
            self.psi_kappa = lambda mu: mu
        elif self.kappa_name == 'squared':
            self.cvx_kappa = lambda z: cp.square(1 + z)
            self.psi_kappa = lambda mu: mu ** 2
        elif self.kappa_name == 'exponential':
            self.cvx_kappa = lambda z: cp.exp(z)
            self.psi_kappa = lambda mu: 1 - math.sqrt(1 - mu ** 2)
        else:
            print('%s is not include' % self.kappa_name)
            print('hinge is the default setting')
            self.cvx_kappa = lambda z: cp.pos(1 + z)
            self.psi_kappa = lambda mu: mu
        if self.delta_name == 'logistic':
            self.cvx_delta = lambda z: 1 - cp.logistic(-z)  # / math.log(2, math.e)
            self.psi_delta = lambda mu: ((1 + mu) * math.log(1 + mu) + (1 - mu) * math.log(1 - mu)) / 2
        elif self.delta_name == 'hinge':
            self.cvx_delta = lambda z: 1 - cp.pos(1 - z)
            self.psi_delta = lambda mu: mu
        elif self.delta_name == 'squared':
            self.cvx_delta = lambda z: 1 - cp.square(1 - z)
            self.psi_delta = lambda mu: mu ** 2
        elif self.delta_name == 'exponential':
            self.cvx_delta = lambda z: 1 - cp.exp(-z)
            self.psi_delta = lambda mu: 1 - math.sqrt(1 - mu ** 2)
        else:
            print('%s is not include' % self.delta_name)
            print('hinge is the default setting')
            self.cvx_delta = lambda z: cp.pos(1 + z)
            self.psi_delta = lambda mu: mu

    def count(self, x, y, s):
        df = pd.concat([x, y, s], axis=1)
        total, pos_num, neg_num = self.s_var.count(df)
        self.p = pos_num / total
        self.pos_num = pos_num
        self.neg_num = neg_num

    def preprocess(self, x):
        self.x_without_sa_ = np.c_[x, np.ones([x.shape[0], 1])]

    def optimize(self):
        self.prob.solve(solver=cp.ECOS, feastol=1e-8, abstol=1e-8, reltol=1e-8, max_iters=200, verbose=False, warm_start=False)
        print('status %s ' % self.prob.status)
        if self.prob.status == cp.OPTIMAL:
            self.coef_ = self.w.value.A1[:-1]
            self.intercept_ = self.w.value.A1[-1]
        else:
            print('Infeasible')
            raise TypeError

    def get_rd_max_min(self, s_var, y_var, x, s, y):
        optimal_clf = BayesUnfairClassifier(y_var, s_var, 'zero_one', 'zero_one')
        optimal_clf.fit(x, y, s)
        rd_min = optimal_clf.rd_min_kappa
        rd_max = optimal_clf.rd_max_delta
        rd_min_kappa = 0.0
        rd_max_delta = 0.0
        x_name = sorted(x.columns)
        df = pd.concat([x, y, s], axis=1)
        total, pos_num, neg_num = s_var.count(df)
        p = pos_num / total
        for key, group in df.groupby(x_name):
            total, pos_num, neg_num = s_var.count(group)
            eta_s = pos_num / total
            crd, _ = compute_minimal_crd(eta_s, self.p, self.kappa_name)
            rd_min_kappa += crd * total / df.__len__()
            crd, _ = compute_maximal_crd(eta_s, self.p, self.delta_name)
            rd_max_delta += crd * total / df.__len__()
        return rd_min, rd_max, rd_min_kappa, rd_max_delta

    def get_alpha(self, ):
        self.gamma = self.prob.solve(verbose = self.logs)
        print("status:", self.prob.status)
        print("optimal value", self.prob.value)
        print('end')
        return (self.alpha).value.reshape(-1)

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

class DE_alpha_Rwgt_search(object):
    """Using Differential Evolution to find FedAvg+Rwgt parameters at each FL rounds"""
    def __init__(self,
                Theta,
                global_model_variable,
                validation_dataset,
                fedavg_w,
                threshold,
                clientvalid_metric,
                client_idx,
                test_batch_size=64):
        self.Theta = Theta
        self.global_model_variable = copy.deepcopy(global_model_variable)
        self.validation_dataset = validation_dataset
        self.fedavg_w = fedavg_w
        self.threshold = threshold
        self.test_batch_size = test_batch_size
        self.clientvalid_metric = clientvalid_metric
        self.client_idx = client_idx

    def compute_spd_computed_global_model(self,vars):
        alpha=vars[0]
        beta=vars[1]
        w_avg = copy.deepcopy(self.Theta)        
        nc=[self.client_idx[i] * self.prelec_function_SPD(self.clientvalid_metric[i]['Statistical Parity Difference'], alpha, beta) for i in range(len(self.fedavg_w))]
        n = sum(nc)
        for i in range(len(self.Theta)):
            for j in range(len(self.Theta[i])):
                w_avg[i][j] = w_avg[i][j] * nc[i]
                w_avg[i][j] = w_avg[i][j] / n
        w_avg = np.sum(w_avg, axis=0)
        self.global_model_variable.set_weights(w_avg)
        try:
            accuracy, loss, metrics = self.global_model_variable.test_inference(
                self.validation_dataset, self.test_batch_size)
            if abs(metrics["Statistical Parity Difference"]) > self.threshold:
                return -accuracy + abs(metrics["Statistical Parity Difference"])
            else :
                return -accuracy
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf
   
    def compute_rwgt_weights(self,vars):
        alpha=vars[0]
        beta=vars[1]     
        return [self.client_idx[i] * self.prelec_function_SPD(self.clientvalid_metric[i]['Statistical Parity Difference'], alpha, beta) for i in range(len(self.fedavg_w))]

    def compute_spd_final(self,vars):
        nc = vars
        w_avg = copy.deepcopy(self.Theta)
        n = sum(nc)
        for i in range(len(self.Theta)):
            for j in range(len(self.Theta[i])):
                w_avg[i][j] = w_avg[i][j] * nc[i]
                w_avg[i][j] = w_avg[i][j] / n
        w_avg = np.sum(w_avg, axis=0)
        self.global_model_variable.set_weights(w_avg)
        try:
            _, _, metrics = self.global_model_variable.test_inference(
                self.validation_dataset, self.test_batch_size)
            print(abs(metrics["Statistical Parity Difference"]))
            return abs(metrics["Statistical Parity Difference"])
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf

    def prelec_function_SPD(self, p, alpha, beta):
        if p <= 0:
            return np.exp(-beta * (pow(-np.log(p+1), alpha)))
        else :
            return np.exp(-beta * (pow(-np.log(-p+1), alpha)))
    
    def get_alpha(self, ):
        bounds = [[0, 100],[0, 100]]
        result = differential_evolution(self.compute_spd_computed_global_model,tol=0.001,popsize=100,strategy='best1bin', bounds=bounds,workers=8,polish=False)
        # summarize the result
        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])
        # evaluate solution
        print(list(result['x']))
        solution = self.compute_rwgt_weights(list(result['x']))
        if self.compute_spd_final(solution)[0] > self.threshold:
             return 0
        return solution

class Bayesian_alpha_search(object):
    """Previously explored optimization method using Bayesian optimization"""
    def __init__(self,
                 Theta,
                 global_model,
                 validation_dataset,
                 fedavg_w,
                 threshold,
                 test_batch_size=128):
        self.Theta = Theta
        self.global_model_variable_obj = copy.deepcopy(global_model)
        self.global_model_variable_cons = copy.deepcopy(global_model)
        self.validation_dataset = validation_dataset
        self.fedavg_w = fedavg_w
        self.threshold = threshold
        self.test_batch_size = test_batch_size
        self.currmetrics= []

    def compute_obj_computed_global_model(self,**kwargs):
        nc = list(kwargs.values())
        w_avg = copy.deepcopy(self.Theta)
        n = sum(nc)
        for i in range(len(self.Theta)):
            for j in range(len(self.Theta[i])):
                w_avg[i][j] = w_avg[i][j] * nc[i]
                w_avg[i][j] = w_avg[i][j] / n
        w_avg = np.sum(w_avg, axis=0)
        self.global_model_variable_obj.set_weights(w_avg)
        try:
            accuracy, loss, metrics = self.global_model_variable_obj.test_inference(
                self.validation_dataset, self.test_batch_size)
            self.currmetrics= metrics
            return accuracy
        except:
            return np.nan

    def compute_constraint_computed_global_model(self,**kwargs):
        try:
            return abs(self.currmetrics["Statistical Parity Difference"])
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf

    def get_alpha(self, ):
        alpha = []
        pbounds = {'c'+str(i): (-1, 1) for i in range(len(self.fedavg_w))}
        initial_param={'c'+str(i):self.fedavg_w[i] for i in range(len(self.fedavg_w))}
        constraint = NonlinearConstraint(self.compute_constraint_computed_global_model, -np.inf, self.threshold)
        optimizer = BayesianOptimization(
            f=self.compute_obj_computed_global_model,
            constraint=constraint,
            pbounds=pbounds,
            verbose=
            2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=25,
        )
        optimizer.probe(params=initial_param, lazy=True)
        optimizer.maximize(init_points=40, n_iter=40, )
        print(optimizer.max)
        if optimizer.max['params'] is None:
            return self.fedavg_w
        else:
            alpha = [optimizer.max['params']['c'+str(i)] for i in range(len(self.fedavg_w))]
            return alpha

## EOD metric 

class DE_alpha_search_EOD(object):
    """ASTRAL-Hrt optimization method implementation using Differential Evolution"""
    def __init__(self,
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

    def compute_eod_computed_global_model(self,vars):
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

            accuracy,metrics = global_model.test_inference_quick_EOD(
                self.validation_dataset, self.test_batch_size)
            total_constraints_num = len(metrics["Equal Opportunity Difference"])
            unrespected_constraint = total_constraints_num
            eod_not_respect=[]
            for i in metrics["Equal Opportunity Difference"]:
                if abs(i) < self.threshold:
                    unrespected_constraint-=1
                else:
                    eod_not_respect.append(abs(i))    
            if unrespected_constraint != 0:
                return -accuracy + sum(eod_not_respect)
            return -accuracy
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf
  
    def compute_eod_computed_global_model_single_constraint(self,vars):
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
            accuracy,metrics = global_model.test_inference_quick_EOD(
                self.validation_dataset, self.test_batch_size)
            if abs(metrics["Equal Opportunity Difference"]) > self.threshold:
#                if self.calls == 0:
#                    self.calls = 1
                return -accuracy + abs(metrics["Equal Opportunity Difference"])
            else :
#                if self.calls == 0:
#                    self.old_is_fair = True
#                    self.calls = 1
                return -accuracy

        except Exception as e:
            print("The error raised is: ", e)
            return np.inf

    def compute_eod_final_single(self,vars):
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
            _,metrics = global_model.test_inference_quick_EOD(
                self.validation_dataset, self.test_batch_size)
            return [abs(metrics["Equal Opportunity Difference"])]
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf
    def compute_eod_final(self,vars):
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
            _,metrics = global_model.test_inference_quick_EOD(
                self.validation_dataset, self.test_batch_size)
            return [abs(metrics["Equal Opportunity Difference"][i]) for i in range(len(metrics["Equal Opportunity Difference"]))]
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf

    def get_alpha_eod(self, ):
        r_min, r_max = -1, 1
        bounds = [[r_min, r_max] for i in range(len(self.fedavg_w))]
        if self.constraints=="multiple":
            result = differential_evolution(self.compute_eod_computed_global_model,tol=self.tol,popsize=self.popsize,maxiter=self.maxiter,strategy='best1bin', bounds=bounds,workers=self.workers,disp=True)
            print('Status : %s' % result['message'])
            print('Total Evaluations: %d' % result['nfev'])
            # evaluate solution
            solution = list(result['x'])
            eods=self.compute_eod_final(solution)
            print(eods)
            for i in eods:
                if abs(i) > self.threshold:
                    return 0
        else :
            print("getting alpha")
            result = differential_evolution(self.compute_eod_computed_global_model_single_constraint,tol=self.tol,popsize=self.popsize,maxiter=self.maxiter, mutation = self.mutation, recombination=self.recombination, strategy='best1bin', bounds=bounds,workers=self.workers,seed=1,disp=True)
            print('Status : %s' % result['message'])
            print('Total Evaluations: %d' % result['nfev'])
            # evaluate solution
            solution = list(result['x'])
            if ((self.compute_eod_final_single(solution)[0] > self.threshold)): # & (self.old_is_fair == True)):
                return 0
        return solution


class CMAES_alpha_search(object):
    """ASTRAL-Hrt optimization method implementation using CMA-ES"""
    def __init__(self,
                metric_name,
                Theta,
                global_model_variable,
                validation_dataset,
                fedavg_w,
                threshold,
                workers=-1,
                maxiter=1000,
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

    def compute_spd_computed_global_model(self,vars):
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

            accuracy, metrics = getattr(global_model, self.test_bias_quick)(self.validation_dataset,
                                                                            self.test_batch_size)
            total_constraints_num = len(metrics[self.metric_name])
            unrespected_constraint = total_constraints_num
            spd_not_respect=[]
            for i in metrics[self.metric_name]:
                if abs(i) < self.threshold:
                    unrespected_constraint-=1
                else:
                    spd_not_respect.append(abs(i))    
            if unrespected_constraint != 0:
                return -accuracy + 100*sum([(s- self.threshold)**2 for s in spd_not_respect])
            return -accuracy
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf
  
    def compute_spd_computed_global_model_single_constraint(self,vars):
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
        except:
            return np.inf

    def compute_spd_final_single(self,vars):
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
            _, metrics = getattr(global_model, self.test_bias_quick)(self.validation_dataset,
                                                                            self.test_batch_size)
            return [abs(metrics[self.metric_name])]
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf
    def compute_spd_final(self,vars):
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
            _, metrics = getattr(global_model, self.test_bias_quick)(self.validation_dataset,
                                                                            self.test_batch_size)
            return [abs(metrics[self.metric_name][i]) for i in range(len(metrics[self.metric_name]))]
        except Exception as e:
            print("The error raised is: ", e)
            return np.inf

    def get_alpha(self, ):
        r_min, r_max = -1, 1
        bounds = [[r_min]*100, [r_max]*100]
        if self.constraints=="multiple":
            with cma.fitness_transformations.EvalParallel2(self.compute_spd_computed_global_model, 8) as eval_para:
                xopt, es = cma.fmin2(self.compute_spd_computed_global_model, [x / sum(self.fedavg_w) for x in self.fedavg_w] , 0.01,parallel_objective=eval_para)            # evaluate solution
            solution = list(xopt)
            # # evaluate solution
            # solution = list(xopt)
            # spds=self.compute_spd_final(solution)
            # print(spds)
            # for i in spds:
            #     if abs(i) > self.threshold:
            #         return 0
        else :
            print("getting alpha")
            # es = cma.CMAEvolutionStrategy([x / sum(self.fedavg_w) for x in self.fedavg_w] , 1)
            with cma.fitness_transformations.EvalParallel2(self.compute_spd_computed_global_model_single_constraint, 8) as eval_para:
                xopt, es = cma.fmin2(self.compute_spd_computed_global_model_single_constraint, [x / sum(self.fedavg_w) for x in self.fedavg_w] , 1,restarts=1, restart_from_best='True', options={'AdaptSigma': cma.sigma_adaptation.CMAAdaptSigmaCSA,'maxfevals': 1e3},bipop=True,parallel_objective=eval_para)            # evaluate solution
            solution = list(xopt)
            # if ((self.compute_spd_final_single(solution)[0] > self.threshold)): # & (self.old_is_fair == True)):
            #     return 0
        return solution
    
# class BB_alpha_search(object):
#     """ASTRAL-Hrt optimization method implementation using RBF BB algorithm"""
#     def __init__(self,
#                 Theta,
#                 global_model_variable,
#                 validation_dataset,
#                 fedavg_w,
#                 threshold,
#                 mutation = (0.5,1),
#                 recombination = 0.7,
#                 workers=-1,
#                 maxiter=1000,
#                 popsize=15,
#                 tol=0.01,
#                 test_batch_size=64):
                        
#         self.Theta = Theta
#         self.global_model_variable = copy.deepcopy(global_model_variable)
#         self.validation_dataset = validation_dataset
#         self.fedavg_w = fedavg_w
#         self.threshold = threshold
#         self.old_is_fair = False
#         self.test_batch_size = test_batch_size
#         self.workers=workers
#         self.maxiter=maxiter
#         self.popsize=popsize
#         self.tol=tol
#         self.mutation = mutation
#         self.recombination = recombination
#         self.calls = 0
#         if isinstance(self.validation_dataset.sensitive_att,list):
#             self.constraints="multiple"
#         else:
#             self.constraints="single"

#     def compute_spd_computed_global_model(self,vars):
#         nc = vars
#         w_avg = copy.deepcopy(self.Theta)
#         n = sum(nc)
#         for i in range(len(self.Theta)):
#             for j in range(len(self.Theta[i])):
#                 w_avg[i][j] = w_avg[i][j] * nc[i]
#                 w_avg[i][j] = w_avg[i][j] / n
#         w_avg = np.sum(w_avg, axis=0)
#         global_model= copy.deepcopy(self.global_model_variable)
#         global_model.set_weights(w_avg)
#         try:

#             accuracy,metrics = global_model.test_inference_quick(
#                 self.validation_dataset, self.test_batch_size)
#             total_constraints_num = len(metrics["Statistical Parity Difference"])
#             unrespected_constraint = total_constraints_num
#             spd_not_respect=[]
#             for i in metrics["Statistical Parity Difference"]:
#                 if abs(i) < self.threshold:
#                     unrespected_constraint-=1
#                 else:
#                     spd_not_respect.append(abs(i))    
#             if unrespected_constraint != 0:
#                 return -accuracy + 100*sum([(s- self.threshold)**2 for s in spd_not_respect])
#             return -accuracy
#         except Exception as e:
#             print("The error raised is: ", e)
#             return np.inf
  
#     def compute_spd_computed_global_model_single_constraint(self,vars):
#         nc = vars
#         w_avg = copy.deepcopy(self.Theta)
#         n = sum(nc)
        
#         for i in range(len(self.Theta)):
#             for j in range(len(self.Theta[i])):
#                 w_avg[i][j] = w_avg[i][j] * nc[i]
#                 w_avg[i][j] = w_avg[i][j] / n
#         w_avg = np.sum(w_avg, axis=0)
#         global_model= copy.deepcopy(self.global_model_variable)
#         global_model.set_weights(w_avg)
#         try:
#             accuracy,metrics = global_model.test_inference_quick(
#                 self.validation_dataset, self.test_batch_size)
#             if abs(metrics["Statistical Parity Difference"]) > self.threshold:
# #                if self.calls == 0:
# #                    self.calls = 1
#                 return -accuracy + (100*abs(metrics["Statistical Parity Difference"]) - 100*self.threshold)**2
#             else :
# #                if self.calls == 0:
# #                    self.old_is_fair = True
# #                    self.calls = 1
#                 return -accuracy
#         except:
#             return np.inf

#     def compute_spd_final_single(self,vars):
#         nc = vars
#         w_avg = copy.deepcopy(self.Theta)
#         n = sum(nc)     
#         for i in range(len(self.Theta)):
#             for j in range(len(self.Theta[i])):
#                 w_avg[i][j] = w_avg[i][j] * nc[i]
#                 w_avg[i][j] = w_avg[i][j] / n
#         w_avg = np.sum(w_avg, axis=0)
#         global_model= copy.deepcopy(self.global_model_variable)
#         global_model.set_weights(w_avg)
#         try:
#             _,metrics = global_model.test_inference_quick(
#                 self.validation_dataset, self.test_batch_size)
#             return [abs(metrics["Statistical Parity Difference"])]
#         except Exception as e:
#             print("The error raised is: ", e)
#             return np.inf
#     def compute_spd_final(self,vars):
#         nc = vars
#         w_avg = copy.deepcopy(self.Theta)
#         n = sum(nc)
#         for i in range(len(self.Theta)):
#             for j in range(len(self.Theta[i])):
#                 w_avg[i][j] = w_avg[i][j] * nc[i]
#                 w_avg[i][j] = w_avg[i][j] / n
#         w_avg = np.sum(w_avg, axis=0)
#         global_model= copy.deepcopy(self.global_model_variable)
#         global_model.set_weights(w_avg)
#         try:
#             _,metrics = global_model.test_inference_quick(
#                 self.validation_dataset, self.test_batch_size)
#             return [abs(metrics["Statistical Parity Difference"][i]) for i in range(len(metrics["Statistical Parity Difference"]))]
#         except Exception as e:
#             print("The error raised is: ", e)
#             return np.inf

#     def get_alpha(self, ):
#         r_min, r_max = -1, 1
#         bounds = [[r_min, r_max] for i in range(len(self.fedavg_w))]
#         if self.constraints=="multiple":
#             with cma.fitness_transformations.EvalParallel2(self.compute_spd_computed_global_model, 8) as eval_para:
#                 xopt, es = cma.fmin2(self.compute_spd_computed_global_model, [x / sum(self.fedavg_w) for x in self.fedavg_w] , 0.01,parallel_objective=eval_para)            # evaluate solution
#             solution = list(xopt)
#             # # evaluate solution
#             # solution = list(xopt)
#             # spds=self.compute_spd_final(solution)
#             # print(spds)
#             # for i in spds:
#             #     if abs(i) > self.threshold:
#             #         return 0
#         else :
#             print("getting alpha")
#             # es = cma.CMAEvolutionStrategy([x / sum(self.fedavg_w) for x in self.fedavg_w] , 1)
#             solution = bb.search_min(f = self.compute_spd_computed_global_model_single_constraint,  # given function
#                             domain = bounds,
#                             budget = 100,  # total number of function calls available
#                             batch = 4,  # number of calls that will be evaluated in parallel
#                             resfile = 'output.csv') 
#             solution = list(solution)
#             # if ((self.compute_spd_final_single(solution)[0] > self.threshold)): # & (self.old_is_fair == True)):
#             #     return 0
#         return solution