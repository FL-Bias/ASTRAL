import copy
import random
import abc
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from opacus import PrivacyEngine
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import SGDClassifier
from bias_metrics import *
from model_metrics import *


class FLModel(abc.ABC):
    """
    Base class for all FLModels. This includes supervised and unsupervised ones.
    """
    def __init__(self, model_spec, **kwargs):
        """
        Initializes an `FLModel` object

        :param model_name: String describing the model e.g., keras_cnn
        :type model_name: `str`
        :param model_spec: Specification of the the model
        :type model_spec: `dict`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        """
        self.model_name = model_spec['model_name']
        self.model_type = None
        self.model_spec = model_spec if model_spec else {}
        self.test_metrics = []
        self.validation_metrics = []

    @abc.abstractmethod
    def set_weights(self, model_weights, **kwargs):
        """
        Updates model using provided `model_update`. Additional arguments
        specific to the model can be added through `**kwargs`

        :param model_update: Model with update. This is specific to each model \
        type e.g., `ModelUpdateSGD`. The specific type should be checked by \
        the corresponding FLModel class.
        :type model_update: `ModelUpdate`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_weights(self, **kwargs):
        """
        Updates model using provided `model_update`. Additional arguments
        specific to the model can be added through `**kwargs`

        :param model_update: Model with update. This is specific to each model \
        type e.g., `ModelUpdateSGD`. The specific type should be checked by \
        the corresponding FLModel class.
        :type model_update: `ModelUpdate`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, train_data, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: Training data.
        :type train_data: Data structure containing training data, \
        varied based on model types.
        :param kwargs: Dictionary of model-specific arguments for fitting, \
        e.g., hyperparameters for local training, information provided \
        by aggregator, etc.
        :type kwargs: `dict`
        :return: None
        """
        raise NotImplementedError


    @abc.abstractmethod
    def test_inference(self, testing_dataset, batch_size=128, **kwargs):
        """
        Evaluates model given the test dataset.
        Multiple evaluation metrics are returned in a dictionary

        :param test_dataset: Provided test dataset to evalute the model.
        :type test_dataset: `tuple` of `np.ndarray` or data generator.
        :param batch_size: batch_size: Size of batches.
        :type batch_size: `int`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: Dictionary with all evaluation metrics provided by specific \
        implementation.
        :rtype: `dict`
        """
        raise NotImplementedError

class LR_PyTorch_old(torch.nn.Module):
    def __init__(self, input_dim, output_dim, seed = 42):
        torch.manual_seed(seed)
        super(LR_PyTorch_old, self).__init__()
        if output_dim == 2:
            output_dim = 1
        self.num_classes = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        logits = self.linear(x)
        probas = torch.sigmoid(logits)
        return probas, logits

class LR_PyTorch(torch.nn.Module):

    def __init__(self, input_dim, output_dim, disparity_type="DP", dataset="adult", seed = 42):
        super(LR_PyTorch, self).__init__()
        if output_dim == 2:
            output_dim = 1
        self.num_classes = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.loss = nn.BCELoss()
        self.disparity_type = disparity_type
        self.dataset = dataset

    def forward(self, x, y, A):
        ys_pre = self.linear(x)
        ys = torch.sigmoid(ys_pre)

        hat_ys = (ys >= 0.5).float()
        task_loss = self.loss(ys, y)
        accs = torch.mean((hat_ys == y).float()).item()
        try:
            if self.disparity_type == "DP":
                pred_dis = torch.sum(torch.sigmoid(10 * ys_pre) * A) / torch.sum(A) - torch.sum(torch.sigmoid(10 * ys_pre) * (1 - A)) / torch.sum(1 - A)
                disparitys = torch.sum(hat_ys * A) / torch.sum(A) - \
                                 torch.sum(hat_ys * (1 - A)) / torch.sum(1 - A)
            elif self.disparity_type == "Eoppo":
                if "eicu_d" in self.dataset:
                    pred_dis = torch.sum(torch.sigmoid(10 * (1 - ys_pre)) * A * (1 - y)) / torch.sum(A * (1 - y)) - torch.sum(torch.sigmoid(10 * (1 - ys_pre)) * (1 - A) * (1 - y)) / torch.sum((1 - A) * (1 - y))
                    disparitys = torch.sum((1 - hat_ys) * A * (1 - y)) / torch.sum(A * (1 - y)) - \
                                     torch.sum((1 - hat_ys) * (1 - A) * (1 - y)) / \
                                     torch.sum((1 - A) * (1 - y))
                else:
                    pred_dis = torch.sum(torch.sigmoid(10 * ys_pre) * A * y) / torch.sum(A * y) - torch.sum(torch.sigmoid(10 * ys_pre) * (1 - A) * y) / torch.sum((1 - A) * y)
                    disparitys = torch.sum(hat_ys * A * y) / torch.sum(A * y) - \
                                     torch.sum(hat_ys * (1 - A) * y) / torch.sum((1 - A) * y)
            disparitys = disparitys.item()
        except:
            pred_dis, disparitys = 0, 0
        return task_loss, accs, 0, pred_dis, disparitys, ys, ys_pre

    def randomize(self):
        self.model.apply(weights_init)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.weight.data *= 0.1

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
                # y = torch.sigmoid(y_temp)
        return y








class LogisticRegressionPytorch(FLModel):
    """docstring for LogisticRegressionPytorch"""
    def __init__(self, model_spec, warmup, **kwargs):
        super(LogisticRegressionPytorch, self).__init__(model_spec)
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 'cpu'
        self.nb_features = model_spec['nb_inputs']
        self.nb_classes = model_spec['nb_outputs']
        self.model = LR_PyTorch(self.nb_features, self.nb_classes)
        self.nb_epochs = model_spec['nb_epochs']
        self.batch_size = model_spec['batch_size']
        self.learning_rate = model_spec['learning_rate']
        optimizer_class = model_spec['optimizer_class']
        optimizer_class = getattr(torch.optim, optimizer_class)
        try :
            self.weight_decay = model_spec['weight_decay']
        except :
            self.weight_decay = 0
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        try :
            self.learning_rate_type = model_spec['learning_rate_type']
            if self.learning_rate_type == 'dynamic':
                self.learning_rate_freq = model_spec['learning_rate_frequency']
                self.learning_rate_fraction = model_spec['learning_rate_fraction']
                self.learning_rate_min = model_spec['learning_rate_min']
            elif self.learning_rate_type == 'scheduler':
                scheduler_class = model_spec['scheduler_class']
                gamma = model_spec['gamma']
                scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class)
                self.scheduler = scheduler_class(self.optimizer, gamma = gamma )
        except : 
            self.learning_rate_type = 'constant'
        print(self.learning_rate_type)
        print(self.learning_rate)
        loss_class = model_spec['loss_class']
        loss_class = getattr(nn, loss_class)
        self.criterion = loss_class() #nn.BCEWithLogitsLoss()
        if warmup == 1:
            with open(model_spec["warmup_parameters_path"], "rb") as f:  # Unpickling
                weights = pickle.load(f)
                self.set_weights(weights)
    def init_warm_up(self, model_weights):
        w_new = copy.deepcopy(self.model.state_dict())
        layer = []
        for i in range(len(model_weights)-1):
            layer.append(model_weights[i])
        w_new['linear.weight'][0] = torch.FloatTensor(layer)
        w_new['linear.bias'][0] = torch.FloatTensor([model_weights[-1]])
        self.model.load_state_dict(w_new)

    def set_weights(self, model_weights):
        w_new = copy.deepcopy(self.model.state_dict())
        layer = []
        for i in range(len(model_weights)-1):
            layer.append(model_weights[i])
        w_new['linear.weight'][0] = torch.FloatTensor(layer)
        w_new['linear.bias'][0] = torch.FloatTensor([model_weights[-1]])
        self.model.load_state_dict(w_new)


    def set_weights_rand(self, rand_proportion):
        model_weights = self.get_weights()
        print("RADNOMNESS=============================================================================")
        print(model_weights)
        rand_positions = random.sample(range(0, len(model_weights)), round(len(model_weights)*rand_proportion))
        print(rand_positions)
        w_new = copy.deepcopy(self.model.state_dict())
        layer = []
        for i in range(len(model_weights)-1):
            if i in rand_positions:
                layer.append(torch.randn(1,1).item())
            else:
                layer.append(model_weights[i])
        print(layer)
        w_new['linear.weight'][0] = torch.FloatTensor(layer)
        if (len(model_weights)-1) in rand_positions:
            w_new['linear.bias'][0] = torch.randn(1,1).item()
        else:
            w_new['linear.bias'][0] = torch.FloatTensor([model_weights[-1]])
        self.model.load_state_dict(w_new)
        print(self.get_weights())

    def get_weights(self):
        model_weights = []
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                for x in layer.weight[0] :
                    model_weights.append(x.item()) 
                model_weights.append(layer.bias[0].item())
        return model_weights

    def train(self, train_dataset, seed = 42, test = False, test_dataset = None, validation = False, validation_dataset = None):
        self.model.train()
        epoch_loss = []
        # set seed
        #np.random.seed(seed)
        #random.seed(seed)
        #torch.manual_seed(seed)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle= True)
        for i in range(self.nb_epochs):
            batch_loss = []
            for (batch_idx, (features, labels, SA)) in enumerate(self.train_loader):
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                labels = labels.view(len(labels), -1)
                self.model.zero_grad()
                task_loss, accs, aucs, pred_dis, disparitys, probas, logits = self.model(features, labels, SA)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if test :
                test_acc, test_loss, test_metrics = self.test_inference(test_dataset)
                self.test_metrics.append([i, test_acc, test_loss,test_metrics])
                print(test_acc)
                self.model.train()
            if validation :
                pass
            self.update_learning_rate(i)
        return sum(epoch_loss) / len(epoch_loss)

    def update_learning_rate(self, e):
        if (self.learning_rate_type == 'dynamic') :
            if  (e != 0) & (e % self.learning_rate_freq == 0) & (self.learning_rate > self.learning_rate_min) :
                self.learning_rate = self.learning_rate / self.learning_rate_fraction
                for g in self.optimizer.param_groups:
                    g['lr'] = self.learning_rate
        elif self.learning_rate_type == 'scheduler':
            self.scheduler.step()
       
    def test_inference(self, testing_dataset, test_batch_size = 64):
        # Returns the test accuracy and loss. 
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)
        model = copy.deepcopy(self.model)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        device = "cpu"
        criterion = self.criterion.to(device)
        features_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad() :
            batch_loss = []
            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                features_fairness = np.concatenate((features_fairness, features), axis=0) if features_fairness.size else features
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels
                SA = SA.view(len(SA), -1)
                labels = labels.view(len(labels), -1)
                # Inference
                task_loss, accs, aucs, pred_dis, disparitys, probas, logits = model(features, labels, SA)
                labels = labels.view(len(labels), -1)
                batch_loss.append(criterion(logits, labels).item())
                # Prediction
                pred_labels =probas.round()
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1)), axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1)
                pred_labels = pred_labels.view(len(pred_labels), -1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)      
        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(features_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.numpy()
            pred_labels_fairness=pred_labels_fairness.numpy()
            features_fairness = features_fairness.numpy()
        additional_metrics = get_eval_metrics_for_classificaton(labels_fairness.astype(int), pred_labels_fairness.astype(int))
        fairness_rep = fairness_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, np.array(self.get_weights()), testing_dataset.columns)
        dict_metrics = {**additional_metrics, **fairness_rep}
        loss = sum(batch_loss)/len(batch_loss)
        accuracy = correct / total
        return accuracy, loss, dict_metrics

    def test_inference_quick(self, testing_dataset, test_batch_size = 64):
        # Returns the test accuracy and loss. """
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)
        model = copy.deepcopy(self.model)
        model.eval()
        total, correct = 0.0, 0.0
        device = "cpu"
        criterion = self.criterion.to(device)
        features_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad() :
            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device) )
                features_fairness = np.concatenate((features_fairness, features), axis=0) if features_fairness.size else features
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels
                SA = SA.view(len(SA), -1)
                labels = labels.view(len(labels), -1)
                # Inference
                task_loss, accs, aucs, pred_dis, disparitys, probas, logits = model(features, labels, SA)
                # Prediction
                pred_labels =probas.round()
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1)), axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1)
                pred_labels = pred_labels.view(len(pred_labels), -1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)
        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(features_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.numpy()
            pred_labels_fairness=pred_labels_fairness.numpy()
            features_fairness = features_fairness.numpy()
        fairness_rep = spd_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, np.array(self.get_weights()), testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
        print(accuracy)
        print(dict_metrics)
        return accuracy, dict_metrics


class LogisticRegressionPytorch_DP(LogisticRegressionPytorch):
    def __init__(self, model_spec, warmup, **kwargs):
        super(LogisticRegressionPytorch, self).__init__(model_spec)
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 'cpu'
        self.nb_features = model_spec['nb_inputs']
        self.nb_classes = model_spec['nb_outputs']
        self.model = LR_PyTorch(self.nb_features, self.nb_classes)
        self.nb_epochs = model_spec['nb_epochs']
        self.batch_size = model_spec['batch_size']
        self.learning_rate = model_spec['learning_rate']
        self.eps = model_spec['eps']
        self.nb_rounds = model_spec['nb_rounds']
        optimizer_class = model_spec['optimizer_class']
        optimizer_class = getattr(torch.optim, optimizer_class)
        try :
            self.weight_decay = model_spec['weight_decay']
        except :
            self.weight_decay = 0
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        try :
            self.learning_rate_type = model_spec['learning_rate_type']
            if self.learning_rate_type == 'dynamic':
                self.learning_rate_freq = model_spec['learning_rate_frequency']
                self.learning_rate_fraction = model_spec['learning_rate_fraction']
                self.learning_rate_min = model_spec['learning_rate_min']
            elif self.learning_rate_type == 'scheduler':
                scheduler_class = model_spec['scheduler_class']
                gamma = model_spec['gamma']
                scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class)
                self.scheduler = scheduler_class(self.optimizer, gamma = gamma )
        except :
            self.learning_rate_type = 'constant'
        print(self.learning_rate_type)
        print(self.learning_rate)
        loss_class = model_spec['loss_class']
        loss_class = getattr(nn, loss_class)
        self.criterion = loss_class() #nn.BCEWithLogitsLoss()
        if warmup == 1:
            with open(model_spec["warmup_parameters_path"], "rb") as f:  # Unpickling
                weights = pickle.load(f)
                self.set_weights(weights)
    def train(self,  train_dataset, seed=42, test=False, test_dataset=None, validation=False, validation_dataset=None):
        # Set mode to train model
        print(self.model.state_dict())
        model_copy = copy.deepcopy(self.model)
        self.model.train()
        epoch_loss = []
        accuracies = []
        eps = self.eps
        x = copy.deepcopy(self.model.state_dict())

        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        sigma = compute_noise(len(self.train_loader.dataset), self.batch_size, self.eps,
                              self.nb_epochs * self.nb_rounds, 0.00001, 0.1)

        #privacy_engine = PrivacyEngine(module=self.model,batch_size=self.batch_size,
        #    sample_size=len(self.train_loader.dataset),
        #    noise_multiplier=sigma,
        #    max_grad_norm=1, )
        #privacy_engine.attach(self.optimizer)
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=sigma,
            max_grad_norm=1.0,
        )

        epoch_loss = []
        # set seed
        #np.random.seed(seed)
        #random.seed(seed)
        #torch.manual_seed(seed)
        for i in range(self.nb_epochs):
            batch_loss = []
            for (batch_idx, (features, labels, SA)) in enumerate(self.train_loader):
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                labels = labels.view(len(labels), -1)
                self.model.zero_grad()
                self.optimizer.zero_grad()
                task_loss, accs, aucs, pred_dis, disparitys, probas, logits = self.model(features, labels, SA)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if test :
                test_acc, test_loss, test_metrics = self.test_inference(test_dataset)
                self.test_metrics.append([i, test_acc, test_loss,test_metrics])
                print(test_acc)
                self.model.train()
            if validation :
                pass
            self.update_learning_rate(i)
        state_dict = copy.deepcopy(self.model).state_dict()
        state_dict_cp = copy.deepcopy(self.model).state_dict()
        for key in state_dict_cp.keys():
            new_key = key.replace('_module.', '')
            state_dict[new_key] = state_dict.pop(key)
        model_copy.load_state_dict(state_dict)
        self.model = model_copy
        print('end')
        print(self.model)
        print(self.model.state_dict)
        return sum(epoch_loss) / len(epoch_loss)
