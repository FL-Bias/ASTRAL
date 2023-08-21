import copy
import random
import abc
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from torchvision.models import resnet18
import torch.nn.init as init
# from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
# from opacus import PrivacyEngine
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import SGDClassifier
import numpy as np
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
        elif warmup == 3:
            self.set_weights([0.028442950919270515, -0.02285325713455677, 0.005783435422927141, -0.03911207616329193, -0.010760334320366383, 0.10212109237909317, 0.07681117206811905, -0.005573232192546129, -0.005946800112724304, -0.06727273017168045, -0.047146983444690704, -0.026407018303871155, -0.021584730595350266, -0.04715927317738533, -0.06648177653551102, -0.04858715832233429, 0.012095877900719643, -0.0010482934303581715, 0.14306649565696716, -0.018278127536177635, 0.1478782296180725, -0.05425938963890076, -0.013149967417120934, 0.13093167543411255, 0.142441987991333, 0.005301222205162048, -0.03919677436351776, 0.0027979922015219927, 0.02499425783753395, 0.010931920260190964, -0.020898498594760895, 0.0770137831568718, 0.010398867540061474, -0.08653223514556885, -0.016044212505221367, 0.02869449369609356, 0.005721542052924633, -0.00014057441148906946, -0.0047289105132222176, -0.0013315926771610975, 0.026428014039993286, -0.024563275277614594, 0.007941553369164467, 0.047802139073610306, 0.01602407917380333, -0.00845649465918541, 0.07658935338258743, 0.0402335599064827, 0.0011115394299849868, 0.018291130661964417, -0.03849441558122635, 0.03107895888388157, -0.011234031058847904, -0.022057322785258293, 0.03203990310430527, -0.0594278909265995, -0.04642641916871071, 0.01775267906486988, 0.03518754243850708, 0.03744926676154137, -0.09832267463207245, -9.352892084280029e-05, 0.12433340400457382, -0.0010910952696576715, -0.035398487001657486, -0.03899345546960831, -0.03836790472269058, -0.044882241636514664, 0.011779827997088432, -0.018167544156312943, 0.09459953010082245, 0.004055758006870747, 0.04652877897024155, 0.0014352384023368359, 0.026195762678980827, -0.008152218535542488, 0.004729786422103643, 0.008503793738782406, -0.010423232801258564, -0.0016518783522769809, 0.005864882376044989, 0.008757718838751316, 0.004349811933934689, -0.013957378454506397, -0.024374201893806458, 0.02222970873117447, 0.010566913522779942, 0.00010991505405399948, 0.003069645958021283, 0.009146622382104397, -0.011456954292953014, -0.0027360031381249428, -0.032649409025907516, 0.08818073570728302, 0.041547857224941254, 0.07019490748643875, -0.11359041184186935, 0.024799223989248276, 0.0008395556942559779, -0.010270015336573124, -0.0012446288019418716, 0.008344865404069424, -0.005941114388406277, 0.0006211460568010807, 0.00491901021450758, -0.013142234645783901, -0.000636776618193835, -0.007271301466971636, -0.010430919006466866, 0.001894226879812777, -0.005060259718447924, 0.009087024256587029, 0.006292879581451416, 0.0018096348503604531, 0.02080979384481907, -0.012767184525728226, 0.007449811324477196, -0.002068754518404603, -0.009420187212526798, -0.0025393019896000624, -0.0013824322959408164, -0.006403452251106501, 0.00241277483291924, 0.0020068103913217783, -0.00740509619936347, 0.008188745006918907, -0.012360597029328346, -0.009001840837299824, -0.012897360138595104, -0.010319361463189125, 0.001212287344969809, 0.00257712765596807, -0.0018874014494940639, 0.007337150629609823, -0.009056157432496548, 0.017813052982091904, -0.004018764011561871, 0.0026980063412338495, -0.00553864473477006, 0.005257025361061096, -0.003501259721815586, 0.0024191313423216343, 0.0004276778781786561, 0.00394259812310338, -0.0031594594474881887, 0.00492175342515111, 0.0004666648746933788, 0.00799409206956625, -0.01033939328044653, -0.00815865769982338, -0.010116602294147015, 0.0007403204799629748, -0.0025852597318589687, 0.009336029179394245, -0.01555859949439764, -0.005326410755515099, -0.051122862845659256, -0.011015864089131355, -0.000665815023239702, -0.005189611576497555, -8.639422958367504e-06, -0.006108592264354229, -0.010220563039183617, 1.2242323464306537e-05, 0.0009837700054049492, -0.005602110642939806, -0.009987856261432171, -0.0030771461315453053, -0.013378591276705265, -0.0055290935561060905, -2.193282125517726e-05, -0.0012445757165551186, -0.00773663492873311, -0.008718813769519329, 0.0876016616821289, -0.007916350848972797, 0.02274707518517971, -0.002127370797097683, -0.007085842080414295, -0.003401507157832384, -0.004092665389180183, 0.0002729542029555887, -0.0008110803901217878, 0.0009831784991547465, -0.0006529887323267758, -0.0036482643336057663, 9.976894216379151e-05, -0.0025071643758565187, -0.012971354648470879, -0.0012823458528146148, -0.01130230538547039, -0.047270048409700394, -0.05466683208942413, -0.0025607445277273655, -0.010437194257974625, -0.0020311528351157904, 0.08995489031076431, -0.005061397794634104, -0.017615048214793205, -0.047539640218019485, 0.0033715451136231422, -0.0064609358087182045, -0.004397541284561157, 0.005232858471572399, -0.013674234971404076, -0.009095803834497929, 0.007416103966534138, 0.003942091949284077, 0.0012256186455488205, 0.0006498419679701328, 0.0011710033286362886, -0.002191022504121065, 0.003932462073862553, -0.003377725835889578, 0.007394018582999706, 0.0013343151658773422, -0.011141648516058922, 0.0007518742349930108, 0.0012867647456005216, -0.008007864467799664, -0.0017477304209023714, -0.0017195101827383041, 0.01035984419286251, 0.007626710459589958, 0.0013291402719914913, -0.011248543858528137, -0.008516835980117321, -0.0015714602777734399, 0.006325304042547941, -0.004349695052951574, 0.008496791124343872, -0.00797947496175766, 0.007973289117217064, -0.002635143930092454, -0.03088872693479061, -0.00901183020323515, 0.013584146276116371, -0.007116071879863739, 0.012750199995934963, 0.014227860607206821, -0.00830528512597084, 0.009201493114233017, -0.004631977528333664, -0.00723189627751708, -0.011955990456044674, 0.010557274334132671, -0.007358289789408445, -0.003012490924447775, 0.0020146742463111877, -0.011690955609083176, -0.01080214325338602, -0.008177928626537323, -0.0011400863295421004, -0.003158395644277334, 0.010463683865964413, -0.000723384553566575, 0.006209059618413448, 0.012036549858748913, 0.011771055869758129, -0.012900535948574543, 0.025207122787833214, -0.0028047398664057255, -0.006424468010663986, -0.005175452213734388, 0.006559455301612616, -0.0038716422859579325, -0.0053576454520225525, -0.008071723394095898, 0.0072759282775223255, 0.01237731333822012, -0.003220482962206006, -0.0050759087316691875, -0.0012724888511002064, 0.015272003598511219, -0.003906699363142252, -0.0023496176581829786, -0.0036237735766917467, -0.0022054840810596943, 0.0021405741572380066, 0.0003998340107500553, 0.008641507476568222, 0.005654545966535807, -0.007304242346435785, 0.0015265881083905697, -0.0030105209443718195, -0.006270615383982658, -0.006805194541811943, 0.011825226247310638, 0.01777639426290989, 0.010577261447906494, -0.004615108948200941, -0.0077664232812821865, -0.011639343574643135, -6.6025368141708896e-06, -0.005700360517948866, -0.0021714831236749887, 0.0038377069868147373, 0.004561762325465679, 0.001244633342139423, 0.010939603671431541, 0.0013960649957880378, -0.015131351538002491, 0.008820529095828533, -0.00011749417171813548, -0.006791714113205671, 0.0030505333561450243, -0.011582473292946815, -0.000624484964646399, -0.0064701419323682785, -0.0010514851892367005, 0.017339982092380524, 0.01585850678384304, -0.006709597073495388, 0.011047803796827793, 0.001811642199754715, -0.0006348080351017416, -0.002784519689157605, -0.00022395304404199123, -0.004403978120535612, -0.005753856152296066, -2.2113705199444667e-05, -0.009481541812419891, 0.0023453461471945047, -0.01822900027036667, -0.003683798247948289, -0.006393227260559797, -0.0003920564486179501, -0.0115281967446208, 0.0024498843122273684, -0.0030396783258765936, 0.01781030371785164, -0.01222830731421709, 0.012080498971045017, 0.004123176448047161, 0.0029137185774743557, -0.0021657689940184355, -0.0033689311239868402, -0.014353111386299133, 0.011677910573780537, -0.004802628420293331, 0.006806045770645142, -0.003385822521522641, 0.010273212566971779, -0.004833787679672241, -0.01583242602646351, -0.006084902677685022, -0.009127301163971424, -0.00417390326038003, -0.011593939736485481, -0.005679151974618435, -0.00849978718906641, 0.005302959121763706, 0.0008788060513325036, -0.0036968335043638945, 0.006688629277050495, 0.013483196496963501, 0.00012838325346820056, 0.008822362869977951, -0.0017822328954935074, -0.010403268970549107, 0.01268391590565443, -0.0087044108659029, 0.007311530411243439, -0.009464433416724205, 0.016387268900871277, 0.001537637203000486, 0.0027331311721354723, -0.0026382897049188614, 0.03163246810436249, -0.026466786861419678, -0.005690638907253742, 0.01203624252229929, 0.0055023725144565105, -0.007130986545234919, 0.004621703643351793, 0.009754272177815437, -0.009471261873841286, -0.005307766143232584, -0.016910811886191368, -0.004720612894743681, 0.01486518606543541, -0.06442456692457199, -0.006625302601605654, 0.32959309220314026, 0.12241876125335693, 0.32600632309913635, 0.09741032868623734, -0.03007362224161625, 0.015166711062192917, 0.1601521223783493, -0.001959477784112096, -0.0009124343050643802, -0.817751407623291, -0.2984744906425476, -1.169452428817749])
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
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
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
            print('epoch: {}  loss: {}'.format(i, sum(batch_loss) / len(batch_loss)))
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
        fairness_rep = spd_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
        print(accuracy)
        print(dict_metrics)
        return accuracy, dict_metrics


    def test_inference_quick_EOD(self, testing_dataset, test_batch_size = 64):
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
        fairness_rep = eod_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
        print(accuracy)
        print(dict_metrics)
        return accuracy, dict_metrics
    



    def test_inference_quick_discr_idx(self, testing_dataset, test_batch_size = 64):
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
        fairness_rep = discr_idx_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
        # print(accuracy)
        # print(dict_metrics)
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
























class Binary_SVM(nn.Module):
    def __init__(self, n_features, nb_classes):
        super(Binary_SVM, self).__init__()
        self.linear = nn.Linear(n_features, nb_classes)

    def forward(self, x):
        out = self.linear(x)
        #out = out.squeeze()
        return out


class BinarySVMPytorch(FLModel):
    """docstring for LogisticRegressionPytorch"""

    def __init__(self, model_spec, warmup, **kwargs):
        super(BinarySVMPytorch, self).__init__(model_spec)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 'cpu'
        self.nb_features = model_spec['nb_inputs']
        self.nb_classes = model_spec['nb_outputs']
        self.model = Binary_SVM(self.nb_features, self.nb_classes)
        self.nb_epochs = model_spec['nb_epochs']
        self.batch_size = model_spec['batch_size']
        self.learning_rate = model_spec['learning_rate']
        optimizer_class = model_spec['optimizer_class']
        optimizer_class = getattr(torch.optim, optimizer_class)
        try:
            self.weight_decay = model_spec['weight_decay']
        except:
            self.weight_decay = 0
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        try:
            self.learning_rate_type = model_spec['learning_rate_type']
            if self.learning_rate_type == 'dynamic':
                self.learning_rate_freq = model_spec['learning_rate_frequency']
                self.learning_rate_fraction = model_spec['learning_rate_fraction']
                self.learning_rate_min = model_spec['learning_rate_min']
            elif self.learning_rate_type == 'scheduler':
                scheduler_class = model_spec['scheduler_class']
                gamma = model_spec['gamma']
                scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class)
                self.scheduler = scheduler_class(self.optimizer, gamma=gamma)
        except:
            self.learning_rate_type = 'constant'
        print(self.learning_rate_type)
        print(self.learning_rate)
        loss_class = model_spec['loss_class']
        loss_class = getattr(nn, loss_class)
        self.criterion = loss_class()  # nn.BCEWithLogitsLoss()
        if warmup == 1:
            with open(model_spec["warmup_parameters_path"], "rb") as f:  # Unpickling
                weights = pickle.load(f)
                print(weights)
                self.set_weights(weights)

    def init_warm_up(self, model_weights):
        w_new = copy.deepcopy(self.model.state_dict())
        layer = []
        for i in range(len(model_weights) - 1):
            layer.append(model_weights[i])
        w_new['linear.weight'][0] = torch.FloatTensor(layer)
        w_new['linear.bias'][0] = torch.FloatTensor([model_weights[-1]])
        self.model.load_state_dict(w_new)

    def set_weights(self, model_weights):
        w_new = copy.deepcopy(self.model.state_dict())
        layer = []
        for i in range(len(model_weights) - 1):
            layer.append(model_weights[i])
        w_new['linear.weight'][0] = torch.FloatTensor(layer)
        w_new['linear.bias'][0] = torch.FloatTensor([model_weights[-1]])
        self.model.load_state_dict(w_new)

    def set_weights_rand(self, rand_proportion):
        model_weights = self.get_weights()
        print("RADNOMNESS=============================================================================")
        print(model_weights)
        rand_positions = random.sample(range(0, len(model_weights)), round(len(model_weights) * rand_proportion))
        print(rand_positions)
        w_new = copy.deepcopy(self.model.state_dict())
        layer = []
        for i in range(len(model_weights) - 1):
            if i in rand_positions:
                layer.append(torch.randn(1, 1).item())
            else:
                layer.append(model_weights[i])
        print(layer)
        w_new['linear.weight'][0] = torch.FloatTensor(layer)
        if (len(model_weights) - 1) in rand_positions:
            w_new['linear.bias'][0] = torch.randn(1, 1).item()
        else:
            w_new['linear.bias'][0] = torch.FloatTensor([model_weights[-1]])
        self.model.load_state_dict(w_new)
        print(self.get_weights())

    def get_weights(self):
        model_weights = []
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                for x in layer.weight[0]:
                    model_weights.append(x.item())
                model_weights.append(layer.bias[0].item())
        return model_weights

    def train(self, train_dataset, seed=42, test=False, test_dataset=None, validation=False, validation_dataset=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        epoch_loss = []
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for i in range(self.nb_epochs):
            batch_loss = []

            for (batch_idx, (features, labels, SA)) in enumerate(self.train_loader):
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                labels = labels.view(len(labels), -1)
                labels[labels == 0] = -1
                self.model.zero_grad()

                output = self.model(features.float())

                loss = self.criterion(output, labels.float())
                
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if test:
                test_acc, test_loss, test_metrics = self.test_inference(test_dataset)
                self.test_metrics.append([i, test_acc, test_loss, test_metrics])
                print(test_acc)
                self.model.train()
            if validation:
                pass
            self.update_learning_rate(i)
        return sum(epoch_loss) / len(epoch_loss)


    def update_learning_rate(self, e):
        if (self.learning_rate_type == 'dynamic'):
            if (e != 0) & (e % self.learning_rate_freq == 0) & (self.learning_rate > self.learning_rate_min):
                self.learning_rate = self.learning_rate / self.learning_rate_fraction
                for g in self.optimizer.param_groups:
                    g['lr'] = self.learning_rate
        elif self.learning_rate_type == 'scheduler':
            self.scheduler.step()

    def test_inference(self, testing_dataset, test_batch_size=64):
        # Returns the test accuracy and loss.
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)
        model = copy.deepcopy(self.model)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        features_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad():
            batch_loss = []
            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                features_fairness = np.concatenate((features_fairness, features),
                                                   axis=0) if features_fairness.size else features

                labels[labels == 0] = -1
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels

                labels = labels.view(len(labels), -1)
                # Inference
                output = model(features.float())
                labels = labels.view(len(labels), -1)
                batch_loss.append(self.criterion(output, labels.float()).item())
                # Prediction
                pred_labels = 2 * (output >= 0) - 1
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1)),
                                                      axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1)
                pred_labels = pred_labels.view(len(pred_labels), -1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)
        labels_fairness[labels_fairness == -1] = 0
        pred_labels_fairness[pred_labels_fairness == -1] = 0
        if type(labels_fairness) == torch.Tensor and type(pred_labels_fairness) == torch.Tensor and type(
                features_fairness) == torch.Tensor:
            labels_fairness = labels_fairness.numpy()
            pred_labels_fairness = pred_labels_fairness.numpy()
            features_fairness = features_fairness.numpy()
        additional_metrics = get_eval_metrics_for_classificaton(labels_fairness.astype(int),
                                                                pred_labels_fairness.astype(int))
        fairness_rep = fairness_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int),
                                       testing_dataset.sensitive_att, np.array(self.get_weights()),
                                       testing_dataset.columns)
        dict_metrics = {**additional_metrics, **fairness_rep}
        loss = sum(batch_loss) / len(batch_loss)
        accuracy = correct / total
        return accuracy, loss, dict_metrics
        


    def test_inference_quick(self, testing_dataset, test_batch_size=64):
        # Returns the test accuracy and loss.
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)
        model = copy.deepcopy(self.model)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        features_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad():
            batch_loss = []
            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                features_fairness = np.concatenate((features_fairness, features),
                                                   axis=0) if features_fairness.size else features

                labels[labels == 0] = -1
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels

                labels = labels.view(len(labels), -1)
                # Inference
                output = model(features.float())
                labels = labels.view(len(labels), -1)
                batch_loss.append(self.criterion(output, labels.float()).item())
                # Prediction
                pred_labels = 2 * (output >= 0) - 1
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1)),
                                                      axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1)
                pred_labels = pred_labels.view(len(pred_labels), -1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)
        labels_fairness[labels_fairness == -1] = 0
        pred_labels_fairness[pred_labels_fairness == -1] = 0
        if type(labels_fairness) == torch.Tensor and type(pred_labels_fairness) == torch.Tensor and type(
                features_fairness) == torch.Tensor:
            labels_fairness = labels_fairness.numpy()
            pred_labels_fairness = pred_labels_fairness.numpy()
            features_fairness = features_fairness.numpy()
        fairness_rep = spd_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int),
                                       testing_dataset.sensitive_att,
                                       testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
        return accuracy,  dict_metrics


    def test_inference_quick_EOD(self, testing_dataset, test_batch_size=64):
        # Returns the test accuracy and loss.
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)
        model = copy.deepcopy(self.model)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        features_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad():
            batch_loss = []
            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                features_fairness = np.concatenate((features_fairness, features),
                                                   axis=0) if features_fairness.size else features

                labels[labels == 0] = -1
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels

                labels = labels.view(len(labels), -1)
                # Inference
                output = model(features.float())
                labels = labels.view(len(labels), -1)
                batch_loss.append(self.criterion(output, labels.float()).item())
                # Prediction
                pred_labels = 2 * (output >= 0) - 1
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1)),
                                                      axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1)
                pred_labels = pred_labels.view(len(pred_labels), -1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)
        labels_fairness[labels_fairness == -1] = 0
        pred_labels_fairness[pred_labels_fairness == -1] = 0
        if type(labels_fairness) == torch.Tensor and type(pred_labels_fairness) == torch.Tensor and type(
                features_fairness) == torch.Tensor:
            labels_fairness = labels_fairness.numpy()
            pred_labels_fairness = pred_labels_fairness.numpy()
            features_fairness = features_fairness.numpy()
        fairness_rep = eod_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int),
                                       testing_dataset.sensitive_att,
                                       testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
        return accuracy,  dict_metrics


    def test_inference_quick_discr_idx(self, testing_dataset, test_batch_size=64):
        # Returns the test accuracy and loss.
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)
        model = copy.deepcopy(self.model)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        features_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad():
            batch_loss = []
            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                features_fairness = np.concatenate((features_fairness, features),
                                                   axis=0) if features_fairness.size else features

                labels[labels == 0] = -1
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels

                labels = labels.view(len(labels), -1)
                # Inference
                output = model(features.float())
                labels = labels.view(len(labels), -1)
                batch_loss.append(self.criterion(output, labels.float()).item())
                # Prediction
                pred_labels = 2 * (output >= 0) - 1
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1)),
                                                      axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1)
                pred_labels = pred_labels.view(len(pred_labels), -1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)
        labels_fairness[labels_fairness == -1] = 0
        pred_labels_fairness[pred_labels_fairness == -1] = 0
        if type(labels_fairness) == torch.Tensor and type(pred_labels_fairness) == torch.Tensor and type(
                features_fairness) == torch.Tensor:
            labels_fairness = labels_fairness.numpy()
            pred_labels_fairness = pred_labels_fairness.numpy()
            features_fairness = features_fairness.numpy()
        fairness_rep = discr_idx_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int),
                                       testing_dataset.sensitive_att,
                                       testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
        return accuracy,  dict_metrics










# Step 1: Define the ResNet model architecture
class ImageTabularModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes):
        super(ImageTabularModel, self).__init__()
        
        torch.manual_seed(42)

        # Step 1: Define the ResNet-based image processing module
        self.resnet = resnet18(pretrained=True)
        num_resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the original classification layer

        # Step 2: Define the fully connected layer-based tabular data processing module
        self.tabular_fc = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Step 3: Define the final fully connected layer for binary classification
        self.final_fc = nn.Sequential(
            nn.Linear(num_resnet_features + 32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Sigmoid()  # Use Sigmoid for binary classification to get probabilities
        )

        # Step 4: Initialize the fully connected layers using He initialization
        for m in self.tabular_fc.modules():
            if isinstance(m, nn.Linear):
                if isinstance(m.weight, nn.Parameter):
                    init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    init.zeros_(m.bias.data)

        for m in self.final_fc.modules():
            if isinstance(m, nn.Linear):
                if isinstance(m.weight, nn.Parameter):
                    init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    init.zeros_(m.bias.data)
                    

    def forward(self, images, tabular_data):
        # Process images using the ResNet module
        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0), -1)

        # Process tabular data using the fully connected layer module
        tabular_features = self.tabular_fc(tabular_data)

        # Concatenate the image and tabular features
        combined_features = torch.cat((image_features, tabular_features), dim=1)

        # Feed the concatenated features to the final fully connected layer for binary classification
        output = self.final_fc(combined_features)

        return output



class ResnetPytorch(FLModel):
    """docstring for LogisticRegressionPytorch"""

    def __init__(self, model_spec, warmup, **kwargs):
        super(ResnetPytorch, self).__init__(model_spec)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nb_tabular_features = model_spec['nb_tabular_inputs']
        self.nb_classes = model_spec['nb_outputs']
        self.model = ImageTabularModel(self.nb_tabular_features, self.nb_classes)
        self.nb_epochs = model_spec['nb_epochs']
        self.batch_size = model_spec['batch_size']
        self.learning_rate = model_spec['learning_rate']
        optimizer_class = model_spec['optimizer_class']
        optimizer_class = getattr(torch.optim, optimizer_class)
        try:
            self.weight_decay = model_spec['weight_decay']
        except:
            self.weight_decay = 0
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        try:
            self.learning_rate_type = model_spec['learning_rate_type']
            if self.learning_rate_type == 'dynamic':
                self.learning_rate_freq = model_spec['learning_rate_frequency']
                self.learning_rate_fraction = model_spec['learning_rate_fraction']
                self.learning_rate_min = model_spec['learning_rate_min']
            elif self.learning_rate_type == 'scheduler':
                scheduler_class = model_spec['scheduler_class']
                gamma = model_spec['gamma']
                scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class)
                self.scheduler = scheduler_class(self.optimizer, gamma=gamma)
        except:
            self.learning_rate_type = 'constant'
        print(self.learning_rate_type)
        print(self.learning_rate)
        loss_class = model_spec['loss_class']
        loss_class = getattr(nn, loss_class)
        self.criterion = loss_class()  # nn.BCEWithLogitsLoss()
        if warmup == 1:
            raise NotImplementedError("This function is not yet implemented.")


    def init_warm_up(self, model_weights):
        raise NotImplementedError("This function is not yet implemented.")
    def set_weights(self, model_weights):
        self.model.load_state_dict(model_weights)


    def get_weights(self):
        return copy.deepcopy(self.model.state_dict())

    def train(self, train_dataset, seed=42, test=False, test_dataset=None, validation=False, validation_dataset=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        epoch_accuracy = []
        epoch_loss = []
        print(self.batch_size)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.to(self.device)
        self.model.train()

        for i in range(self.nb_epochs):
            batch_loss = []
            total_correct, total_samples = 0, 0
            for (batch_idx, (images, labels, SA)) in enumerate(self.train_loader):
                (images, labels, SA) = (images.to(self.device), labels.to(self.device), SA.to(self.device))
                labels = labels.view(len(labels), -1)
                self.model.zero_grad()

                output = self.model(images.float(), SA.float())
                loss = self.criterion(output, labels.float())
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())

                predicted_labels = (output >= 0.5).float()  # Threshold at 0.5 for binary classification
                total_correct += torch.sum(predicted_labels == labels).item()
                total_samples += len(labels)

            epoch_accuracy.append( total_correct / total_samples)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print(i, epoch_loss[-1])
            #print(i, epoch_accuracy)
            if test:
                test_acc, test_loss, test_metrics = self.test_inference(test_dataset)
                self.test_metrics.append([i, test_acc, test_loss, test_metrics])
                self.model.train()
            if validation:
                pass
            self.update_learning_rate(i)
        self.model.to('cpu')
        print(i, epoch_accuracy)
        self.test_inference(train_dataset, test_batch_size = self.batch_size)
        return sum(epoch_loss) / len(epoch_loss)


    def update_learning_rate(self, e):
        if (self.learning_rate_type == 'dynamic'):
            if (e != 0) & (e % self.learning_rate_freq == 0) & (self.learning_rate > self.learning_rate_min):
                self.learning_rate = self.learning_rate / self.learning_rate_fraction
                for g in self.optimizer.param_groups:
                    g['lr'] = self.learning_rate
        elif self.learning_rate_type == 'scheduler':
            self.scheduler.step()

    def test_inference(self, testing_dataset, test_batch_size = 64):
        # Returns the test accuracy and loss. 
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)
        model = copy.deepcopy(self.model)
        model.to(self.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = self.criterion.to(self.device)
        SA_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad() :
            batch_loss = []

            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                SA_fairness = np.concatenate((SA_fairness, SA), axis=0) if SA_fairness.size else SA
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                labels = labels.view(len(labels), -1)
                # Inference
                output = model(features, SA)
                labels = labels.view(len(labels), -1)
                batch_loss.append(criterion(output, labels).item())
                # Prediction
                pred_labels = output.round()
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1).cpu()), axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1).cpu()
                pred_labels = pred_labels.view(len(pred_labels), -1)
                #print(labels.view(-1))
                #print(output.view(-1))
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)
        model.to('cpu')
       
        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(SA_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.to('cpu').numpy()
            pred_labels_fairness=pred_labels_fairness.to('cpu').numpy()
            SA_fairness = SA_fairness.to('cpu').numpy()
        additional_metrics = get_eval_metrics_for_classificaton(labels_fairness.astype(int), pred_labels_fairness.astype(int))
        fairness_rep = fairness_report(SA_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, np.array(self.get_weights()), testing_dataset.columns)
        dict_metrics = {**additional_metrics, **fairness_rep}
        loss = sum(batch_loss)/len(batch_loss)
        accuracy = correct / total
        print('ACCCCCCCCURACy  '+str(accuracy))
        return accuracy, loss, dict_metrics
    


    def test_inference_quick_old(self, testing_dataset, test_batch_size = 64):


        
        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(SA_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.to('cpu').numpy()
            pred_labels_fairness=pred_labels_fairness.to('cpu').numpy()
            SA_fairness = SA_fairness.to('cpu').numpy()

        fairness_rep = spd_report(SA_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
        print(accuracy, dict_metrics )
        return accuracy,  dict_metrics


    def test_inference_quick(self, testing_dataset, test_batch_size = 64):
        # Returns the test accuracy and loss.
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)

        self.model.to(self.device)
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = self.criterion.to(self.device)
        SA_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad() :
            batch_loss = []

            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                SA_fairness = np.concatenate((SA_fairness, SA), axis=0) if SA_fairness.size else SA
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                labels = labels.view(len(labels), -1)
                # Inference
                output = self.model(features, SA)
                if torch.isnan(output).any():
                    return -100, {'Discrimination Index': [100 for i in range(testing_dataset.nb_sa)]}
                labels = labels.view(len(labels), -1)
                batch_loss.append(criterion(output, labels).item())
                # Prediction
                pred_labels = output.round()
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1).cpu()), axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1).cpu()
                pred_labels = pred_labels.view(len(pred_labels), -1)
#                print(pred_labels.view(-1))
#                print(output.view(-1))
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)      
        self.model.to('cpu') 


        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(SA_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.to('cpu').numpy()
            pred_labels_fairness=pred_labels_fairness.to('cpu').numpy()
            SA_fairness = SA_fairness.to('cpu').numpy()

        fairness_rep = spd_report(SA_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
#        print(accuracy, dict_metrics )
        return accuracy,  dict_metrics   


    def test_inference_quick_EOD(self, testing_dataset, test_batch_size = 64):
        # Returns the test accuracy and loss.
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)

        self.model.to(self.device)
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = self.criterion.to(self.device)
        SA_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad() :
            batch_loss = []

            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                SA_fairness = np.concatenate((SA_fairness, SA), axis=0) if SA_fairness.size else SA
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                labels = labels.view(len(labels), -1)
                # Inference
                output = self.model(features, SA)
                if torch.isnan(output).any():
                    return -100, {'Discrimination Index': [100 for i in range(testing_dataset.nb_sa)]}
                labels = labels.view(len(labels), -1)
                batch_loss.append(criterion(output, labels).item())
                # Prediction
                pred_labels = output.round()
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1).cpu()), axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1).cpu()
                pred_labels = pred_labels.view(len(pred_labels), -1)
#                print(pred_labels.view(-1))
#                print(output.view(-1))
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)      
        self.model.to('cpu') 


        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(SA_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.to('cpu').numpy()
            pred_labels_fairness=pred_labels_fairness.to('cpu').numpy()
            SA_fairness = SA_fairness.to('cpu').numpy()

        fairness_rep = eod_report(SA_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
#        print(accuracy, dict_metrics )
        return accuracy,  dict_metrics




    def test_inference_quick_discr_idx(self, testing_dataset, test_batch_size = 64):
        # Returns the test accuracy and loss.
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)

        self.model.to(self.device)
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = self.criterion.to(self.device)
        SA_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad() :
            batch_loss = []

            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                SA_fairness = np.concatenate((SA_fairness, SA), axis=0) if SA_fairness.size else SA
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                labels = labels.view(len(labels), -1)
                # Inference
                output = self.model(features, SA)
                if torch.isnan(output).any():
                    return -100, {'Discrimination Index': [100 for i in range(testing_dataset.nb_sa)]}
                labels = labels.view(len(labels), -1)
                batch_loss.append(criterion(output, labels).item())
                # Prediction
                pred_labels = output.round()
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.reshape(-1).cpu()), axis=0) if pred_labels_fairness.size else pred_labels.reshape(-1).cpu()
                pred_labels = pred_labels.view(len(pred_labels), -1)
#                print(pred_labels.view(-1))
#                print(output.view(-1))
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)      
        self.model.to('cpu') 


        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(SA_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.to('cpu').numpy()
            pred_labels_fairness=pred_labels_fairness.to('cpu').numpy()
            SA_fairness = SA_fairness.to('cpu').numpy()

        fairness_rep = discr_idx_report(SA_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
#        print(accuracy, dict_metrics )
        return accuracy,  dict_metrics
        
        
        
###########################################################################################################################################################################################
################################ MLP model
###########################################################################################################################################################################################
        
 # Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, nb_features, nb_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(nb_features, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 32)           # Second fully connected layer
        self.fc3 = nn.Linear(32, nb_classes)   # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
 
class MLPPytorch(FLModel):
    """docstring for MLPPytorch"""
    def __init__(self, model_spec, warmup, **kwargs):
        super(MLPPytorch, self).__init__(model_spec)
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nb_features = model_spec['nb_inputs']
        self.nb_classes = model_spec['nb_outputs']
        self.model = MLPModel(self.nb_features, self.nb_classes)
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
        raise NotImplementedError("This function is not yet implemented.")
        
    def set_weights(self, model_weights):
        self.model.load_state_dict(model_weights)

    def get_weights(self):
        return copy.deepcopy(self.model.state_dict())


    def train(self, train_dataset, seed = 42, test = False, test_dataset = None, validation = False, validation_dataset = None):
        self.model.train()
        epoch_loss = []
        # set seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle= True)
        for i in range(self.nb_epochs):
            batch_loss = []
            for (batch_idx, (features, labels, SA)) in enumerate(self.train_loader):
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                #labels = labels.view(len(labels), -1)
                labels = labels.type(torch.LongTensor).to(self.device)
                self.model.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            print('epoch: {}  loss: {}'.format(i, sum(batch_loss) / len(batch_loss)))
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
        model.to(self.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        criterion = self.criterion.to(self.device)
        features_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad() :
            batch_loss = []
            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                features_fairness = np.concatenate((features_fairness, features), axis=0) if features_fairness.size else features
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                # Inference
                labels = labels.type(torch.LongTensor).to(self.device)
                outputs = model(features)
                batch_loss.append(criterion(outputs, labels).item())
                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.to('cpu').reshape(-1)), axis=0) if pred_labels_fairness.size else pred_labels.to('cpu').reshape(-1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)    
        model.to('cpu')  
        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(features_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.to('cpu').numpy()
            pred_labels_fairness=pred_labels_fairness.to('cpu').numpy()
            features_fairness = features_fairness.to('cpu').numpy()
        additional_metrics = get_eval_metrics_for_classificaton(labels_fairness.astype(int), pred_labels_fairness.astype(int))
        fairness_rep = fairness_report(features_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, np.array(self.get_weights()), testing_dataset.columns)
        dict_metrics = {**additional_metrics, **fairness_rep}
        loss = sum(batch_loss)/len(batch_loss)
        accuracy = correct / total

        return accuracy, loss, dict_metrics
    

 
  
  
    def test_inference_quick(self, testing_dataset, test_batch_size = 64):
        # Returns the test accuracy and loss. 
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)

        self.model.to(self.device)
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        criterion = self.criterion.to(self.device)
        features_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad() :
            batch_loss = []
            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                features_fairness = np.concatenate((features_fairness, features), axis=0) if features_fairness.size else features
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                #labels = labels.view(len(labels), -1)
                # Inference
                labels = labels.type(torch.LongTensor).to(self.device)
                outputs = self.model(features)
                if torch.isnan(output).any():
                    return -100, {'Statistical Parity Difference': [100 for i in range(testing_dataset.nb_sa)]}
                #labels = labels.view(len(labels), -1)
                batch_loss.append(criterion(outputs, labels).item())
                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.to('cpu').reshape(-1)), axis=0) if pred_labels_fairness.size else pred_labels.to('cpu').reshape(-1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)      
        self.model.to('cpu')
        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(SA_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.to('cpu').numpy()
            pred_labels_fairness=pred_labels_fairness.to('cpu').numpy()
            SA_fairness = SA_fairness.to('cpu').numpy()

        fairness_rep = spd_report(SA_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
#        print(accuracy, dict_metrics )
        return accuracy,  dict_metrics



    def test_inference_quick_EOD(self, testing_dataset, test_batch_size = 64):
        # Returns the test accuracy and loss. 
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)

        self.model.to(self.device)
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        criterion = self.criterion.to(self.device)
        features_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad() :
            batch_loss = []
            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                features_fairness = np.concatenate((features_fairness, features), axis=0) if features_fairness.size else features
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                #labels = labels.view(len(labels), -1)
                # Inference
                labels = labels.type(torch.LongTensor).to(self.device)
                outputs = self.model(features)
                if torch.isnan(output).any():
                    return -100, {'Equal Opportunity Difference': [100 for i in range(testing_dataset.nb_sa)]}
                #labels = labels.view(len(labels), -1)
                batch_loss.append(criterion(outputs, labels).item())
                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.to('cpu').reshape(-1)), axis=0) if pred_labels_fairness.size else pred_labels.to('cpu').reshape(-1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)      
        self.model.to('cpu')
        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(SA_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.to('cpu').numpy()
            pred_labels_fairness=pred_labels_fairness.to('cpu').numpy()
            SA_fairness = SA_fairness.to('cpu').numpy()

        fairness_rep = eod_report(SA_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
#        print(accuracy, dict_metrics )
        return accuracy,  dict_metrics

    def test_inference_quick_discr_idx(self, testing_dataset, test_batch_size = 64):
        # Returns the test accuracy and loss. 
        test_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)

        self.model.to(self.device)
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        criterion = self.criterion.to(self.device)
        features_fairness = np.array([])
        labels_fairness = np.array([])
        pred_labels_fairness = np.array([])
        with torch.no_grad() :
            batch_loss = []
            for batch_idx, (features, labels, SA) in enumerate(test_loader):
                features_fairness = np.concatenate((features_fairness, features), axis=0) if features_fairness.size else features
                labels_fairness = np.concatenate((labels_fairness, labels), axis=0) if labels_fairness.size else labels
                (features, labels, SA) = (features.to(self.device), labels.to(self.device), SA.to(self.device))
                SA = SA.view(len(SA), -1)
                #labels = labels.view(len(labels), -1)
                # Inference
                labels = labels.type(torch.LongTensor).to(self.device)
                outputs = self.model(features)
                if torch.isnan(output).any():
                    return -100, {'Discrimination Index': [100 for i in range(testing_dataset.nb_sa)]}
                #labels = labels.view(len(labels), -1)
                batch_loss.append(criterion(outputs, labels).item())
                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels_fairness = np.concatenate((pred_labels_fairness, pred_labels.to('cpu').reshape(-1)), axis=0) if pred_labels_fairness.size else pred_labels.to('cpu').reshape(-1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)      
        self.model.to('cpu')
        if type(labels_fairness)==torch.Tensor and type(pred_labels_fairness)==torch.Tensor and type(SA_fairness)==torch.Tensor:
            labels_fairness=labels_fairness.to('cpu').numpy()
            pred_labels_fairness=pred_labels_fairness.to('cpu').numpy()
            SA_fairness = SA_fairness.to('cpu').numpy()

        fairness_rep = discr_idx_report(SA_fairness, labels_fairness.astype(int), pred_labels_fairness.astype(int), testing_dataset.sensitive_att, testing_dataset.columns)
        dict_metrics = {**fairness_rep}
        accuracy = correct / total
#        print(accuracy, dict_metrics )
        return accuracy,  dict_metrics
