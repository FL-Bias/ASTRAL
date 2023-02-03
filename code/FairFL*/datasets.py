import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class MyDataset(Dataset):
    def __init__(
        self,
        data,
        sensitive_att_name,
        target_var_name,
        nb_features,
        nb_classes,
        transform=None,
    ):
        self.dataframe = data
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.target_var_name = target_var_name
        self.sensitive_att = sensitive_att_name
        self.columns = data.columns.tolist()
        self.columns.remove(target_var_name)
        x = np.array(data.iloc[:, :-1])
        y = np.array(data.iloc[:, -1])
        sa = np.array(data[sensitive_att_name])
        self.data = torch.from_numpy(x).float()
        self.target = torch.from_numpy(y).float()
        self.sa = torch.from_numpy(sa).float()
        if isinstance(sensitive_att_name,list):
            data.drop(sensitive_att_name + [target_var_name], axis=1, inplace=True)
        else:
            data.drop([sensitive_att_name, target_var_name], axis=1, inplace=True)
        self.data_without_sa = torch.from_numpy(np.array(data)).float()
        self.transform = transform
        if (isinstance(sensitive_att_name,list)):
            self.nb_sa = len(sensitive_att_name)
        else :
            self.nb_sa = 1

    def __getitem__(self, index):
        x = self.data[index]
        sa = self.sa[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y, sa#torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.data)
