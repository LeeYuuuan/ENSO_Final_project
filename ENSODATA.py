
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

class EarthDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    

def Load_Data(datatp='SODA', merge_ft=None , dataset_type=None, deactivate_feature=None):
    PATH = "/home/lzhang51/Documents/Applied_AI_FP/ENSO_data/"
    if datatp == 'SODA':
        train = xr.open_dataset(PATH+'SODA_train.nc')
        label = xr.open_dataset(PATH+'SODA_label.nc')
    if datatp == 'CMIP':
        train = xr.open_dataset(PATH+'CMIP_train.nc')
        label = xr.open_dataset(PATH+'CMIP_label.nc')


    feature_list = ['sst', 't300', 'ua', 'va']
    data_merge = []
    for feature in feature_list:
        data_merge.append(train[feature].values)
    data_merge = np.array(data_merge) # (4, N, 12, 24, 72) | (4645, 12, 24, 72) [numpy.ndarray] CMIP

    if deactivate_feature != None:
        for feature in deactivate_feature:
            data_merge[feature_list.index(feature)] = 0 # set deactivate feature to 0.

    nan_indices = np.isnan(data_merge)
    data_merge[nan_indices] = 0
    data_merge = torch.Tensor(data_merge) # torch.Size([4, 100, 36(24 month + 12 month), 24, 72])
    data_merge = data_merge.permute(1, 0, 2, 3, 4) # torch.Size([100, 4, 36(24 month + 12 month), 24, 72])
    
    if dataset_type is None:
         
        train_all_data = data_merge[:, :, :12, :, :] # torch.Size([100, 4, 12, 24, 72])
        train_all_label = label['nino'][:, 12:].values # torch.Size([100, 24])
        train_all_label = torch.Tensor(train_all_label)
    elif dataset_type == 'feature_to_feature':
        train_all_data = data_merge[:, :, :12, :, :] # torch.Size([100, 4, 12, 24, 72])
        train_all_label = data_merge[:, :, 12:, :, :] # torch.Size([100, 4, 24, 24, 72])
    
    elif dataset_type == 'feature_to_current_label':
        train_all_data = data_merge[:, :, :12, :, :]
        train_all_label = label['nino'][:, :12].values # torch.Size([100, 12])
        train_all_label = torch.Tensor(train_all_label)
        


    if merge_ft != None:
        train_all_data = train_all_data.reshape(train_all_data.shape[0], train_all_data.shape[1] * train_all_data.shape[2], train_all_data.shape[3], train_all_data.shape[4])
    
    
    N = int(len(train_all_label)*0.8) # 8:2 for train / test dataset
    
        
    tensor_train = train_all_data[:N]
    tensor_valid = train_all_data[N:]
    train_label = train_all_label[:N]
    valid_label = train_all_label[N:]


    train_dataset = EarthDataSet(tensor_train.to(torch.float32), train_label.to(torch.float32))
    valid_dataset = EarthDataSet(tensor_valid, valid_label)
    print('Train samples: {}, Valid samples: {}'.format(len(train_label), len(valid_label)))



    
    
    return train_dataset, valid_dataset