import os
import numpy as np
import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe, load_from_arff_to_dataframe
import torch
import torch.nn.functional as f
from torch.utils.data import Dataset
from sklearn.utils import class_weight

def load_ucr_dataset(dataset, split):
    datadir = './data/Univariate_ts'
    if split in ['TRAIN', 'TEST']:
        filename = dataset + '_' + split
        filepath = os.path.join(datadir, dataset, filename)
        if os.path.isfile(filepath + '.ts'):
            data, labels = load_from_tsfile_to_dataframe(filepath + '.ts')
        elif os.path.isfile(filepath + '.arff'):
            data, labels = load_from_arff_to_dataframe(filepath + '.arff')
        else:
            raise ValueError("Invalid dataset")
    else:
        raise ValueError("Invalid split value")

    return data, labels


def load_uea_dataset(dataset, split):
    datadir = './data/Multivariate_ts'
    if split in ['TRAIN', 'TEST']:
        filename = dataset + '_' + split
        filepath = os.path.join(datadir, dataset, filename)
        if os.path.isfile(filepath + '.ts'):
            data, labels = load_from_tsfile_to_dataframe(filepath + '.ts')
        elif os.path.isfile(filepath + '.arff'):
            data, labels = load_from_arff_to_dataframe(filepath + '.arff')
        else:
            raise ValueError("Invalid dataset")
    else:
        raise ValueError("Invalid split value")

    return data, labels


class TimeSeriesWithLabels(Dataset):
    def __init__(self, dataset, datatype, split, op, **kwargs): 
        super().__init__()
        self.dataset = dataset

        if datatype == 'univar':
            data, labels = load_ucr_dataset(dataset, split)
        elif datatype == 'multivar':
            data, labels = load_uea_dataset(dataset, split)
        else:
            raise ValueError("Invalid vartype")

        class_weight_vec = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(labels), y=labels)
        self.weight = class_weight_vec
        
        self.data, self.labels = self._preprocess(data, labels)
        
        if op=='train':
            data_list = []
            label_list = []
            for i in range(len(np.unique(self.labels))):
                data_list.append(self.data[np.where(self.labels==i)][:int(len(self.labels[np.where(self.labels==i)])*0.8)])
                label_list.append(self.labels[np.where(self.labels==i)][:int(len(self.labels[np.where(self.labels==i)])*0.8)])
            if len(torch.cat(data_list))!=0:
                self.data = torch.cat(data_list)
                self.labels = torch.cat(label_list)                                
        elif op=='valid':
            data_list = []
            label_list = []
            for i in range(len(np.unique(self.labels))):
                data_list.append(self.data[np.where(self.labels==i)][int(len(self.labels[np.where(self.labels==i)])*0.8):])
                label_list.append(self.labels[np.where(self.labels==i)][int(len(self.labels[np.where(self.labels==i)])*0.8):])
            if len(torch.cat(data_list))!=0:
                self.data = torch.cat(data_list)
                self.labels = torch.cat(label_list) 
        else:
            pass     
        
        self.input_size = self.data.shape[1]
        self.timelength = self.data.shape[2]
        self.num_classes = len(np.unique(labels))

    def _preprocess(self, data, labels):
        data = np.array([np.array([data.values[iidx, vidx].to_numpy(dtype=np.float) \
                                for vidx in range(data.values.shape[1])]) \
                                for iidx in range(data.values.shape[0])]) 
        data = torch.Tensor(data)
        data[torch.isnan(data)] = 0.0
        
        label2idx = {label: idx for idx, label in enumerate(np.unique(labels))}
        labels = np.array([label2idx[label] for label in labels])
        labels = torch.LongTensor(labels.astype(np.float))
        
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'labels': self.labels[idx],
        } 