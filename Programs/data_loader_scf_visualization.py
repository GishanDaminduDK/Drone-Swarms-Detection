import os
import sys
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy
from scf import *

class RadarDatasetSCF(Dataset):
    def __init__(self, data_dir, transform = None, binary=False):
        self.data_dir = data_dir
        self.label_file = pd.read_csv(data_dir+'labels/labels.csv')
        self.data_paths = [os.path.join(data_dir, f"{file_name}") for file_name in os.listdir(data_dir) if file_name.endswith(".pt")]
        self.transform = transform
        self.binary = binary

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        label_name = data_path.split('/')[-1].split('.')[0]
        data_tensor = torch.load(data_path)

        label = self.label_file.loc[self.label_file['Name'] == label_name, 'Binary Class'].values[0]
        class_label = self.label_file.loc[self.label_file['Name'] == label_name, 'Class'].values[0]

        if self.transform:
            data_tensor = self.transform(data_tensor)
        
        # return data_tensor, label, label_name, class_label
        return data_tensor[0].unsqueeze(0), label, label_name, class_label


def data_loader_visualize(data_dir, batch_size=32, shuffle=False, transform=None, binary = False):
    return DataLoader(RadarDatasetSCF(data_dir, transform=transform, binary=binary), batch_size=batch_size, shuffle=shuffle)