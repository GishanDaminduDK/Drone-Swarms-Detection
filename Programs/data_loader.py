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

class RadarDataset(Dataset):
    def __init__(self, data_dir, transform = None, scf_save = False, scf_index = -1, clipped=False):
        self.data_dir = data_dir
        try:
            self.label_file = pd.read_csv(data_dir+'labels/labels.csv')
        except:
            self.label_file = -1
        self.data_paths = [os.path.join(data_dir, f"{file_name}") for file_name in os.listdir(data_dir) if file_name.endswith(".csv")]
        self.transform = transform
        self.scf_save = scf_save
        self.scf_index = scf_index
        self.clipped = clipped
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        label_name = data_path.split('/')[-1].split('.')[0]
        data_df = pd.read_csv(data_path)
        channel_1 = torch.tensor(data_df['Channel 1']).to(dtype=torch.float)
        channel_2 = torch.tensor(data_df['Channel 2']).to(dtype=torch.float)
        channel_3 = torch.tensor(data_df['Channel 3']).to(dtype=torch.float)
        channel_4 = torch.tensor(data_df['Channel 4']).to(dtype=torch.float)

        max_data = max(len(channel_1), len(channel_2), len(channel_3), len(channel_4)) #look into this
        if max_data > 10000:
            channel_1 = channel_1[:10000]
            channel_2 = channel_2[:10000]
            channel_3 = channel_3[:10000]
            channel_4 = channel_4[:10000]
        else:
            channel_1 = channel_1[:max_data]
            channel_2 = channel_2[:max_data]
            channel_3 = channel_3[:max_data]
            channel_4 = channel_4[:max_data]


        data_tensor = torch.stack([channel_1, channel_2, channel_3, channel_4], dim=0)
        if isinstance(self.label_file, int):
            label = -1
        else:
            label = self.label_file.loc[self.label_file['Name'] == label_name, 'Class'].values[0]

        if self.transform:
            data_tensor = self.transform(data_tensor)

        if self.scf_save:
            file_name = data_path.split('/')[-1].split('.')[0]
            if self.scf_index == -1:
                save_path = f'{self.data_dir}scf/{file_name}.pt'
            else:
                save_path = f'{self.data_dir}scf_{self.scf_index}/{file_name}.pt'
            torch.save(data_tensor, save_path)

        if self.clipped:
            data_tensor = torch.clamp(data_tensor, max=4)

        return data_tensor, label

class Augmentation: 
    def __init__(
            self,
            snr = 0,
            sequence_length = 2048,
            validation = False,
            mean = torch.tensor([2747.8738, 2718.6184, 2587.1130, 2680.0017], dtype=torch.float),
            std = torch.tensor([670.0775,  943.0208, 1153.2297,  969.1709], dtype=torch.float)
    ):
        self.sequence_length = sequence_length
        self.snr = snr
        self.validation = validation
        self.mean = mean
        self.std = std


    def lowpass(self, data, cutoff_freq = 180, fs=1000, order=3):
        nyquist_freq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist_freq

        b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)

        filtered_data = scipy.signal.filtfilt(b, a, data)
        return filtered_data.copy()
    

    
    def notch(self, data, lower=195, upper=205, fs=1000, order=3):
        nyquist_freq = 0.5 * fs
        lower = lower / nyquist_freq
        upper = upper / nyquist_freq

        b, a = scipy.signal.butter(order, [lower, upper], btype='bandstop', analog=False)

        filtered_data = scipy.signal.filtfilt(b, a, data)
        return filtered_data.copy()



    def __call__(self, sample):
        if len(sample.shape) != 2:
            raise ValueError("Input tensor should have two dimensions")
        
        if self.validation:
            print(sample.shape, sample.shape[1])
            sliced_sample = sample[:,sample.shape[1]//2 - self.sequence_length//2: sample.shape[1]//2 + self.sequence_length//2]
            assert sliced_sample.shape[1] == self.sequence_length, f"Sequence length is {sliced_sample.shape}"
        else:
            total_length = sample.shape[1] - self.sequence_length 
            start_index = random.randint(0, total_length)
            end_index = start_index + self.sequence_length

            sliced_sample = sample[:,start_index:end_index]
        
        sliced_sample = sliced_sample.unsqueeze(-1)
        normalize = transforms.Normalize(self.mean, self.std)
        sliced_sample = normalize(sliced_sample).squeeze(-1)

        if self.snr != 0:
            noise = torch.tensor(np.random.normal(0, self.snr, sliced_sample.shape))
            sliced_sample = torch.add(sliced_sample, noise)

        sliced_sample = self.lowpass(sliced_sample)
        sliced_sample = self.notch(sliced_sample)

        return torch.tensor(sliced_sample)
 
def data_loader(data_dir, batch_size=32, shuffle=False, transform=None, scf_save = False, scf_index = 0, clipped=False):
    return DataLoader(RadarDataset(data_dir, transform=transform, scf_save=scf_save, scf_index=scf_index, clipped=clipped), batch_size=batch_size, shuffle=shuffle)
