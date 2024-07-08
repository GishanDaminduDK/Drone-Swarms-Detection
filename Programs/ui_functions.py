import os
import torch
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
from scf import scf_cube_old
from data_loader import data_loader, Augmentation
from data_reader import SerialDataReader
from model_types import efficientnet_1



def get_data(file_name = 'data/example.csv', port='COM4', baud_rate=500000, duration = 4):
    data_reader = SerialDataReader(port, baud_rate, file_name)
    data_reader.write_data(duration=duration, file_name=file_name)

def get_scf(file_dir, node):

    SCF = lambda x: scf_cube_old(x, node)
    transform_val = transforms.Compose([Augmentation(snr = 0, validation=True), SCF])
    dataloader = data_loader(file_dir, batch_size=1, shuffle=False, transform=transform_val, scf_save=False, clipped=True)
    data, label = next(iter(dataloader))

    return data


def scf_save(data, filepath):
    data = torch.clamp(data[0,2,:,:], max=5)
    # data = data[0,0,:,:]
    plt.matshow(data.cpu().numpy())
    plt.colorbar()
    plt.title("SCD visualization")

    filepath = os.path.join(filepath, 'example.png')
    plt.savefig(filepath)

    plt.clf()

def get_output(data, state_dict_path, device='cpu'):
    model = efficientnet_1.Model(in_channels=4)
    model.unfreeze_params()
    state_dict = torch.load(state_dict_path)
    data = data.to(dtype=torch.float)
    data = data.to(device)

    model.load_state_dict(state_dict['Model state'])
    model.eval()
    model = model.to(device)

    output = int(model(data).item() > 0)
    return output
