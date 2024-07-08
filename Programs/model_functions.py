import numpy as np
import torch
import torch.nn as nn
from scf import *

def train_model(model, epochs, criterion, optimizer, data_loader, device='cpu'):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        correct_predictions = 0
        total_samples = 0

        for inputs, labels in data_loader:
            print(inputs.shape)
            inputs = inputs.to(dtype=torch.float)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        accuracy = correct_predictions / total_samples
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss}, Training accuracy: {accuracy}')

    print(f"Training for {epochs} epochs is done")
    return loss, optimizer


def evaluate_model(model, data_loader, device='cpu'):
    model = model.to(device)
    model.eval()

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(dtype=torch.float)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    print(f'Validation accuracy: {accuracy * 100 :.2f}%')



def train_model_binary(model, epochs, criterion, optimizer, data_loader, device='cpu'):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        correct_predictions = 0
        total_samples = 0

        for inputs, labels in data_loader:
            inputs = inputs.to(dtype=torch.float)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).reshape(-1)
            loss = criterion(outputs, labels.to(torch.float32))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            predicted = (outputs > 0).float()
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        accuracy = correct_predictions / total_samples
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss}, Training accuracy: {accuracy}')

    print(f"Training for {epochs} epochs is done")
    return loss, optimizer



def evaluate_model_binary(model, data_loader, device='cpu'):
    model = model.to(device)
    model.eval()

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(dtype=torch.float)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).reshape(-1)
            predicted = (outputs > 0).float()
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    print(f'Validation accuracy: {accuracy * 100 :.2f}%')
    return accuracy * 100


def save_model(save_path, model, model_name, optimizer=None, loss=None, epoch=None, state_dict=True):

    if state_dict:
        training_state = {
            "Model state": model.state_dict(),
            "Optimizer state": optimizer.state_dict(),
            "Epoch": epoch,
            "Loss": loss
        }
        torch.save(training_state, f'{save_path}/{model_name}.pt')
    else:
        torch.save(model, f'{save_path}/{model_name}.pt')



def load_model(path):
    model = torch.load(path)
    return model



def load_model_from_state_dict(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model