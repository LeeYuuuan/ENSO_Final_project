
import torch
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ENSODATA import Load_Data, EarthDataSet
from model.CNNModel import CNNModel
from model.LSTMModel import LSTMModel
from model.TransformerModel import TransformerModel, TransformerModel_CNN

from utils import draw_or_save_loss_fig, check_path, create_saving_fig_path
import os

def train(model, train_loader_1,train_loader_2, valid_loader_1, valid_loader_2, criterion, optimizer, n_epochs, device):
    """ Train a model using the given training and validation data loaders """
    
    model.to(device)
    
    train_losses = []
    valid_losses = []
    
    for epoch in range(n_epochs):
        
        # Training
        model.train()
        train_loss = 0.0
        for data, target in train_loader_1:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        for data, target in train_loader_2:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader_1) + len(train_loader_2)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data, target in valid_loader_1:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
            
            for data, target in valid_loader_2:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
        
        valid_loss /= len(valid_loader_1) + len(valid_loader_2)
        valid_losses.append(valid_loss)
        
        tqdm.tqdm.write(f'Epoch {epoch+1}/{n_epochs} : Train Loss {train_loss:.6f}, Valid Loss {valid_loss:.6f}')
        
    print('Finished Training')
    return train_losses, valid_losses



    

def train_type_model(model_type="CNN",
                     dataset_type="ALL",
                     batch_size=64,
                     device="cuda",
                     is_show=False,
                     deactivate_feature=None,
                     is_return=False,
                     lr=0.001,
                     optimizer_type="Adam",
                     n_epochs=200
):
    """ Train a model of the given type on the given dataset """
    
    # Load the dataset
    if dataset_type == "SODA":
        
        train_soda, valid_soda = Load_Data(datatp="SODA", merge_ft=None, deactivate_feature=deactivate_feature)
        
        train_loader = DataLoader(train_soda, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_soda, batch_size=batch_size, shuffle=False)

    elif dataset_type == "CMIP":
        train_cmip, valid_cmip = Load_Data(datatp="CMIP", merge_ft=None, deactivate_feature=deactivate_feature)
        
        train_loader = DataLoader(train_cmip, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_cmip, batch_size=batch_size, shuffle=False)
        
    elif dataset_type == "ALL":
        train_soda, valid_soda = Load_Data(datatp="SODA", merge_ft=None, deactivate_feature=deactivate_feature)
        train_cmip, valid_cmip = Load_Data(datatp="CMIP", merge_ft=None, deactivate_feature=deactivate_feature)
        
        train_loader_1 = DataLoader(train_soda, batch_size=batch_size, shuffle=True)
        valid_loader_1 = DataLoader(valid_soda, batch_size=batch_size, shuffle=False)
        train_loader_2 = DataLoader(train_cmip, batch_size=batch_size, shuffle=True)
        valid_loader_2 = DataLoader(valid_cmip, batch_size=batch_size, shuffle=False)
        
    else:
        raise ValueError(f"Invalid dataset type '{dataset_type}'")
    
    # Load the model
    criterion = torch.nn.MSELoss()
    
    if model_type == "CNN":
        model = CNNModel()
    elif model_type == "LSTM":
        model = LSTMModel()
    elif model_type == "Transformer":
        model = TransformerModel()
    elif model_type == "Transformer_CNN":
        model = TransformerModel_CNN()
    else:
        raise ValueError(f"Invalid model type '{model_type}'")
    
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Invalid optimizer type '{optimizer_type}'")

    # Train the model
    
    train_losses, valid_losses = train(model, train_loader_1,train_loader_2, valid_loader_1, valid_loader_2, criterion, optimizer, n_epochs, device)
    
    # Save the training and validation loss plot
    root_path = os.path.join('results', model_type, "losses/") 
    check_path(root_path)
    path = create_saving_fig_path(root_path, dataset_type, model_type, deactivate_feature, lr, optimizer_type)
    draw_or_save_loss_fig(train_losses, valid_losses, save_path=path, is_show=is_show)
    
    if is_return:
        return train_losses, valid_losses
    else:
        return None


def train_mul_models(Number=0,
                     is_save=True,
                     is_show=True,
                     model_info=[
                            {"model_name": "CNN", "dataset_type": "ALL", "deactivate_feature": None, "optimizer_type": "Adam", "lr": 0.001, "n_epochs": 200},
                            {"model_name": "LSTM", "dataset_type": "ALL", "deactivate_feature": None, "optimizer_type": "Adam", "lr": 0.001, "n_epochs": 200},
                            {"model_name": "Transformer", "dataset_type": "ALL", "deactivate_feature": None, "optimizer_type": "Adam", "lr": 0.001, "n_epochs": 200},
                     ],
                     draw_train_loss=False,
                     ):
    train_loss_list = []
    valid_loss_list = []
    for info in model_info:
        model_name, d_type, deactivate_feature, optimizer_type, lr, n_epochs = info["model_name"], info["dataset_type"], info["deactivate_feature"], info["optimizer_type"], info["lr"], info["n_epochs"]
        train_loss, valid_loss = train_type_model(model_type=model_name, dataset_type=d_type, deactivate_feature=deactivate_feature, is_show=is_show, is_return=True, lr=lr, optimizer_type=optimizer_type, n_epochs=n_epochs)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
    
    fig, ax = plt.subplots()
    for i, info in enumerate(model_info):
        model_name, d_type, deactivate_feature, optimizer_type, lr, n_epochs = info["model_name"], info["dataset_type"], info["deactivate_feature"], info["optimizer_type"], info["lr"], info["n_epochs"]
        if deactivate_feature:
            ax.plot(valid_loss_list[i], label=f'{model_name}_{optimizer_type}_{lr}_without_{deactivate_feature} Validation Loss')
            if draw_train_loss:
                ax.plot(train_loss_list[i], label=f'{model_name}_{optimizer_type}_{lr}_without_{deactivate_feature} Training Loss')
        else:
            ax.plot(valid_loss_list[i], label=f'{model_name}_{optimizer_type}_{lr} Validation Loss')
            if draw_train_loss:
                ax.plot(train_loss_list[i], label=f'{model_name}_{optimizer_type}_{lr} Training Loss')
            
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss comparison')
    ax.legend()
    if is_save:
        check_path('results/comparison')
        plt.savefig(f'results/comparison/comparison_{Number}.png')
        Number += 1
    if is_show:
        plt.show()
    plt.close(fig)
    
    


    

        