import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.model_task1 import SampleCNN

from train.train import train_loop, test_loop

from data.download_mnist import download_mnist

if __name__ == "__main__":

    path = './data'
    dataset = download_mnist(path)
    dataset1 = download_mnist(path, train = False)


    width, height = 28, 28  
 
    # the Parameters we need to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32   
    parameters = {
        'num_classes' : 10, 
        'channels' : [1, 4, 8, 16, 32, 64],
        'kernels' : [3, 3, 3, 3,3],
        'strides' : [1, 1, 1, 1,1],
        'fc_features' : 128,

        'batch_size' : 64,
        'device' : device,
        'lr': 1e-3,
        'nb_epochs': 10,
    }
    

    train_dataloader = DataLoader(dataset, batch_size=parameters['batch_size'])
    train_dataloader1 = DataLoader(dataset1, batch_size=parameters['batch_size'])
    


    # Essential for TensorBoard Tool 
    writer = SummaryWriter(f"runs/model13 == batch_size = {parameters['batch_size']}, batch_size = {parameters['batch_size']}, lr = {parameters['lr']}")
    
    #save model parameters
    with open(f"model_N13_params.pkl", "wb") as file:
        pickle.dump(parameters, file)

    
    model = SampleCNN(
        num_classes=parameters['num_classes'], 
        channels=parameters['channels'],
        kernels=parameters['kernels'],
        strides=parameters['strides'],
        fc_features=parameters['fc_features'])
    
    
    model = model.to(dtype = dtype, device= device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])
    criterion = nn.CrossEntropyLoss()

        
    nb_epochs = parameters['nb_epochs']
    # The optimization loops
    for epoch in range(nb_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, criterion, optimizer, epoch, writer, device)
        train_loop(train_dataloader1, model, criterion, optimizer, epoch, writer, device)

    #save the model weights
    torch.save(model.state_dict(), f"model_N13.pt")
    
    print('Done')
    