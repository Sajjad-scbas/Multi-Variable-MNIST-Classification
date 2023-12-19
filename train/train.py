import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm


def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer, device):

    nb_batchs = dataloader.batch_size
    model.train()
    for idx, (x, label) in enumerate(tqdm(dataloader)):
        x, label = x.to(device), label.to(device)
        
        pred = model(x.to(device))
        loss = loss_fn(pred, label.long().to(device))
        #loss = F.nll_loss(pred, label)
        
        #Back Pass
        loss.backward()
        
        _, predicted = torch.max(pred, 1)
        
        #Weights Update
        optimizer.step()
        optimizer.zero_grad()
        
        acc = ((predicted == label.to(device)).sum())/nb_batchs

        writer.add_scalar('training_loss', loss.item(), (epoch*len(dataloader)) + idx)
        print(f'training-loss : {loss.item():>7f} | [{(idx+1)}/ {len(dataloader)}]')
        writer.add_scalar('Accuracy_train', acc, (epoch*len(dataloader)) + idx)
        print(f'Accuracy : {acc:>7f} | [{(idx+1)}/ {len(dataloader)}]')

"""

def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer, device):
    
    nb_batchs = dataloader.batch_size
    model.train()
    total_loss = 0
    for idx, (x, label) in enumerate(tqdm(dataloader)):

        label = torch.sparse.torch.eye(10).index_select(dim=0, index=label)
        output, reconstructions, masked = model(x)
        loss = model.loss(x, output, label, reconstructions)
        
        correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(label.data.cpu().numpy(), 1))
        train_loss = loss.item()
        total_loss += train_loss
        
        
        #Back Pass
        loss.backward()
        
        #_, predicted = torch.max(pred, 1)
        
        #Weights Update
        optimizer.step()
        optimizer.zero_grad()
        
        #acc = ((predicted == label.to(device)).sum())/nb_batchs

        writer.add_scalar('training_loss', loss.item(), (epoch*len(dataloader)) + idx)
        print(f'training-loss : {loss.item():>7f} | [{(idx+1)}/ {len(dataloader)}]')
        writer.add_scalar('Accuracy_train', correct/nb_batchs, (epoch*len(dataloader)) + idx)
        print(f'Accuracy : {correct/nb_batchs:>7f} | [{(idx+1)}/ {len(dataloader)}]')

"""


def test_loop(dataloader, model, loss_fn, epoch, writer, device):
    test_loss = 0
    correct = 0
    nb_batchs = dataloader.batch_size
    model.eval()
    with torch.no_grad():
        for idx, (x, label) in enumerate(tqdm(dataloader)):
            pred = model(x.to(device))
            test_loss += loss_fn(pred, label.long().to(device)).item()
            
            _, predicted = torch.max(pred, 1)
            
            correct += (predicted==label.to(device)).sum()/nb_batchs
            
        test_loss /= len(dataloader)
        writer.add_scalar('testing_loss', test_loss, epoch)
        writer.add_scalar('Accuracy_test', correct/len(dataloader), epoch)
        print(f'Test MSE loss : {test_loss:>7f}, Accuracy : {correct/len(dataloader):>7f}')
