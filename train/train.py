import torch

from tqdm import tqdm


def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer, device):
    """
    Trains a model using the given dataloader for one epoch.
    
    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the training data.
        model (torch.nn.Module): The model to train.
        loss_fn (torch.nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating model weights.
        epoch (int): The current epoch number.
        writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard SummaryWriter to log training metrics.
        device (str or torch.device): The device to use for training.
    """

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
