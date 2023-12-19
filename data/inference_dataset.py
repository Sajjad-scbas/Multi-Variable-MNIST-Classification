import torch 
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path):
        
        
        #self.data = data
        #self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        if self.transforms:
            sample = self.transforms(sample)

        return sample, label