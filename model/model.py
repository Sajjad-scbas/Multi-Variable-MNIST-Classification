import torch 
import torch.nn as nn 

from typing import Sequence


# Define a Sample CNN Architecture 
class SampleCNN(nn.Module):
    def __init__(self, num_classes : int, channels : Sequence[int], kernels : Sequence[int], strides : Sequence[int], fc_features : int):
        super(SampleCNN, self).__init__()
        
               
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i+1], 
                kernel_size=kernels[i],
                stride=strides[i],
                padding=1
            )
            for i in range(5)
        ])
                
        
        self.relus = nn.ModuleList([
            nn.ReLU()
            for _ in range(6)
        ])
        
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(
                kernel_size=2, 
                stride=2
            )
            for _ in range(5)
        ])

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(channels[-1] * 1 * 1,  fc_features)
        self.fc2 = nn.Linear(fc_features, num_classes)
    
    
    def forward(self, x):
        for i in range(5):
            x = self.convs[i](x)
            x = self.relus[i](x)
            if i != 0 :
                x = self.maxpools[i](x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relus[-1](x)
        x = self.fc2(x)
        
        return x
    

    
if __name__ == "__main__":
    # Parameters 
    num_classes = 10 
    channels = [1, 8, 16, 32, 64]
    kernels = [3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1]
    fc_features = 128
    
    model = SampleCNN(num_classes, channels, kernels, strides, fc_features)
    x = model(torch.randn(3, 1, 28, 28))
    print('Done')
