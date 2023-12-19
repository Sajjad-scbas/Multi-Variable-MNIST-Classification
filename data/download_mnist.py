import torch
import cv2
import numpy as np

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

# Define a custom transform that applies downsampling and padding based on probabilities
class CustomTransform(object):
    def __init__(self, downscale_probability):
        self.downscale_probability = downscale_probability

    def __call__(self, image):
        # With a certain probability, apply downsampling
        #if torch.rand(1).item() < self.downscale_probability:
        image = transforms.Resize((20,20))(image)

        # With a certain probability, apply padding with zeros
        padding_size = (28 - image.shape[-1])//2
        padding = transforms.Pad((padding_size, padding_size, padding_size, padding_size), fill=0)
        image = padding(image)
            
        return image



def download_mnist(path : str = './data'):
    transform = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize(0.130, 0.308)
        ])
    dataset = MNIST(root=path, train=True, download=True, transform=transform)
    return dataset



if __name__ == "__main__":
    x = torch.randn(2,1,28,28)
    tran = CustomTransform(0.2)
    y = tran.__call__(x)
    print('Done')



