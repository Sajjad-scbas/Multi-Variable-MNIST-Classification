import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST


def download_mnist(path : str = './data', train = True):
    """
    transform = transforms.Compose([
        transforms.RandomRotation(10),           # Random rotation by up to 10 degrees
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random horizontal and vertical translation
        transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),  # Random resizing and cropping
        transforms.RandomHorizontalFlip(),      # Random horizontal flip
        transforms.ToTensor()                # Convert PIL Image to PyTorch tensor
    ])
    """
    
    transform = torchvision.transforms.Compose([
        #transforms.AutoAugment(),
        transforms.RandAugment(),
        #transforms.AugMix(),
        #transforms.TrivialAugmentWide(),
        transforms.ToTensor()
    ])
    
    dataset = MNIST(root=path, train=train, download=True, transform=transform)
    return dataset
