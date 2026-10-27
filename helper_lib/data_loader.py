import torch
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True):
    # Scale MNIST pixels from [0, 1] to [-1, 1] to match the generator's Tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0
    )

    return loader
