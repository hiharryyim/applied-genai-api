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


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = self.transform(sample["img"].convert("RGB"))
        return image, int(sample["label"])


def get_cifar10_loader(batch_size=128, train=True):
    # CIFAR-10 from Hugging Face, the same copy used in Assignment 2
    from datasets import load_dataset

    data = load_dataset("uoft-cs/cifar10", split="train" if train else "test")

    # RGB pixels from [0, 1] to [-1, 1], the range both CIFAR-10 models work in
    flip = [transforms.RandomHorizontalFlip()] if train else []
    transform = transforms.Compose(flip + [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = torch.utils.data.DataLoader(
        CIFAR10Dataset(data, transform),
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=0
    )

    return loader
