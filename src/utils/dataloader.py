import torchvision
import torch


class Dataset:
    def __init__(self, path):
        self.path = path

    def load(self, batch_size, shuffle=True):
        dataset = torchvision.datasets.ImageFolder(
            root=self.path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=shuffle
        )

        return loader
