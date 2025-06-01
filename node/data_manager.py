import torch
from torchvision import datasets, transforms


class MNISTDataManager:
    """
    Manages data for the MNIST dataset.
    """

    def __init__(self, node_id: int, num_nodes: int):
        """
        Initializes the data manager with specified batch sizes.
        """
        self.node_id = node_id
        self.num_nodes = num_nodes

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)

    def get_num_train_samples(self) -> int:
        """
        Returns the number of training samples.
        """
        quotient, remainder = divmod(self.train_set.__len__(), self.num_nodes)
        return quotient + (1 if remainder > self.node_id else 0)
    
    def get_batch_samples(self, indices: list[int]) -> None:
        """
        Returns a batch of samples from the dataset.
        """
        return torch.utils.data.Subset(self.train_set, indices)