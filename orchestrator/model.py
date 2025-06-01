import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Simple Neural Network model.
    """
    def __init__(self):
        """
        Initialize the model.
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.activations = {}

    def forward(self, x):
        """
        Define the forward pass of the model.
        """
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
    
    def forward_partial(self, x):
        """
        Define the partial forward pass of the model.
        """
        x = self.fc3(x)
        self.activations["fc3"] = x
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)