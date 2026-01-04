import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from model import Net
from typing import Any

torch.manual_seed(1)


class ModelManager:
    """
    Manages the neural network model, its optimizer, and learning rate scheduler.
    It also handles data loading for testing and orchestrates the distributed
    forward and backward passes.
    """

    def __init__(self, lr: float, gamma: float, device: torch.device):
        """
        Initializes the ModelManager with a neural network model, optimizer,
        learning rate scheduler, and test data loader.
        """
        self.model = Net().to(device)
        self.device = device
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_set = datasets.MNIST('../data', train=False, transform=transform, download=True)
        self.test_loader = torch.utils.data.DataLoader(test_set)

    def get_parameters(self):
        """
        Returns the current state dictionary of the model's parameters.
        """
        return self.model.state_dict()
    
    def update_parameters(self):
        """
        Performs a single optimization step (updates model parameters)
        based on the computed gradients.
        """
        self.optimizer.step()

    def step_scheduler(self):
        """
        Steps the learning rate scheduler, adjusting the learning rate
        according to the defined schedule.
        """
        self.scheduler.step()
    
    def forward_pass(self, task_responses: dict[str, Any], batch: dict[int, list[int]]):
        """
        Performs the forward pass for the orchestrator's part of the model
        and calculates gradients for the orchestrator's layers based on
        responses from distributed nodes.
        """
        self.model.train()

        batch_size = 0
        total_loss = 0
        activations = []
        grads = []
        for node_id in task_responses:
            batch_size += len(batch[node_id])
            total_loss += task_responses[node_id]["loss"] * len(batch[node_id])
            activations.append(task_responses[node_id]["act_fc2"])
            grads.append(task_responses[node_id]["grad_z4"])
        act_fc2 = torch.cat(activations, dim=0)
        grad_z4 = torch.cat(grads, dim=0) / batch_size

        _ = self.model.forward_partial(act_fc2)
        
        # fc4
        self.model.fc4.weight.grad = torch.matmul(grad_z4.T, self.model.activations["fc3"])
        self.model.fc4.bias.grad = torch.sum(grad_z4, 0)

        # fc3
        grad_z3 = torch.matmul(grad_z4, self.model.fc4.weight)

        self.model.fc3.weight.grad = torch.matmul(grad_z3.T, act_fc2)
        self.model.fc3.bias.grad = torch.sum(grad_z3, 0)

        # fc2
        grad_z2 = torch.matmul(grad_z3, self.model.fc3.weight)

        self.model.fc2.bias.grad = torch.sum(grad_z2, 0)

        # Padding
        padding_left, padding_right = {}, {}
        size_accum = 0
        for node_id in batch:
            padding_left[node_id] = size_accum
            size_accum += len(batch[node_id])
            padding_right[node_id] = batch_size - size_accum

        return total_loss / batch_size, grad_z2, padding_left, padding_right
    
    def backward_second(self, task_responses: dict[str, Any], grad_z2: torch.Tensor):
        """
        Performs the second part of the backward pass on the orchestrator,
        using gradients received from nodes for fc2 and calculating gradients
        for fc1.
        """
        grads = None
        for node_id in task_responses:
            if grads == None:
                grads = task_responses[node_id]["grad_fc2"]
            else:
                grads = torch.add(grads, task_responses[node_id]["grad_fc2"])
        
        # fc2
        self.model.fc2.weight.grad = grads
        
        # fc1
        grad_z1 = torch.matmul(grad_z2, self.model.fc2.weight)

        self.model.fc1.bias.grad = torch.sum(grad_z1, 0)
        
        return grad_z1
    
    def backward_first(self, task_responses: dict[str, Any], grad_z1: torch.Tensor):
        """
        Performs the final part of the backward pass on the orchestrator,
        using gradients received from nodes for fc1.
        """
        grads = None
        for node_id in task_responses:
            if grads == None:
                grads = task_responses[node_id]["grad_fc1"]
            else:
                grads = torch.add(grads, task_responses[node_id]["grad_fc1"])

        # fc1
        self.model.fc1.weight.grad = grads

    def test(self):
        """
        Evaluates the model's performance on the test dataset.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        logging.info('Orchestrator: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))
