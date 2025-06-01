import logging
import torch
import torch.nn.functional as F
from typing import Any, Dict

from data_manager import MNISTDataManager
from model import Net

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TaskProcessor:
    """
    Handles the processing of tasks received by the node.
    This encapsulates the 'business logic' of the node.
    """
    def __init__(self, node_id: int, num_nodes: int, device: torch.device):
        self.node_id = node_id
        self.data_manager = MNISTDataManager(node_id, num_nodes)
        self.model = Net().to(device)
        self.device = device

    def get_num_samples(self) -> int:
        """
        Fetches the number of samples from the DataManager.
        """
        logging.info(f"TaskProcessor {self.node_id}: Completed task to get number of samples.")
        return self.data_manager.get_num_train_samples()
    
    def forward_pass(self, task_data: Dict[str, Any]) -> None:
        """
        Forward pass of a neural network.
        """
        indices = task_data.get("indices")
        model_parameters = task_data.get("model_parameters")

        # Get the batch of samples
        samples = self.data_manager.get_batch_samples(indices)
        batch_size = len(samples)

        # Parameters are loaded into the model
        self.model.load_state_dict(model_parameters)

        # Prepare data and target
        train_loader = torch.utils.data.DataLoader(samples, batch_size=batch_size)
        data, target = next(iter(train_loader))
        data, target = data.to(self.device), target.to(self.device)
        
        # Forward pass
        output = self.model(data)

        # Compute loss
        loss = F.nll_loss(output, target)

        # fc4
        grad_z4 = torch.exp(output)
        grad_z4[torch.arange(batch_size), target] -= 1.0

        results = {
            "loss": loss,
            "act_fc2": self.model.activations["fc2"],
            "grad_z4": grad_z4,
        }
        logging.info(f"TaskProcessor {self.node_id}: Completed forward pass task.")
        return results

    def backward_second(self, task_data: Dict[str, Any]) -> None:
        """
        Compute the gradients of the second layer.
        """
        grad_z2 = task_data.get("grad_z2")
        self.padding_left = task_data.get("padding_left")
        self.padding_right = task_data.get("padding_right")
        
        act_fc1 = torch.cat((
            torch.zeros(self.padding_left, self.model.activations["fc1"].shape[1]),
            self.model.activations["fc1"],
            torch.zeros(self.padding_right, self.model.activations["fc1"].shape[1])
        ), dim=0)

        grad_fc2 = torch.matmul(grad_z2.T, act_fc1)

        results = {
            "grad_fc2": grad_fc2,
        }
        logging.info(f"TaskProcessor {self.node_id}: Completed backward for second layer task.")
        return results
    
    def backward_first(self, task_data: Dict[str, Any]) -> None:
        """
        Compute the gradients of the first layer.
        """
        grad_z1 = task_data.get("grad_z1")
        
        x = torch.cat((
            torch.zeros(self.padding_left, self.model.activations["x"].shape[1]),
            self.model.activations["x"],
            torch.zeros(self.padding_right, self.model.activations["x"].shape[1])
        ), dim=0)

        grad_fc1 = torch.matmul(grad_z1.T, x)

        results = {
            "grad_fc1": grad_fc1,
        }
        logging.info(f"TaskProcessor {self.node_id}: Completed backward for first layer task.")
        return results