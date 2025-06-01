# Traversal Learning (TL) Orchestrator Module

This directory contains the source code for the `Orchestrator` component, the central brain of the Traversal Learning (TL) framework. The orchestrator is responsible for managing the entire distributed training process, coordinating all connected nodes, and performing the critical centralized backward propagation.

## Role of the Orchestrator in Traversal Learning

The TL Orchestrator's key responsibilities include:

* **Connection Management**: The orchestrator queries all participating nodes to collect index ranges. It uses the `connection_manager` to manage active connections.
* **Virtual Batch Creation**: The orchestrator is tasked with generating virtual batches and planning the sequential node visits of the model during FP, aligning them with the ordered index of the data within these batches. The orchestrator maintains virtual batches with data indices of the local samples in a random order.
* **Task Scheduling**: The orchestrator initiates the training process by implementing the traversal plan generated during the virtual batch creation phase. This scheduler determines the sequence in which nodes are visited and assigns specific subsets of data from the virtual batches to each node.
* **Centralized Backward Propagation (BP)**: The orchestrator performs global backward propagation (BP) based on the first-layer activations, first-layer gradients, and last-layers gradients collected from the nodes. This centralized approach addresses common challenges in decentralized learning methods, such as model drift and inconsistent gradient updates, ensuring synchronized parameter optimization.
* **Model Synchronization**: Once BP is complete, the orchestrator redistributes the updated model to all nodes, enabling the next iteration of FP. This ensures that every node operates with the most recent global parameters in the next iteration.
* **Overall Training Management**: Manages training epochs, evaluates model performance, and applies learning rate schedules.

## Directory Structure

```bash
orchestrator/
├── connection_manager.py
├── connection.py
├── data_handler.py
├── main.py
├── model_manager.py
├── model.py
├── orchestrator.py
└── task_scheduler.py
```

## File Descriptions

* **`connection_manager.py`**:
    * Manages listening for and accepting new client (node) connections.
    * Maintains a registry of all active and connected nodes.
    * Handles the initial handshake with connecting nodes, including ID verification.
    * Provides mechanisms for the orchestrator to wait until all expected nodes are connected before starting training.
* **`connection.py`**:
    * Represents a single TCP connection to a client node.
    * Encapsulates the socket object and connection status.
    * Implements TCP Keepalive options to maintain robust connections and detect disconnections.
* **`data_handler.py`**:
    * (Shared with the `node` module) Provides static methods for robust serialization, compression (`zlib`), deserialization, and efficient sending/receiving of data over network sockets. Ensures consistent communication protocol between orchestrator and nodes.
* **`main.py`**:
    * The primary entry point for starting the Orchestrator server.
    * Parses configuration arguments such as number of nodes, epochs, batch size, learning rate, etc.
    * Initializes and runs the `Orchestrator` class, starting the distributed training process.
* **`model_manager.py`**:
    * Manages the global deep learning model, optimizer (`Adadelta`), and learning rate scheduler (`StepLR`).
    * Responsible for loading model parameters for distribution to nodes.
    * Performs the crucial **centralized backward propagation**, using the activations and gradients collected from nodes to update the model's parameters.
    * Handles model evaluation on a test dataset.
* **`model.py`**:
    * Defines the complete neural network architecture (`Net`) used by the orchestrator. This model is identical to the one on the nodes.
    * It includes a `forward_partial` method, allowing the orchestrator to perform specific forward passes on intermediate activations received from nodes, facilitating efficient centralized BP.
    * Crucially stores intermediate activations during its forward pass for BP calculations.
* **`orchestrator.py`**:
    * The core class that orchestrates the entire TL training process.
    * Manages the training loop across epochs and batches.
    * Coordinates the creation of virtual batches and the traversal plan.
    * Delegates tasks to `task_scheduler` and interacts with `model_manager` for centralized BP and model updates.
    * Ensures synchronization with nodes using `connection_manager`.
* **`task_scheduler.py`**:
    * Schedules and sends specific training tasks (e.g., "forward_pass", "backward_second", "backward_first") to active nodes.
    * Collects and aggregates responses (activations, gradients) from nodes.
    * Manages timeouts and handles potential communication errors during task distribution and response collection.

## How to Run the Orchestrator

Refer to the main `README.md` in the root directory for detailed instructions on setting up and running the entire Traversal Learning system, including the orchestrator.

```bash
python orchestrator/main.py --num_nodes <number_of_nodes> --epochs <num_epochs> --batch_size <batch_size> --lr <learning_rate> --gamma <gamma_for_scheduler> [--no-accel]