# Traversal Learning (TL) Node Module

This directory contains the source code for the `Node` component in the Traversal Learning (TL) framework. Each instance of this module represents a single distributed client that holds its local data and participates in the TL training process under the coordination of an `Orchestrator`.

## Role of a Node in Traversal Learning

A TL Node is responsible for:

* **Local Data Management**: Stores and manages its private local dataset. Raw data never leaves the node. Each node independently indexes its local dataset.
* **Partial Forward Propagation (FP)**: The orchestrator first separates the forward propagation (FP) concerns to the nodes. During FP, the model traverses through the nodes in the specified order, with each node processing its allocated data.
* **Gradient Computation**: Each node computes the second-layer activations and last-layer gradients for its local data subset. Each node also performs forward propagation through the full model locally to compute the predicted outputs and true labels, then calculates the last-layer gradient.
* **Secure Communication**: Nodes send their second-layer activations and last-layer gradients to the orchestrator. By transmitting only these specific values, TL minimizes communication overhead compared to transferring the entire model or activations and gradients from all layers.
* **Model Synchronization**: Receives updated model parameters from the orchestrator.

## Directory Structure

```bash
node/
├── data_handler.py
├── data_manager.py
├── main.py
├── model.py
├── network_client.py
├── node.py
└── task_processer.py
```

## File Descriptions

* **`data_handler.py`**:
    * Handles the serialization and deserialization of data for network transmission. It compresses data (using `zlib`) before sending and decompresses upon reception to optimize communication overhead.
    * Provides methods for sending and receiving data chunks over a socket, ensuring robust and efficient data transfer.
* **`data_manager.py`**:
    * Manages the local dataset for the node (e.g., MNIST training data).
    * Determines the number of training samples assigned to this node based on the total number of nodes.
    * Provides methods to retrieve specific batches of samples corresponding to indices sent by the orchestrator (part of the virtual batch mechanism).
* **`main.py`**:
    * The entry point for starting a node instance.
    * Parses command-line arguments for node ID, orchestrator host/port, and device selection.
    * Initializes and runs the `Node` class.
* **`model.py`**:
    * Defines the neural network architecture (`Net`) that resides on each node.
    * Crucially, it is designed to store intermediate activations during its forward pass, which are then extracted and sent to the orchestrator.
    * This model is designed to support the specialized gradient computations needed by the `task_processor` to support the orchestrator's centralized BP.
* **`network_client.py`**:
    * Manages the TCP socket connection between the node and the orchestrator.
    * Handles connection attempts, reconnections, and the initial handshake (sending `node_id`).
    * Utilizes the `data_handler` for sending and receiving structured data.
    * Implements TCP Keepalive options for robust, long-lived connections.
* **`node.py`**:
    * The core class encapsulating the overall behavior of a TL node.
    * Manages the main loop for connecting to the orchestrator and processing incoming tasks.
    * Delegates specific computations to the `task_processor` and handles sending results back via the `network_client`.
    * Includes robust error handling and graceful shutdown mechanisms.
* **`task_processer.py`**:
    * Performs the actual computational tasks assigned by the orchestrator.
    * Loads model parameters received from the orchestrator into the local `Net` model.
    * Executes the forward pass on the assigned data batch.
    * Calculates and prepares the necessary activations (`act_fc2`) and gradients (`grad_z4`, `grad_fc2`, `grad_fc1`) to be sent back to the orchestrator for centralized backward propagation.

## How to Run a Node

Refer to the main `README.md` in the root directory for detailed instructions on setting up and running the entire Traversal Learning system, including the nodes.

```bash
python node/main.py --node_id <node_id> --host <orchestrator_host> --port <orchestrator_port> --num_nodes <total_num_nodes> [--no-accel]
