import threading
import logging
import time
import numpy as np
import torch
from typing import Any, Dict

from connection_manager import ConnectionManager
from task_scheduler import TaskScheduler
from model_manager import ModelManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')


class Orchestrator:
    """
    Orchestrate the overall distributed training process, managing epochs
    and interacting with ConnectionManager and TaskScheduler.
    """
    def __init__(self, host: str, port: int, num_nodes: int, epochs: int, batch_size: int, lr: float, gamma: float, device: torch.device) -> None:
        if num_nodes <= 0:
            raise ValueError("Number of expected nodes must be positive.")
        if epochs <= 0:
            raise ValueError("Number of epochs must be positive.")

        self.num_nodes = num_nodes
        self.epochs = epochs
        self.current_epoch = 0 # Epochs are 1-indexed for display
        self.batch_size = batch_size
        self.current_batch = 0 # Batches are 1-indexed for display
        self.virtual_batches = []
        self.running = True
        self._shutdown_event = threading.Event() # Central shutdown event for the orchestrator

        # Initialize sub-components
        self.connection_manager = ConnectionManager(
            host=host, port=port, num_expected_nodes=num_nodes,
            on_nodes_status_change=self._on_nodes_status_change # Callback for status updates
        )
        self.task_scheduler = TaskScheduler(
            connection_manager=self.connection_manager
        )
        self.model_manager = ModelManager(lr, gamma, device) # Model initialization

    def _on_nodes_status_change(self) -> None:
        """Callback triggered by ConnectionManager when node connection status changes."""
        # This callback can be used for logging or to trigger re-evaluation of training state.
        # The primary synchronization for full node availability is handled by `wait_for_all_nodes`.
        logging.debug("Orchestrator: Node connection status updated by ConnectionManager.")

    def run(self) -> None:
        """Main entry point to start the distributed training orchestration."""
        try:
            logging.info("Orchestrator: Starting orchestrator.")
            self.connection_manager.start()
            self._run_training()
        except Exception as e:
            logging.critical(f"Orchestrator: Critical error during execution: {e}", exc_info=True)
        finally:
            self.shutdown() # Ensure shutdown is always called

    def _create_virtual_batches(self, responses: Dict[int, Any]) -> None:
        """
        Creates a virtual batches using the number of samples per node.
        """
        node_ids = sorted(responses.keys())
        total_samples = sum(responses.values())

        # 1. Create a global mapping of original (node_id, local_index) to a global index.
        # This helps us track which sample came from where after shuffling.
        global_to_original_map = []
        accumulator = 0 # Only used for simulated testing
        for node_id in node_ids:
            for local_idx in range(accumulator, responses[node_id] + accumulator):
                global_to_original_map.append((node_id, local_idx))
            accumulator += responses[node_id]

        # 2. Create a global list of indices (0 to total_samples - 1)
        global_indices = np.arange(total_samples)

        # 3. Shuffle all samples globally
        np.random.seed(1)  # Set a seed for reproducibility
        np.random.shuffle(global_indices)

        current_offset = 0

        while current_offset < total_samples:
            batch = {}
            # Initialize an empty list for each node in the current batch
            for node_id in node_ids:
                batch[node_id] = []

            # Determine the indices for the current global batch
            batch_global_indices = sorted(global_indices[current_offset : current_offset + self.batch_size])

            # Distribute these global indices back to their original nodes
            for global_idx in batch_global_indices:
                original_node_id, original_local_idx = global_to_original_map[global_idx]
                batch[original_node_id].append(original_local_idx)

            self.virtual_batches.append(batch)

            current_offset += self.batch_size

    def _run_training(self) -> None:
        """
        Manages the main training loop, iterating through epochs.
        Ensures all required nodes are connected and respond for EACH epoch.
        """
        logging.info("Orchestrator: Waiting for all expected nodes to register before starting initial epochs...")
        # Indefinite wait for initial connection unless shutdown is signaled
        # Timeout is None so it waits as long as needed initially.
        if not self.connection_manager.wait_for_all_nodes(timeout=None):
            if not self.running: # Check if shutdown was requested during initial wait
                logging.warning("Orchestrator: Initial node connection aborted by shutdown signal. Cannot start epochs.")
            else:
                logging.error("Orchestrator: Initial node connection failed unexpectedly (e.g., orchestrator stopped unexpectedly). Aborting training.")
            self.running = False # Set flag to false if wait failed or was interrupted
            return

        logging.info(f"Orchestrator: All {self.num_nodes} nodes registered. Starting virtual batches creation...")

        # Create virtual batches by retrieving number of samples from each node
        task_responses = self.task_scheduler.orchestrate_tasks(self.num_nodes, "retrieve_num_samples", None)
        self._create_virtual_batches(task_responses)

        logging.info(f"Orchestrator: Starting training epochs...")

        while self.running and self.current_epoch < self.epochs:
            self.current_epoch += 1
            for batch_id, batch in enumerate(self.virtual_batches):
                self.current_batch = batch_id + 1
                
                logging.info(f"--- Orchestrator: Starting Batch {self.current_batch}/{len(self.virtual_batches)} in Epoch {self.current_epoch}/{self.epochs} ---")

                # At the start of EACH epoch, re-confirm all nodes are active.
                # If any node disconnected, the orchestrator will pause here and wait for reconnection.
                # This is the "stop all other nodes and wait for them to reconnect" mechanism.
                # No explicit "pause" message is sent to active nodes; instead, tasks are held back.
                reconnect_timeout_per_epoch = 120.0 # Allow 2 minutes for nodes to reconnect per epoch
                if not self.connection_manager.wait_for_all_nodes(timeout=reconnect_timeout_per_epoch):
                    if not self.running:
                        logging.info(f"Orchestrator: Shutting down during wait for nodes for Batch {self.current_batch} in Epoch {self.current_epoch}. Aborting.")
                    else:
                        logging.error(f"Orchestrator: Failed to get all nodes active for Batch {self.current_batch} in Epoch {self.current_epoch} within {reconnect_timeout_per_epoch}s. Aborting training.")
                    self.running = False # Halt training if nodes are not ready
                    break

                # If orchestrator was signalled to shutdown during the wait_for_all_nodes, exit
                if not self.running:
                    logging.info("Orchestrator: Shutdown signal received, ending epochs early.")
                    break

                # Set scheduler's running state to ensure it respects global shutdown
                self.task_scheduler.set_running_state(self.running)

                # Collecting forward pass outputs from nodes
                data = {"batch": batch, "model_parameters": self.model_manager.get_parameters()}
                task_responses = self.task_scheduler.orchestrate_tasks(self.num_nodes, "forward_pass", data)
                
                loss, grad_z2, padding_left, padding_right = self.model_manager.forward_pass(task_responses, batch)

                # Collecting 2nd layer gradients from nodes
                data = {"grad_z2": grad_z2, "padding_left": padding_left, "padding_right": padding_right}
                task_responses = self.task_scheduler.orchestrate_tasks(self.num_nodes, "backward_second", data)

                grad_z1 = self.model_manager.backward_second(task_responses, grad_z2)

                # Collecting 1st layer gradients from nodes
                data = {"grad_z1": grad_z1}
                task_responses = self.task_scheduler.orchestrate_tasks(self.num_nodes, "backward_first", data)

                self.model_manager.backward_first(task_responses, grad_z1)

                if len(task_responses) == self.num_nodes:
                    logging.info(f"Orchestrator: All {self.num_nodes} nodes responded successfully.")
                    # Placeholder for aggregation/model update logic here
                    # self._aggregate_results(list(task_responses.values()))
                else:
                    logging.error(f"Orchestrator: Epoch {self.current_epoch} completed with only {len(task_responses)}/{self.num_nodes} responses. This is an incomplete epoch.")
                    # Decision point: abort or continue with fewer nodes?
                    # For robust training, usually abort if a full consensus is needed.
                    self.running = False # Halt training due to incomplete epoch
                    break

                logging.info(f"--- Orchestrator: Epoch {self.current_epoch} | Batch {self.current_batch} | Loss: {loss:.6f} ---")
                if self.current_epoch < self.epochs:
                    # Small pause between epochs to allow for state synchronization or human observation
                    logging.debug("Orchestrator: Pausing briefly before next epoch.")
                    time.sleep(1) # This sleep is minimal and should not block shutdown significantly

                self.model_manager.update_parameters()
                
            logging.info(f"Orchestrator: Epoch {self.current_epoch} completed. Total batches processed: {self.current_batch}.")
            
            self.model_manager.test()
            self.model_manager.step_scheduler()

        logging.info(f"Orchestrator: Training loop finished. Total epochs processed: {self.current_epoch}.")
        self.running = False # Ensure flag is set to false after loop finishes naturally

    def shutdown(self) -> None:
        """Initiates a graceful shutdown of the entire orchestrator system."""
        if not self.running and self._shutdown_event.is_set():
            logging.info("Orchestrator: Orchestrator is already in shutdown process or shut down.")
            return

        logging.info("Orchestrator: Initiating system shutdown.")
        self.running = False # Signal primary loop to stop
        self._shutdown_event.set() # Signal any waiting threads
        self.task_scheduler.set_running_state(False) # Signal scheduler to stop its operations

        # Give a moment for threads to acknowledge shutdown flags and exit their loops
        # This is a heuristic; more complex systems might use thread.join() with timeouts.
        time.sleep(0.5)

        # ConnectionManager.stop() will attempt to send shutdown messages to nodes
        # and then close all sockets.
        self.connection_manager.stop()

        logging.info("Orchestrator: System shutdown complete.")