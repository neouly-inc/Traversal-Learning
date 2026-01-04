import socket
import logging
import threading
import torch

from network_client import NetworkClient
from task_processor import TaskProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')


class Node:
    """
    Represents a single node in the distributed learning setup.
    Orchestrates the NetworkClient and TaskProcessor.
    """
    RECONNECT_RETRY_DELAY_NODE = 5 # Delay in Node's main loop before attempting a full reconnect cycle after a connection failure

    def __init__(self, node_id: int, orchestrator_host: str, orchestrator_port: int, num_nodes: int, device: torch.device) -> None:
        if node_id <= 0:
            raise ValueError("Node ID must be a positive integer.")
        self.node_id = node_id
        self.orchestrator_host = orchestrator_host
        self.orchestrator_port = orchestrator_port

        self._shutdown_event = threading.Event() # Event to signal Node and NetworkClient to exit gracefully
        self.network_client = NetworkClient(orchestrator_host, orchestrator_port, node_id, self._shutdown_event)
        self.task_processor = TaskProcessor(node_id, num_nodes, device)
        self.running = True # Controls the main operational loop of the node

    def signal_shutdown(self) -> None:
        """Sets the internal flag and event to signal the main loop and NetworkClient to terminate."""
        logging.info(f"Node {self.node_id}: Received signal_shutdown request. Initiating graceful exit.")
        self.running = False # Stop the main loop
        self._shutdown_event.set() # Unblock any blocking operations (like `wait()` or `select`)

    def run(self) -> None:
        """
        Main entry point for the node's operation.
        Manages connection state and handles incoming tasks from the orchestrator.
        """
        logging.info(f"Node {self.node_id}: Starting node operation.")
        self.running = True # Ensure this is True at start of run
        self._shutdown_event.clear() # Clear event at start of run, in case of re-run

        while self.running:
            # Always check for shutdown at the top of the loop for immediate reaction
            if self._shutdown_event.is_set():
                logging.info(f"Node {self.node_id}: Shutdown event set. Exiting main loop.")
                break

            # 1. Connection Management: Ensure connected before proceeding to task handling
            if not self.network_client.connected:
                logging.info(f"Node {self.node_id}: Not connected. Initiating connection process to orchestrator immediately...")
                initial_handshake_data = {"node_id": self.node_id}

                # network_client.connect() handles its own internal retries and respects _shutdown_event
                # It returns False if it exhausted retries OR if shutdown was signaled OR connection rejected
                connection_successful = self.network_client.connect(initial_handshake_data)

                if not connection_successful:
                    # Connection failed, was interrupted by shutdown, or explicitly rejected by orchestrator
                    if self._shutdown_event.is_set():
                        logging.info(f"Node {self.node_id}: Shutdown event set during connection failure. Exiting.")
                        break
                    # If here, connection failed for other reasons (e.g., server not up, invalid ID, transient issue)
                    logging.error(f"Node {self.node_id}: Failed to establish connection after multiple attempts. Will re-attempt connection process in {self.RECONNECT_RETRY_DELAY_NODE} seconds.")
                    
                    # Use _shutdown_event.wait() to make this sleep interruptible by shutdown signal
                    if self._shutdown_event.wait(self.RECONNECT_RETRY_DELAY_NODE):
                        logging.info(f"Node {self.node_id}: Shutdown event set during reconnect wait. Exiting.")
                        break # Exit if event was set while waiting
                    
                    continue # Continue to the next loop iteration to re-attempt connection

            # Check if shutdown was requested immediately after connection attempt (successful or not)
            if self._shutdown_event.is_set():
                logging.info(f"Node {self.node_id}: Shutdown event set after connection logic. Exiting main loop.")
                break

            # 2. Communication and Task Processing (only if connected)
            if self.network_client.connected:
                try:
                    # Use a timeout for receiving data to allow for periodic checks of self.running
                    # This prevents blocking indefinitely if orchestrator is silent but alive
                    receive_timeout = 300
                    received_data = self.network_client.receive(timeout=receive_timeout)

                    if received_data is None:
                        # DataHandler returns None if peer (orchestrator) closed connection gracefully (EOF on header read)
                        logging.info(f"Node {self.node_id}: Orchestrator closed connection gracefully. Initiating reconnect.")
                        # The network_client.receive already handled closing its socket if it returned None
                        # continue # Go back to the top of the loop to trigger reconnect
                        break # Exit the main loop

                    if isinstance(received_data, dict) and received_data.get("command") == "shutdown":
                        logging.info(f"Node {self.node_id}: Received shutdown command from orchestrator: '{received_data.get('message', '')}'. Shutting down gracefully.")
                        self.signal_shutdown()
                        break # Exit the main loop

                    if isinstance(received_data, dict) and "task" in received_data:
                        task = received_data["task"]
                        logging.info(f"Node {self.node_id}: Received \"{task}\" task.")
                        if task == "retrieve_num_samples":
                            processed_result = self.task_processor.get_num_samples()
                        elif task == "forward_pass":
                            processed_result = self.task_processor.forward_pass(received_data)
                        elif task == "backward_second":
                            processed_result = self.task_processor.backward_second(received_data)
                        elif task == "backward_first":
                            processed_result = self.task_processor.backward_first(received_data)

                        # Re-check connection status and running flag before sending results
                        if self.network_client.connected and self.running and not self._shutdown_event.is_set():
                            self.network_client.send(processed_result)
                            logging.info(f"Node {self.node_id}: Sent results for \"{task}\" task.")
                        else:
                            logging.warning(f"Node {self.node_id}: Connection lost or shutdown signaled during task processing/before sending results. Skipping result send.")
                            self.signal_shutdown() # Force node shutdown if critical state
                            break
                    else:
                        logging.warning(f"Node {self.node_id}: Received unexpected data format from orchestrator: {received_data}. Ignoring.")

                except socket.timeout:
                    logging.debug(f"Node {self.node_id}: Socket receive timed out. Awaiting next instruction from orchestrator...")
                    # If timeout, immediately check shutdown event
                    if self._shutdown_event.is_set():
                        logging.info(f"Node {self.node_id}: Shutdown event set during socket timeout. Exiting.")
                        break
                    # No data for now, just continue the loop to check connection status and wait again
                    continue

                except ConnectionError as e: # Catches errors raised by DataHandler for connection loss
                    logging.error(f"Node {self.node_id}: Connection lost with orchestrator: {e}. Attempting reconnect...")
                    # If connection lost, immediately check shutdown event
                    if self._shutdown_event.is_set():
                        logging.info(f"Node {self.node_id}: Shutdown event set after connection error. Exiting.")
                        break
                    # If not shutdown, the loop will naturally go back to connection management for reconnect
                    continue
                except ValueError as e: # Catches JSONDecodeError, struct.error from DataHandler
                    logging.error(f"Node {self.node_id}: Data format error from orchestrator: {e}. Cannot process. Shutting down.", exc_info=True)
                    self.signal_shutdown()
                    break
                except Exception as e:
                    logging.critical(f"Node {self.node_id}: An unexpected critical error occurred during communication/task processing: {e}", exc_info=True)
                    self.signal_shutdown()
                    break
            else:
                # This branch is hit if `self.network_client.connected` is False,
                # but the loop continued (e.g., after an internal `_close_socket` call
                # due to an error, but not an explicit `ConnectionError` raised).
                # Wait briefly before re-attempting connection, or exiting on shutdown.
                logging.debug(f"Node {self.node_id}: Node is not connected, waiting for next connection attempt.")
                if self._shutdown_event.wait(1): # Wait for 1 second or until shutdown event is set
                    logging.info(f"Node {self.node_id}: Shutdown event set during not-connected wait. Exiting.")
                    break

        logging.info(f"Node {self.node_id}: Main run loop terminated.")
        self.network_client.disconnect() # Ensure client's socket is properly shut down on exit
