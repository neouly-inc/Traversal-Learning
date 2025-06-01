import time
import socket
import logging
import select
from typing import Any, Dict, Set

from data_handler import DataHandler
from connection_manager import ConnectionManager
from connection import Connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TaskScheduler:
    """
    Responsible for orchestrating tasks within a single epoch,
    sending tasks to nodes, and collecting their responses.
    """
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.running = True # Should align with the overall system's running state

    def set_running_state(self, running: bool) -> None:
        """Allows external control of the scheduler's running state."""
        self.running = running
        logging.debug(f"TaskScheduler: Running state set to {self.running}.")

    def _send_tasks_to_nodes(self, connections_for_epoch: Dict[int, Connection], task: str, task_data: Any) -> Set[int]:
        """
        Sends tasks to a dictionary of active nodes and returns a set of node_ids
        that failed during the send operation.
        """
        nodes_failed_send = set()
        
        if not connections_for_epoch:
            logging.warning(f"TaskScheduler: No active connections to send tasks.")
            return set()

        logging.info(f"TaskScheduler: Attempting to send tasks to {len(connections_for_epoch)} nodes.")

        for node_id, conn_obj in connections_for_epoch.items():
            if not self.running:
                logging.info("TaskScheduler: Aborting task sending due to shutdown signal.")
                break # Exit early if scheduler is shutting down

            if not conn_obj.active: # Double-check active status
                logging.warning(f"TaskScheduler: Node {node_id} connection became inactive before sending task. Skipping.")
                nodes_failed_send.add(node_id)
                self.connection_manager.mark_node_disconnected(node_id) # Ensure state is updated
                continue

            try:
                message_to_node = {
                    "task": task,
                }
                if task == "forward_pass":
                    message_to_node["indices"] = task_data["batch"][node_id]
                    message_to_node["model_parameters"] = task_data["model_parameters"]
                elif task == "backward_second":
                    message_to_node["grad_z2"] = task_data["grad_z2"]
                    message_to_node["padding_left"] = task_data["padding_left"][node_id]
                    message_to_node["padding_right"] = task_data["padding_right"][node_id]
                elif task == "backward_first":
                    message_to_node["grad_z1"] = task_data["grad_z1"]
                
                # Set a small timeout for sending data to avoid indefinite blocking
                conn_obj.sock.settimeout(5.0)
                DataHandler.send_data(conn_obj.sock, message_to_node)
                logging.info(f"TaskScheduler: Sent task to Node {node_id}.")
            except (ConnectionError, socket.timeout, ValueError) as e:
                logging.error(f"TaskScheduler: Communication error with Node {node_id} during task send: {e}. Marking as disconnected.")
                nodes_failed_send.add(node_id)
                self.connection_manager.mark_node_disconnected(node_id)
            except Exception as e:
                logging.critical(f"TaskScheduler: Unexpected critical error sending task to Node {node_id}: {e}. Marking as disconnected.", exc_info=True)
                nodes_failed_send.add(node_id)
                self.connection_manager.mark_node_disconnected(node_id)
        return nodes_failed_send

    def _receive_responses_from_nodes(
        self,
        nodes_to_await_response: Set[int],
        connections_for_epoch: Dict[int, Connection],
        response_collection_timeout: float
    ) -> Dict[int, Any]:
        """
        Collects responses from a set of nodes within a given timeout.
        Returns a dictionary of received responses (node_id -> response_data).
        """
        epoch_responses: Dict[int, Any] = {}
        start_response_collection = time.time()

        if not nodes_to_await_response:
            logging.info(f"TaskScheduler: No nodes to await responses from.")
            return {}
        
        logging.info(f"TaskScheduler: Awaiting responses from {len(nodes_to_await_response)} nodes.")

        # Use a mutable copy that can be modified as responses are received or nodes fail
        remaining_nodes_to_await = set(nodes_to_await_response)
        
        while self.running and remaining_nodes_to_await and (time.time() - start_response_collection < response_collection_timeout):
            timeout_remaining = response_collection_timeout - (time.time() - start_response_collection)
            if timeout_remaining <= 0:
                logging.warning(f"TaskScheduler: Response collection timeout ({response_collection_timeout:.1f}s) reached.")
                break

            # Filter sockets to monitor based on remaining_nodes_to_await and active status
            current_sockets_to_monitor = [conn.sock for node_id, conn in connections_for_epoch.items()
                                          if node_id in remaining_nodes_to_await and conn.active]
            
            if not current_sockets_to_monitor:
                logging.debug("TaskScheduler: No active sockets left to monitor for responses.")
                break # All nodes responded, or became inactive/disconnected

            try:
                # Use select with a small dynamic timeout to allow checking self.running and overall timeout
                # Max 1.0s wait to be responsive to shutdown or overall timeout
                readable_sockets, _, _ = select.select(current_sockets_to_monitor, [], [], min(timeout_remaining, 1.0))

                for sock in readable_sockets:
                    if not self.running: break # Exit early if shutdown
                    
                    # Find the node_id corresponding to the socket
                    responding_node_id = None
                    for node_id, conn_obj in connections_for_epoch.items():
                        if conn_obj.sock == sock and conn_obj.active:
                            responding_node_id = node_id
                            break

                    if responding_node_id is None:
                        logging.warning(f"TaskScheduler: Received data from an unknown/inactive socket {sock.getpeername() if sock.getpeername() else 'unknown'}. Ignoring.")
                        continue # Should not happen often if logic is correct, but for safety

                    try:
                        response = DataHandler.receive_data(sock)

                        if response is None:
                            # DataHandler returns None if peer closed gracefully on header read
                            logging.info(f"TaskScheduler: Node {responding_node_id} ({sock.getpeername()}) closed connection gracefully. Marking as disconnected.")
                            self.connection_manager.mark_node_disconnected(responding_node_id)
                            remaining_nodes_to_await.discard(responding_node_id)
                            continue

                        if isinstance(response, dict) and response.get("command") == "disconnecting":
                            logging.info(f"TaskScheduler: Node {responding_node_id} reported graceful disconnection. Marking as disconnected.")
                            self.connection_manager.mark_node_disconnected(responding_node_id)
                            remaining_nodes_to_await.discard(responding_node_id)
                            continue
                        
                        logging.info(f"TaskScheduler: Received response from Node {responding_node_id}.")
                        epoch_responses[responding_node_id] = response
                        remaining_nodes_to_await.discard(responding_node_id) # Mark as received
                    except socket.timeout:
                        logging.debug(f"TaskScheduler: Node {responding_node_id} socket timed out during receive. Will re-monitor.")
                    except (ConnectionError, ValueError) as e: # Catch ConnectionError and ValueError from DataHandler
                        logging.error(f"TaskScheduler: Communication/data error (receiving) with Node {responding_node_id}: {e}. Marking as disconnected.")
                        self.connection_manager.mark_node_disconnected(responding_node_id)
                        remaining_nodes_to_await.discard(responding_node_id)
                    except Exception as e:
                        logging.critical(f"TaskScheduler: Unexpected error receiving from Node {responding_node_id}: {e}. Marking as disconnected.", exc_info=True)
                        self.connection_manager.mark_node_disconnected(responding_node_id)
                        remaining_nodes_to_await.discard(responding_node_id)

            except select.error as e:
                logging.error(f"TaskScheduler: Select error while monitoring sockets: {e}. Attempting to recover by re-validating sockets.")
                # This could happen if a socket in sockets_to_monitor becomes invalid
                invalid_sockets = []
                for sock in current_sockets_to_monitor:
                    try:
                        # Attempt a non-blocking peek to check socket validity. Can also try sock.fileno() to check if descriptor is valid.
                        # Using a dummy send can also reveal broken pipes quickly.
                        sock.send(b'', socket.MSG_DONTWAIT | socket.MSG_PEEK)
                    except (socket.error, OSError):
                        invalid_sockets.append(sock)
                
                if invalid_sockets:
                    logging.warning(f"TaskScheduler: Detected {len(invalid_sockets)} invalid sockets in select. Removing them.")
                    for invalid_sock in invalid_sockets:
                        for node_id, conn_obj in connections_for_epoch.items():
                            if conn_obj.sock == invalid_sock and conn_obj.active:
                                logging.info(f"TaskScheduler: Forcing disconnect for Node {node_id} due to invalid socket during select.")
                                self.connection_manager.mark_node_disconnected(node_id)
                                remaining_nodes_to_await.discard(node_id)
                                break
                
            except Exception as e:
                logging.critical(f"TaskScheduler: Critical error in response collection loop: {e}. Aborting epoch.", exc_info=True)
                break # Abort collection for this epoch

        # After loop, mark any remaining nodes as disconnected if they haven't responded
        if remaining_nodes_to_await:
            logging.warning(f"TaskScheduler: {len(remaining_nodes_to_await)} nodes did not respond within timeout. Marking as disconnected.")
            for node_id in list(remaining_nodes_to_await): # Iterate over a copy
                self.connection_manager.mark_node_disconnected(node_id)
                # No need to discard from `remaining_nodes_to_await` here as loop is exiting,
                # but if loop continued, it would be essential.

        return epoch_responses

    def orchestrate_tasks(self, num_expected_nodes: int, task: str, task_data: Any) -> Dict[int, Any]:
        """
        Main method to distribute tasks across the orchestrator's connected nodes.
        Returns the collected responses (node_id -> response_data).

        Args:
        - num_expected_nodes (int): Number of expected connected nodes.
        - task_id (int): ID of the task to be orchestrated.
            retrieve_num_samples    : Retrieve number of samples
            forward_pass            : Forward pass
            compute_2nd_grads       : Compute gradients for 2nd layer
            compute_1st_grads       : Compute gradients for 1st layer
        Returns:
        - Dict[int, Any]: Dictionary mapping node_id -> response_data.
        """
        if not self.running:
            logging.info("TaskScheduler: Not running, skipping epoch orchestration.")
            return {}

        connections_at_task_start = self.connection_manager.get_active_connections()

        if len(connections_at_task_start) != num_expected_nodes:
            logging.error(f"TaskScheduler: Mismatch in active connections ({len(connections_at_task_start)} vs {num_expected_nodes}). "
                          "Cannot orchestrate. This implies not all nodes were ready.")
            return {} # Indicate failure to orchestrate due to insufficient nodes

        logging.info(f"TaskScheduler: Starting \"{task}\" task distribution.")
        nodes_failed_send = self._send_tasks_to_nodes(connections_at_task_start, task, task_data)

        if not self.running:
            logging.info("TaskScheduler: Aborting response collection due to shutdown signal after sending tasks.")
            return {}

        # Re-fetch active connections after sending tasks to account for any immediate failures
        # This is important because `_send_tasks_to_nodes` might have called `mark_node_disconnected`.
        current_active_connections_for_receive = self.connection_manager.get_active_connections()

        # Only await responses from nodes that:
        # 1. Successfully received the task (not in nodes_failed_send)
        # 2. Are still considered active now (in current_active_connections_for_receive)
        nodes_to_await_response = (set(connections_at_task_start.keys()) - nodes_failed_send).intersection(set(current_active_connections_for_receive.keys()))

        task_responses = self._receive_responses_from_nodes(
            nodes_to_await_response,
            current_active_connections_for_receive, # Use the most up-to-date active connections for receive
            response_collection_timeout=120.0
        )

        return task_responses