import socket
import logging
import threading
import time
from typing import Optional, Dict, Set, Callable

from data_handler import DataHandler
from connection import Connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')


class ConnectionManager:
    """
    Manages all network connections, accepts new clients, and tracks their status.
    Notifies a callback when node registration status changes.
    """
    def __init__(self, host: str, port: int, num_expected_nodes: int,
                 on_nodes_status_change: Optional[Callable[[], None]] = None) -> None:
        self.host = host
        self.port = port
        self.num_expected_nodes = num_expected_nodes
        self.on_nodes_status_change = on_nodes_status_change

        # Dictionary to store active connections, keyed by node_id.
        # This will hold the `Connection` objects.
        self.connections: Dict[int, Connection] = {}
        self.connections_lock = threading.Lock() # Protects self.connections
        
        # Event for external components (like Orchestrator) to wait on
        self.all_nodes_registered_event = threading.Event()
        self.running = True # Controls this manager's internal loops

        self.orchestrator_socket: Optional[socket.socket] = None
        self.expected_node_ids: Set[int] = set(range(1, num_expected_nodes + 1))
        
        # Keep track of node IDs that are expected but currently disconnected
        self.currently_disconnected_node_ids: Set[int] = set()

    def start(self) -> None:
        """Starts the connection manager, setting up socket and acceptor thread."""
        self._setup_socket()
        self._start_connection_acceptor()

    def stop(self) -> None:
        """Stops the connection manager, closing sockets and threads."""
        logging.info("ConnectionManager: Initiating graceful shutdown.")
        self.running = False
        self.all_nodes_registered_event.set() # Release any threads waiting on this event

        # Close listening socket first to stop new incoming connections
        if self.orchestrator_socket:
            try:
                # Set a small timeout to unblock accept() if it's currently blocking
                self.orchestrator_socket.settimeout(0.1)
                self.orchestrator_socket.close()
                self.orchestrator_socket = None
                logging.info("ConnectionManager: Orchestrator listening socket closed.")
            except OSError as e:
                logging.debug(f"ConnectionManager: Error closing orchestrator socket: {e}")
            except Exception as e:
                logging.error(f"ConnectionManager: Unexpected error closing orchestrator socket: {e}")

        # Send shutdown to active nodes and close their connections
        nodes_to_shutdown_info = []
        with self.connections_lock:
            # Create a list of tuples (node_id, connection_object) to iterate safely
            nodes_to_shutdown_info = [(node_id, conn_obj) for node_id, conn_obj in self.connections.items() if conn_obj.active]
            # Clear connections dictionary after capturing active ones, to prevent new operations on them
            self.connections.clear() # This makes get_active_connections() return empty during shutdown

        for node_id, conn_obj in nodes_to_shutdown_info:
            logging.info(f"ConnectionManager: Attempting to send shutdown command to Node {node_id} ({conn_obj.addr})...")
            try:
                # Attempt to send shutdown command to nodes before closing their sockets
                # Set a small timeout for sending the shutdown message to prevent blocking indefinitely
                conn_obj.sock.settimeout(1.0)
                DataHandler.send_data(conn_obj.sock, {"command": "shutdown", "message": "Orchestrator shutting down."})
                logging.info(f"ConnectionManager: Successfully sent shutdown to Node {node_id}.")
                time.sleep(0.05) # Give a tiny moment for the message to send
            except (socket.error, ConnectionResetError, BrokenPipeError, socket.timeout) as e:
                logging.debug(f"ConnectionManager: Communication error sending shutdown to Node {node_id} ({conn_obj.addr}): {e} (node might already be down or connection broken).")
            except Exception as e:
                logging.error(f"ConnectionManager: Unexpected error sending shutdown to Node {node_id} ({conn_obj.addr}): {e}")
            finally:
                conn_obj.close() # Ensure socket is closed regardless of send success

        logging.info("ConnectionManager: All node connections processed for shutdown.")
        logging.info("ConnectionManager: Shutdown complete.")


    def _setup_socket(self) -> None:
        """Initializes and binds the orchestrator's listening socket."""
        try:
            self.orchestrator_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # SO_REUSEADDR allows binding to a port that was recently in use
            self.orchestrator_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Apply KEEPALIVE to the listening socket itself (less critical but good practice)
            # Note: Keepalive on listening sockets mainly applies to accepted sockets inheriting properties
            self.orchestrator_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            self.orchestrator_socket.bind((self.host, self.port))
            # Listen for up to `num_expected_nodes * 2` incoming connections in the backlog.
            # This allows for reconnections and initial connections concurrently.
            self.orchestrator_socket.listen(self.num_expected_nodes * 2)
            logging.info(f"ConnectionManager: Listening on {self.host}:{self.port}, expecting {self.num_expected_nodes} nodes.")
        except socket.error as e:
            logging.critical(f"ConnectionManager: Failed to set up socket on {self.host}:{self.port}: {e}")
            self.running = False
            raise # Re-raise to halt orchestrator initialization

    def _start_connection_acceptor(self) -> None:
        """Starts a daemon thread to continuously accept incoming connections."""
        # Daemon thread means it will terminate automatically when main program exits
        accept_thread = threading.Thread(target=self._accept_connections_loop, name="AcceptorThread", daemon=True)
        accept_thread.start()
        logging.info("ConnectionManager: Connection acceptor thread started.")

    def _accept_connections_loop(self) -> None:
        """Loop to accept new client connections."""
        if not self.orchestrator_socket:
            logging.error("ConnectionManager: Acceptor loop started without a valid socket.")
            self.running = False
            return

        # Set a small timeout on the listening socket to allow `self.running` to be checked
        self.orchestrator_socket.settimeout(1.0)
        while self.running:
            try:
                conn_sock, addr = self.orchestrator_socket.accept()
                logging.info(f"ConnectionManager: Incoming connection from {addr}")
                # Spawn a new daemon thread to handle the handshake and initial communication
                threading.Thread(target=self._handle_new_connection, args=(conn_sock, addr),
                                 name=f"Handler-{addr[0]}:{addr[1]}", daemon=True).start()
            except socket.timeout:
                # Expected when no connections come in within the timeout, allows loop to re-check self.running
                continue
            except OSError as e:
                # If the socket is closed during accept (e.e., during shutdown)
                if not self.running:
                    logging.info(f"ConnectionManager: Acceptor loop socket error during shutdown: {e}")
                else:
                    logging.error(f"ConnectionManager: OSError accepting connection: {e}")
                break # Exit loop on unexpected OSError
            except Exception as e:
                logging.error(f"ConnectionManager: Unexpected error accepting connection: {e}", exc_info=True)
        logging.info("ConnectionManager: Accept connections loop finished.")

    def _handle_new_connection(self, conn_sock: socket.socket, addr: tuple) -> None:
        """Handles initial handshake for a new connection and registers the node."""
        node_id = None
        try:
            conn_sock.settimeout(10.0) # Timeout for initial handshake data reception
            initial_data = DataHandler.receive_data(conn_sock)

            if not initial_data: # DataHandler returns None if peer closed gracefully on header read
                logging.warning(f"ConnectionManager: Peer {addr} closed connection during initial handshake. Ignoring.")
                conn_sock.close()
                return
            if not isinstance(initial_data, dict) or "node_id" not in initial_data:
                logging.warning(f"ConnectionManager: Invalid or incomplete initial data from {addr}. Closing connection.")
                DataHandler.send_data(conn_sock, {"command": "reject", "message": "Invalid initial data (missing node_id)."})
                conn_sock.close()
                return

            node_id = initial_data["node_id"]
            if not isinstance(node_id, int) or node_id not in self.expected_node_ids:
                logging.warning(f"ConnectionManager: Node ID {node_id} from {addr} is not an expected node ID. Rejecting.")
                DataHandler.send_data(conn_sock, {"command": "reject", "message": "Invalid node ID."})
                conn_sock.close()
                return

            with self.connections_lock:
                # Handle reconnection scenario: if old connection for this node_id is still active, close it
                if node_id in self.connections and self.connections[node_id].active:
                    old_conn = self.connections[node_id]
                    logging.warning(f"ConnectionManager: Node {node_id} already has an active connection from {old_conn.addr}. Closing old connection.")
                    old_conn.close() # Ensure old socket is closed and marked inactive

                new_connection = Connection(conn_sock, addr) # Create new Connection object
                new_connection.node_id = node_id
                self.connections[node_id] = new_connection # Register the new connection

                # If this node was previously disconnected, remove it from the disconnected list
                if node_id in self.currently_disconnected_node_ids:
                    self.currently_disconnected_node_ids.remove(node_id)
                    logging.info(f"ConnectionManager: Node {node_id} reconnected, removed from disconnected list. Currently disconnected: {len(self.currently_disconnected_node_ids)} nodes.")

                logging.info(f"ConnectionManager: Node {node_id} at {addr} successfully registered/re-registered.")
                DataHandler.send_data(conn_sock, {"message": f"Welcome, Node {node_id}!"})
                conn_sock.settimeout(None) # Reset to blocking after handshake

                self._update_nodes_status() # Update registration status and notify orchestrator

        except (socket.timeout, ConnectionError, ValueError) as e:
            logging.error(f"ConnectionManager: Handshake error with {addr} (Node ID: {node_id if node_id else 'unknown'}): {e}. Closing connection.")
            if conn_sock:
                conn_sock.close()
        except Exception as e:
            logging.error(f"ConnectionManager: Unexpected error handling new connection from {addr} (Node ID: {node_id if node_id else 'unknown'}): {e}. Closing connection.", exc_info=True)
            if conn_sock:
                conn_sock.close()

    def _update_nodes_status(self) -> None:
        """
        Internal method to update the node registration event and notify callback.
        This method is designed to be called under `self.connections_lock`.
        """
        active_connection_count = sum(1 for conn in self.connections.values() if conn.active)
        logging.info(f"ConnectionManager: Current active nodes: {active_connection_count}/{self.num_expected_nodes}. Disconnected tracked: {len(self.currently_disconnected_node_ids)}")

        if active_connection_count == self.num_expected_nodes:
            if not self.all_nodes_registered_event.is_set():
                logging.info(f"ConnectionManager: All {self.num_expected_nodes} nodes are now active. Signaling 'all_nodes_registered'.")
                self.all_nodes_registered_event.set()
        else:
            if self.all_nodes_registered_event.is_set():
                logging.warning(f"ConnectionManager: Active nodes dropped to {active_connection_count}. Clearing 'all_nodes_registered' event.")
                self.all_nodes_registered_event.clear()

        if self.on_nodes_status_change:
            # Call the callback only if there's a listener
            self.on_nodes_status_change()

    def wait_for_all_nodes(self, timeout: Optional[float] = None) -> bool:
        """
        Public method to wait for all required nodes to be connected and active.
        Returns True if all nodes become active within the timeout (or indefinitely if timeout is None).
        Returns False if the manager is shutting down, or if the timeout is reached.
        """
        if timeout is None:
            logging.info(f"ConnectionManager: Waiting indefinitely for all {self.num_expected_nodes} nodes to be active...")
        else:
            logging.info(f"ConnectionManager: Waiting up to {timeout:.1f} seconds for all {self.num_expected_nodes} nodes to be active...")

        start_time = time.time()
        while self.running: # Continue waiting as long as the manager is running
            with self.connections_lock:
                current_active_nodes = sum(1 for conn in self.connections.values() if conn.active)

            if current_active_nodes == self.num_expected_nodes:
                if not self.all_nodes_registered_event.is_set():
                    self.all_nodes_registered_event.set() # Ensure event is set
                logging.info(f"ConnectionManager: All {self.num_expected_nodes} nodes are active. Continuing.")
                return True
            
            # Calculate remaining wait time
            effective_wait_timeout = timeout - (time.time() - start_time) if timeout is not None else None
            if effective_wait_timeout is not None and effective_wait_timeout <= 0:
                logging.warning(f"ConnectionManager: Timeout ({timeout:.1f}s) reached. Not all nodes connected/reconnected.")
                return False

            logging.info(f"ConnectionManager: Currently {current_active_nodes} active nodes. Waiting for {self.num_expected_nodes - current_active_nodes} more.")

            # Wait for the event to be set, or for a small interval to re-check `self.running` and overall timeout.
            # Use min with a small fixed value (e.g., 1.0s) to prevent waiting too long if `effective_wait_timeout` is large.
            wait_duration_for_event = min(effective_wait_timeout or float('inf'), 5.0)
            event_set_status = self.all_nodes_registered_event.wait(timeout=wait_duration_for_event)

            if event_set_status: # Event was set, meaning all nodes are active (or were briefly)
                with self.connections_lock:
                    # Double-check condition after event is set, as state might have changed quickly
                    if sum(1 for conn in self.connections.values() if conn.active) == self.num_expected_nodes:
                        return True
                    else:
                        logging.warning("ConnectionManager: Event was set but nodes count changed. Re-evaluating and clearing event.")
                        self.all_nodes_registered_event.clear() # Clear and re-wait if condition is false
            elif not self.running: # If event timed out and manager is no longer running
                logging.info("ConnectionManager: Shutting down during wait for all nodes.")
                return False

        logging.info("ConnectionManager: Shutting down during wait for all nodes (loop exit).")
        return False


    def get_active_connections(self) -> Dict[int, Connection]:
        """Returns a snapshot of currently active connections, thread-safe."""
        with self.connections_lock:
            # Only return truly active connections.
            # This makes sure that even if connection object exists, if its active flag is false, it's excluded.
            return {node_id: conn for node_id, conn in self.connections.items() if conn.active}

    def mark_node_disconnected(self, node_id: int) -> None:
        """
        Marks a specific node as disconnected due to a communication failure
        and updates the connection status.
        """
        with self.connections_lock:
            if node_id in self.connections:
                conn = self.connections[node_id]
                if conn.active:
                    logging.info(f"ConnectionManager: Marking Node {node_id} ({conn.addr}) as disconnected due to communication failure.")
                    conn.close() # This sets conn.active = False and closes the socket
                else:
                    logging.debug(f"ConnectionManager: Node {node_id} was already inactive or being closed.")
            else:
                logging.debug(f"ConnectionManager: Attempted to mark non-existent/unregistered Node {node_id} as disconnected.")

            # Add to currently disconnected list regardless, for tracking.
            # This is important for the `wait_for_all_nodes` logic, even if node was never fully registered.
            if node_id in self.expected_node_ids: # Only track if it's an expected node
                self.currently_disconnected_node_ids.add(node_id)
                logging.info(f"ConnectionManager: Node {node_id} added to disconnected tracking list. Current count: {len(self.currently_disconnected_node_ids)}")
            
            self._update_nodes_status() # Always update status when a node state changes