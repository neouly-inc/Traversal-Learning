import os
import time
import socket
import logging
import threading
import select
import sys
from typing import Optional, Any

from data_handler import DataHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')


class NetworkClient:
    """
    Manages the direct socket connection and data communication with a remote server (Orchestrator).
    Handles connection attempts, sending, and receiving data.
    """
    MAX_CONNECT_ATTEMPTS = 30 # Max internal attempts per single `connect()` call
    RETRY_DELAY_INTERNAL = 2 # Delay between internal connection attempts within `connect()`

    def __init__(self, target_host: str, target_port: int, client_id: int, shutdown_event: threading.Event):
        self.target_host = target_host
        self.target_port = target_port
        self.client_id = client_id
        self.sock: Optional[socket.socket] = None
        self.connected = False
        self._shutdown_event = shutdown_event # Reference to the Node's shutdown event

    def _close_socket(self) -> None:
        """Safely closes the internal socket connection and updates connected status."""
        if self.sock:
            try:
                # Attempt graceful shutdown before closing
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError as e:
                logging.debug(f"NetworkClient {self.client_id}: Error during socket shutdown: {e} (might already be closed or peer disconnected).")
            except Exception as e:
                logging.warning(f"NetworkClient {self.client_id}: Unexpected error during socket shutdown: {e}")
            finally:
                self.sock.close()
                self.sock = None # Important: nullify the socket reference
        if self.connected: # Only log disconnection if it was previously connected
            self.connected = False
            logging.info(f"NetworkClient {self.client_id}: Socket connection closed and marked as disconnected.")

    def connect(self, initial_handshake_data: Any) -> bool:
        """
        Attempts to establish a connection to the target host with its own internal retries.
        Uses non-blocking connect and select.select to allow interruption by shutdown_event.
        Returns True on successful connection and handshake, False on failure or shutdown.
        """
        if self._shutdown_event.is_set():
            logging.info(f"NetworkClient {self.client_id}: Shutdown signal received before connection attempt. Aborting.")
            return False

        for attempt in range(self.MAX_CONNECT_ATTEMPTS):
            if self._shutdown_event.is_set():
                logging.info(f"NetworkClient {self.client_id}: Shutdown signal received, aborting connection attempts.")
                self._close_socket()
                return False

            logging.info(f"NetworkClient {self.client_id}: Connect attempt {attempt + 1}/{self.MAX_CONNECT_ATTEMPTS} to {self.target_host}:{self.target_port}.")
            self._close_socket() # Ensure clean state for a new attempt

            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # Platform-specific TCP Keepalive options
                if sys.platform == 'linux':
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
                    logging.debug(f"NetworkClient {self.client_id}: Applied Linux TCP Keepalive.")
                elif sys.platform == 'darwin': # macOS
                    TCP_KEEPALIVE = getattr(socket, 'TCP_KEEPALIVE', 0x10)
                    self.sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPALIVE, 60)
                    logging.debug(f"NetworkClient {self.client_id}: Applied macOS TCP Keepalive.")
                elif sys.platform == 'win32':
                    logging.debug(f"NetworkClient {self.client_id}: Using default Windows TCP Keepalive.")

                # Set socket to non-blocking mode for connection attempt
                self.sock.setblocking(False)

                connect_in_progress = False
                try:
                    self.sock.connect((self.target_host, self.target_port))
                    # If connect succeeds immediately (e.g., localhost), BlockingIOError is not raised
                except BlockingIOError:
                    connect_in_progress = True # Connection is in progress
                except (ConnectionRefusedError, socket.timeout, socket.error) as e:
                    logging.warning(f"NetworkClient {self.client_id}: Direct connect failed: {e}. Will retry.")
                    self._close_socket()
                    # Wait in an interruptible way before the next attempt
                    if self._shutdown_event.wait(self.RETRY_DELAY_INTERNAL):
                        logging.info(f"NetworkClient {self.client_id}: Shutdown signal during direct connect retry delay.")
                        return False
                    continue # Go to next attempt in for loop

                if connect_in_progress:
                    # Use select to wait for connection to complete or fail
                    timeout_end_time = time.time() + self.RETRY_DELAY_INTERNAL
                    while time.time() < timeout_end_time:
                        if self._shutdown_event.is_set():
                            logging.info(f"NetworkClient {self.client_id}: Shutdown signal during non-blocking connect wait.")
                            self._close_socket()
                            return False # Abort if shutdown

                        # Wait for socket to be writable (connected) or exceptional (error), with a small poll interval
                        # The timeout for select should allow responsiveness to shutdown_event
                        readable, writable, exceptional = select.select([], [self.sock], [self.sock], 0.1)

                        if self.sock in writable:
                            # Socket is writable, connection established or error occurred (check SO_ERROR)
                            err = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                            if err == 0:
                                # Connection successful, break from inner while loop
                                break
                            else:
                                raise socket.error(f"Connection failed after non-blocking connect: {os.strerror(err)}")
                        elif self.sock in exceptional:
                            err = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                            raise socket.error(f"Socket error during non-blocking connect: {os.strerror(err)}")
                    else: # Inner while loop finished due to timeout without connecting
                        raise socket.timeout(f"Connection timed out after {self.RETRY_DELAY_INTERNAL}s during non-blocking connect.")

                # If we reach here, connection is established
                self.sock.setblocking(True) # Set back to blocking for normal operations
                logging.info(f"NetworkClient {self.client_id}: Successfully connected to {self.target_host}:{self.target_port}")

                # Perform initial handshake (e.g., send node ID)
                DataHandler.send_data(self.sock, initial_handshake_data)
                self.sock.settimeout(15.0) # Timeout for initial response from server
                response = DataHandler.receive_data(self.sock)
                self.sock.settimeout(None) # Reset to blocking after handshake

                if response is None:
                    logging.warning(f"NetworkClient {self.client_id}: Server closed connection gracefully after ID send during handshake. Retrying.")
                    self._close_socket()
                    if self._shutdown_event.wait(self.RETRY_DELAY_INTERNAL): return False
                    continue
                elif isinstance(response, dict) and response.get("command") == "reject":
                    logging.error(f"NetworkClient {self.client_id}: Connection rejected by server: {response.get('message', 'No message provided.')}. Not retrying this connection attempt type (e.g., invalid ID, duplicate).")
                    self._close_socket()
                    return False # Explicit rejection, don't retry immediately, Node's main loop should handle
                else:
                    logging.info(f"NetworkClient {self.client_id}: Received welcome/confirmation from server: {response}")
                    self.connected = True
                    return True # Successfully connected and handshaked

            except (socket.error, socket.timeout, ConnectionError, ValueError) as e:
                # Catch specific connection/timeout errors and ValueError from DataHandler (e.g., malformed JSON)
                logging.warning(f"NetworkClient {self.client_id}: Communication/handshake error during connection attempt: {e}. Retrying.")
                self._close_socket()
                if self._shutdown_event.wait(self.RETRY_DELAY_INTERNAL): return False
            except Exception as e:
                # Catch any other unexpected errors during the overall connect process
                logging.critical(f"NetworkClient {self.client_id}: An unexpected critical error occurred during connection attempt: {e}.", exc_info=True)
                self._close_socket()
                if self._shutdown_event.wait(self.RETRY_DELAY_INTERNAL): return False

        logging.error(f"NetworkClient {self.client_id}: Failed to connect after {self.MAX_CONNECT_ATTEMPTS} internal attempts. Signalling main Node loop.")
        self.connected = False
        return False

    def send(self, data: Any) -> None:
        """Sends data over the established connection."""
        if not self.connected or not self.sock:
            raise ConnectionError(f"NetworkClient {self.client_id}: Not connected to send data. Call connect() first.")
        try:
            DataHandler.send_data(self.sock, data)
        except (ConnectionError, socket.timeout, ValueError) as e:
            # DataHandler can raise ConnectionError (for socket issues) or ValueError (for malformed data)
            self.connected = False
            self._close_socket()
            logging.error(f"NetworkClient {self.client_id}: Connection lost or data error during send: {e}")
            raise ConnectionError(f"NetworkClient {self.client_id}: Connection lost during send.") from e
        except Exception as e:
            self.connected = False
            self._close_socket()
            logging.critical(f"NetworkClient {self.client_id}: Unexpected critical error during send: {e}", exc_info=True)
            raise ConnectionError(f"NetworkClient {self.client_id}: Unexpected error during send.") from e

    def receive(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Receives data from the connection.
        Returns deserialized data.
        Returns None if the peer performed a graceful shutdown (as per DataHandler contract).
        Raises ConnectionError if connection is lost due to error (as per DataHandler contract).
        Raises socket.timeout if the receive operation times out.
        """
        if not self.connected or not self.sock:
            raise ConnectionError(f"NetworkClient {self.client_id}: No active connection socket to receive data.")

        original_timeout = self.sock.gettimeout()
        if timeout is not None:
            self.sock.settimeout(timeout)

        try:
            received_data = DataHandler.receive_data(self.sock)
            return received_data
        except socket.timeout:
            raise # Re-raise timeout so the Node's run loop can decide to continue waiting/polling
        except (ConnectionError, ValueError) as e: # Catch ConnectionError and ValueError from DataHandler
            logging.error(f"NetworkClient {self.client_id}: Connection lost or data error during receive: {e}")
            self.connected = False
            self._close_socket()
            raise ConnectionError(f"NetworkClient {self.client_id}: Connection lost during receive.") from e
        except Exception as e:
            logging.critical(f"NetworkClient {self.client_id}: Unexpected critical error during receive: {e}", exc_info=True)
            self.connected = False
            self._close_socket()
            raise ConnectionError(f"NetworkClient {self.client_id}: Unexpected error during receive.") from e
        finally:
            # Only reset timeout if socket is still valid and timeout was temporarily set
            if self.sock and self.sock.gettimeout() != original_timeout:
                try:
                    self.sock.settimeout(original_timeout)
                except socket.error as e:
                    logging.debug(f"NetworkClient {self.client_id}: Error resetting socket timeout: {e} (socket might have closed).")


    def disconnect(self) -> None:
        """Initiates a graceful disconnection (closes the socket)."""
        # This is primarily called by the Node itself when it wants to shut down or before reconnecting
        logging.info(f"NetworkClient {self.client_id}: Disconnect command issued.")
        self._close_socket()