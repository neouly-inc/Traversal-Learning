import socket
import logging
import sys
from typing import Optional


class Connection:
    """Represents an active connection from a client node."""
    def __init__(self, sock: socket.socket, addr: tuple) -> None:
        self.sock = sock
        self.addr = addr
        self.node_id: Optional[int] = None # Will be set after handshake
        self.active = True # Indicates if the connection is currently considered active

        self._apply_keepalive_options()

    def _apply_keepalive_options(self) -> None:
        """Applies TCP Keepalive options to the socket."""
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            # Platform-specific TCP Keepalive options
            if sys.platform == 'linux':
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30) # Start probes after 30s of inactivity
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)  # Send probes every 5s
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)    # 3 failed probes to consider dead
                logging.debug(f"Connection: Applied Linux TCP Keepalive for {self.addr}.")
            elif sys.platform == 'darwin': # macOS
                # TCP_KEEPALIVE on macOS takes the idle time in seconds.
                # No direct equivalents for KEEPINTVL, KEEPCNT via setsockopt in Python's socket module for macOS.
                # Default is usually 2 hours, can be changed system-wide via sysctl.
                # Using 60 seconds as a reasonable application-level idle detection
                TCP_KEEPALIVE = getattr(socket, 'TCP_KEEPALIVE', 0x10) # 0x10 is TCP_KEEPALIVE on macOS
                self.sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPALIVE, 60)
                logging.debug(f"Connection: Applied macOS TCP Keepalive for {self.addr}.")
            # For Windows, TCP Keepalive values are typically set via SIO_KEEPALIVE_VALS ioctl,
            # which is not directly exposed in standard Python socket.setsockopt.
            # Rely on SO_KEEPALIVE=1 and OS defaults or registry settings for Windows.
            elif sys.platform == 'win32':
                # Windows defaults are usually 2 hours.
                # To set specific values on Windows, you'd typically need to use ctypes with WSAIoctl/SIO_KEEPALIVE_VALS.
                # This is beyond standard socket module and portability.
                logging.debug(f"Connection: Using default Windows TCP Keepalive for {self.addr}.")

        except AttributeError:
            logging.warning(f"Connection: Could not set some platform-specific TCP options for {self.addr}. "
                            "Likely unsupported OS/Python version. Keepalive might be less aggressive.")
        except socket.error as e:
            logging.warning(f"Connection: Could not set SO_KEEPALIVE or platform-specific TCP options for {self.addr}: {e}")
        except Exception as e:
            logging.error(f"Connection: Unexpected error applying keepalive options for {self.addr}: {e}")


    def close(self) -> None:
        """Closes the socket connection and marks it as inactive."""
        if not self.active:
            logging.debug(f"Connection: Attempted to close already inactive connection for {self.addr}.")
            return # Already inactive

        self.active = False
        peername = None
        try:
            # Try to get peername before closing, for better logging
            peername = self.sock.getpeername()
        except socket.error:
            # Socket might already be closed or in a bad state, ignore
            pass

        try:
            # Attempt a graceful shutdown if possible
            # This can fail if the peer has already closed or if the socket is already shutdown
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError as e:
            logging.debug(f"Connection: Error during socket shutdown for {self.addr}: {e} (might already be closed).")
        except Exception as e:
            logging.warning(f"Connection: Unexpected error during socket shutdown for {self.addr}: {e}")
        finally:
            self.sock.close()
            self.sock = None # Important to nullify the socket reference

        log_msg = f"Connection from {peername or self.addr} closed."
        if self.node_id is not None:
            log_msg = f"Connection for Node {self.node_id} ({peername or self.addr}) closed."
        logging.info(f"Connection: {log_msg}")