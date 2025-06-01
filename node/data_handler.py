import pickle
import zlib
import socket
import logging
from typing import Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataHandler:
    """Handles serialization, compression, deserialization of data."""
    CHUNK_SIZE = 4096

    @staticmethod
    def serialize_and_compress(data: Any) -> Optional[bytes]:
        """Serializes and compresses data."""
        try:
            serialized_data = pickle.dumps(data)
            compressed_data = zlib.compress(serialized_data)
            return compressed_data
        except (pickle.PickleError, zlib.error) as e:
            print(f"Error during serialization or compression: {e}")
            return None

    @staticmethod
    def decompress_and_deserialize(compressed_data: bytes) -> Optional[Any]:
        """Decompresses and deserializes data."""
        try:
            decompressed_data = zlib.decompress(compressed_data)
            return pickle.loads(decompressed_data)
        except (zlib.error, pickle.PickleError) as e:
            print(f"Error during decompression or deserialization: {e}")
            return None

    @classmethod
    def send_data(cls, sock: socket.socket, data: Any) -> bool:
        """Sends data over the socket in chunks."""
        compressed_data = cls.serialize_and_compress(data)
        if compressed_data is None:
            return False

        try:
            data_len = len(compressed_data).to_bytes(4, 'big')
            sock.sendall(data_len)
            for i in range(0, len(compressed_data), cls.CHUNK_SIZE):
                chunk = compressed_data[i:i + cls.CHUNK_SIZE]
                sock.sendall(chunk)
            return True
        except socket.error as e:
            print(f"Error sending data: {e}")
            return False

    @classmethod
    def receive_data(cls, sock: socket.socket) -> Optional[Any]:
        """Receives data from the socket in chunks and reassembles."""
        try:
            data_len_bytes = sock.recv(4)
            if not data_len_bytes:
                return None
            data_len = int.from_bytes(data_len_bytes, 'big')

            received_chunks = b''
            while len(received_chunks) < data_len:
                chunk = sock.recv(cls.CHUNK_SIZE)
                if not chunk:
                    return None  # Connection closed or error
                received_chunks += chunk
            return cls.decompress_and_deserialize(received_chunks)
        except socket.error as e:
            print(f"Error receiving data: {e}")
            return None
        except ValueError as e:
            print(f"Error processing received data length: {e}")
            return None