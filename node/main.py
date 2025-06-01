import argparse
import logging
import sys
import torch

from node import Node

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    """Main function to parse arguments and start the Node."""
    parser = argparse.ArgumentParser(description="Node for distributed learning.")
    parser.add_argument("--node_id", type=int, help="Unique ID for this node.", required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Orchestrator IP address.")
    parser.add_argument("--port", type=int, default=12345, help="Orchestrator port number.")
    parser.add_argument("--num_nodes", type=int, default=2, help="Number of nodes expected to connect.") # Only used for data splitting
    parser.add_argument('--no-accel', action='store_true', help='Disables accelerator')
    args = parser.parse_args()

    # Check if the accelerator is available and set the device accordingly
    use_accel = not args.no_accel and torch.accelerator.is_available()
    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    # Basic input validation
    if args.node_id <= 0:
        logging.error("Node ID must be a positive integer.")
        sys.exit(1)

    node = None
    try:
        node = Node(args.node_id, args.host, args.port, args.num_nodes, device)
        node.run()
    except Exception as e:
        logging.critical(f"Main Node execution failed: {e}", exc_info=True)
    finally:
        # Ensure signal_shutdown is called even if an exception occurred during run()
        if node:
            node.signal_shutdown()


if __name__ == "__main__":
    main()