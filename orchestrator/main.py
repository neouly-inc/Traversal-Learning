import argparse
import logging
import sys
import torch

from orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    """Parses command-line arguments and starts the Orchestrator."""
    parser = argparse.ArgumentParser(description="Orchestrator for distributed learning.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Orchestrator IP address.")
    parser.add_argument("--port", type=int, default=12345, help="Orchestrator port number.")
    parser.add_argument("--num_nodes", type=int, default=2, help="Number of nodes expected to connect.")
    parser.add_argument("--epochs", type=int, default=14, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of samples per batch.")
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate for the optimizer')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning rate step gamma')
    parser.add_argument('--no-accel', action='store_true', help='Disables accelerator')
    args = parser.parse_args()

    # Check if the accelerator is available and set the device accordingly
    use_accel = not args.no_accel and torch.accelerator.is_available()
    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    # Basic input validation
    if args.num_nodes <= 0:
        logging.error("Number of nodes must be a positive integer.")
        sys.exit(1)
    if args.epochs <= 0:
        logging.error("Number of epochs must be a positive integer.")
        sys.exit(1)

    orchestrator = None
    try:
        orchestrator = Orchestrator(
            host=args.host,
            port=args.port,
            num_nodes=args.num_nodes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            gamma=args.gamma,
            device=device,
        )
        orchestrator.run()
    except Exception as e:
        logging.critical(f"Main Orchestrator execution failed: {e}", exc_info=True)
    finally:
        # Ensure shutdown is called even if an exception occurred during run()
        if orchestrator:
            orchestrator.shutdown()


if __name__ == "__main__":
    main()