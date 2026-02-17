"""Trainer process entry point — starts on :6060.

Usage:
    python -m acc.trainer_main
    python -m acc.trainer_main --port 9787

Starts the trainer HTTP API. The model can be configured via:
1. HTTP API calls (POST /model/create, /datasets/load_builtin, /tasks/add)
2. Or by importing and calling setup() before run()
"""

import argparse

from acc.trainer_api import TrainerAPI
from acc.config import AccConfig


def main():
    parser = argparse.ArgumentParser(description="ACC Trainer Process")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    args = parser.parse_args()

    config = AccConfig()
    if args.port is not None:
        config.trainer_port = args.port
    if args.host is not None:
        config.trainer_host = args.host

    api = TrainerAPI()
    config.print_trainer_info()
    print(f"  Device:    {api.device}")
    api.run(host=config.trainer_host, port=config.trainer_port)


if __name__ == "__main__":
    main()
