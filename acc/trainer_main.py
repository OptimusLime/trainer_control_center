"""Trainer process entry point — starts on localhost:8787.

Usage:
    python -m acc.trainer_main

Starts the trainer HTTP API. The model can be configured via:
1. HTTP API calls (POST /model/create, /datasets/load_builtin, /tasks/add)
2. Or by importing and calling setup() before run()
"""

import argparse

from acc.trainer_api import TrainerAPI


def main():
    parser = argparse.ArgumentParser(description="ACC Trainer Process")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    api = TrainerAPI()
    print(f"ACC Trainer starting on {args.host}:{args.port}")
    print(f"Device: {api.device}")
    print("Use HTTP API or python -m acc.ui to interact.")
    api.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
