"""Trainer process entry point — starts on :6060.

Usage:
    python -m acc.trainer_main
    python -m acc.trainer_main --port 9787

Starts the trainer HTTP API with uvicorn --reload so code changes
take effect without manual restarts.

The model can be configured via:
1. HTTP API calls (POST /model/create, /datasets/load_builtin, /tasks/add)
2. Recipes (POST /recipes/{name}/run)
3. Or by importing and calling setup() before run()
"""

import argparse

from acc.trainer_api import TrainerAPI
from acc.config import AccConfig


def main():
    parser = argparse.ArgumentParser(description="ACC Trainer Process")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--reload", action="store_true", default=True)
    parser.add_argument("--no-reload", action="store_true", default=False)
    args = parser.parse_args()

    config = AccConfig()
    if args.port is not None:
        config.trainer_port = args.port
    if args.host is not None:
        config.trainer_host = args.host

    reload = args.reload and not args.no_reload

    if reload:
        # Use lazy app so uvicorn can reload the process
        import uvicorn
        config.print_trainer_info()
        uvicorn.run(
            "acc.trainer_api:_default_app",
            host=config.trainer_host,
            port=config.trainer_port,
            reload=True,
            reload_dirs=["acc"],
            reload_excludes=["acc/checkpoints_data/*", "acc/recipes/__pycache__/*"],
        )
    else:
        api = TrainerAPI()
        config.print_trainer_info()
        print(f"  Device:    {api.device}")
        api.run(host=config.trainer_host, port=config.trainer_port)


if __name__ == "__main__":
    main()
