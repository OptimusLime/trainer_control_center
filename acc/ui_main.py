"""UI process entry point — starts dashboard on :8081.

Usage:
    python -m acc.ui_main
    python -m acc.ui_main --trainer-url http://paul-cheddar:6060
    python -m acc.ui_main --port 9090

Hot-reloads freely via uvicorn --reload.
All state comes from the trainer API.
"""

import argparse
import os

import uvicorn

from acc.config import AccConfig


def main():
    parser = argparse.ArgumentParser(description="ACC Dashboard UI")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument(
        "--trainer-url",
        type=str,
        default=None,
        help="Trainer API URL (e.g. http://paul-cheddar:6060)",
    )
    parser.add_argument("--reload", action="store_true", default=True)
    parser.add_argument("--no-reload", action="store_true", default=False)
    args = parser.parse_args()

    config = AccConfig()
    if args.port is not None:
        config.ui_port = args.port
    if args.host is not None:
        config.ui_host = args.host
    if args.trainer_url is not None:
        config.trainer_url = args.trainer_url

    # Set env var so the app module can read it on import
    # (uvicorn --reload re-imports the module, so we can't pass it directly)
    os.environ["ACC_TRAINER_URL"] = config.get_trainer_url()

    config.print_ui_info()

    reload = args.reload and not args.no_reload
    uvicorn.run(
        "acc.ui.app:app",
        host=config.ui_host,
        port=config.ui_port,
        reload=reload,
        reload_dirs=["acc/ui"],
    )


if __name__ == "__main__":
    main()
