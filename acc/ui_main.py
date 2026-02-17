"""UI process entry point — starts dashboard on localhost:8080.

Usage:
    python -m acc.ui_main

Hot-reloads freely via uvicorn --reload.
All state comes from the trainer API on localhost:8787.
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="ACC Dashboard UI")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--reload", action="store_true", default=True)
    args = parser.parse_args()

    print(f"ACC Dashboard starting on {args.host}:{args.port}")
    print("Connecting to trainer at http://localhost:8787")
    uvicorn.run(
        "acc.ui.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=["acc/ui"],
    )


if __name__ == "__main__":
    main()
