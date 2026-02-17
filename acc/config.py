"""AccConfig — centralized network configuration for both processes.

Reads from env vars with sane defaults. CLI args override env vars.
Future milestones add more config here without scattering env vars.
"""

import os
import socket
from dataclasses import dataclass, field


@dataclass
class AccConfig:
    """Network configuration for the ACC two-process architecture.

    Defaults to localhost for both processes. Override via CLI args or env vars.

    Env vars:
        ACC_TRAINER_HOST: Trainer bind host (default: 0.0.0.0)
        ACC_TRAINER_PORT: Trainer bind port (default: 8787)
        ACC_UI_HOST: UI bind host (default: 0.0.0.0)
        ACC_UI_PORT: UI bind port (default: 8080)
        ACC_TRAINER_URL: Full trainer URL for UI to connect to (default: computed)
    """

    trainer_host: str = field(
        default_factory=lambda: os.environ.get("ACC_TRAINER_HOST", "0.0.0.0")
    )
    trainer_port: int = field(
        default_factory=lambda: int(os.environ.get("ACC_TRAINER_PORT", "8787"))
    )
    ui_host: str = field(
        default_factory=lambda: os.environ.get("ACC_UI_HOST", "0.0.0.0")
    )
    ui_port: int = field(
        default_factory=lambda: int(os.environ.get("ACC_UI_PORT", "8080"))
    )
    # Explicit trainer URL for the UI to connect to. If set, overrides computed URL.
    trainer_url: str = field(
        default_factory=lambda: os.environ.get("ACC_TRAINER_URL", "")
    )

    def get_trainer_url(self) -> str:
        """Resolve the trainer URL the UI should connect to.

        Priority: explicit trainer_url > computed from host:port.
        """
        if self.trainer_url:
            return self.trainer_url
        return f"http://localhost:{self.trainer_port}"

    def print_trainer_info(self) -> None:
        """Print trainer startup info with accessible addresses."""
        hostname = _get_hostname()
        print(f"ACC Trainer starting on {self.trainer_host}:{self.trainer_port}")
        print(f"  Local:     http://localhost:{self.trainer_port}")
        if hostname:
            print(f"  Hostname:  http://{hostname}:{self.trainer_port}")
        print(f"  Network:   http://<your-ip>:{self.trainer_port}")

    def print_ui_info(self) -> None:
        """Print UI startup info with trainer connection details."""
        trainer_url = self.get_trainer_url()
        hostname = _get_hostname()
        print(f"ACC Dashboard starting on {self.ui_host}:{self.ui_port}")
        print(f"  Local:     http://localhost:{self.ui_port}")
        if hostname:
            print(f"  Hostname:  http://{hostname}:{self.ui_port}")
        print(f"  Trainer:   {trainer_url}")


def _get_hostname() -> str:
    """Get the machine's hostname, or empty string on failure."""
    try:
        return socket.gethostname()
    except Exception:
        return ""
