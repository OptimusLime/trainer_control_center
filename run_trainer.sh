#!/usr/bin/env bash
# Start the ACC trainer process.
#
# Usage:
#   ./run_trainer.sh              # default port 6060
#   ./run_trainer.sh --port 9787  # custom port
#
# The trainer binds to 0.0.0.0, so it's accessible from any machine
# on the same Tailscale network (or local network).

set -e

echo "========================================"
echo "  ACC Trainer Process"
echo "========================================"
echo ""

# Show network addresses for convenience
echo "Machine addresses:"
HOSTNAME=$(hostname 2>/dev/null || echo "unknown")
echo "  Hostname: $HOSTNAME"

# Try to get Tailscale IP
if command -v tailscale &>/dev/null; then
    TS_IP=$(tailscale ip -4 2>/dev/null || echo "")
    if [ -n "$TS_IP" ]; then
        echo "  Tailscale: $TS_IP"
    fi
fi

echo ""

python -m acc.trainer_main "$@"
