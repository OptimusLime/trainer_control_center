#!/usr/bin/env bash
# Start the ACC dashboard UI process.
#
# Usage:
#   ./run_ui.sh                        # local trainer (http://localhost:6060)
#   ./run_ui.sh --remote paul-cheddar  # remote trainer via hostname
#   ./run_ui.sh --trainer-url http://100.121.59.123:6060  # explicit URL
#   ./run_ui.sh --port 9090            # custom UI port
#
# The UI binds to 0.0.0.0, so it's accessible from the browser on
# any machine on the same network.

set -e

# Parse --remote flag and convert to --trainer-url
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)
            REMOTE_HOST="$2"
            ARGS+=("--trainer-url" "http://${REMOTE_HOST}:6060")
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

echo "========================================"
echo "  ACC Dashboard UI"
echo "========================================"
echo ""

python -m acc.ui_main "${ARGS[@]}"
