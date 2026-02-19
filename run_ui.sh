#!/usr/bin/env bash
# Start the ACC Dashboard (Astro dev server).
#
# Usage:
#   ./run_ui.sh              # dev server on port 4321, proxies API to localhost:6060
#   ./run_ui.sh build        # build for production (trainer serves static files)
#
# Dev mode: Astro dev server on http://0.0.0.0:4321 with hot reload.
# API requests are proxied to the trainer on localhost:6060.
#
# Production: run './run_ui.sh build', then the trainer serves the dashboard
# directly at http://localhost:6060/.

set -e

DASHBOARD_DIR="$(dirname "$0")/dashboard"

# Ensure dependencies are installed
if [ ! -d "$DASHBOARD_DIR/node_modules" ]; then
    echo "Installing dashboard dependencies..."
    npm install --prefix "$DASHBOARD_DIR"
fi

if [ "$1" = "build" ]; then
    echo "========================================"
    echo "  ACC Dashboard — Production Build"
    echo "========================================"
    echo ""
    npm run build --prefix "$DASHBOARD_DIR"
    echo ""
    echo "Build complete. Restart the trainer to serve the dashboard."
else
    echo "========================================"
    echo "  ACC Dashboard — Dev Server"
    echo "========================================"
    echo ""
    echo "  Dashboard: http://localhost:4321"
    echo "  Trainer:   http://localhost:6060 (API proxy)"
    echo ""
    npm run dev --prefix "$DASHBOARD_DIR"
fi
