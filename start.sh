#!/usr/bin/env bash
# Double-click / run this to start the Salomon Earnings Desk locally.
cd "$(dirname "$0")"
echo "Starting Salomon Earnings Desk…"
echo "Open this in your browser: http://localhost:8000"
echo "Press Ctrl+C to stop."
exec python3 server.py
