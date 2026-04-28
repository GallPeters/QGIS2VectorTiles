#!/bin/bash

# Kill processes using selected port
lsof -ti :18111991 | xargs kill -9 2>/dev/null

# Go to project root (one level above this script's location)
cd "$(dirname "$0")"

echo "Starting MBTiles server..."

# Launch the server in the background (non-blocking)
_Q2VT_PYTHON _Q2VT_UTILS --port 18111991 &

# Give the server a moment to start before opening the browser
sleep 2

# Open the viewer in the default browser
if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:18111991/maplibre_viewer.html"   # Linux
elif command -v open &>/dev/null; then
    open "http://localhost:18111991/maplibre_viewer.html"       # macOS
fi
