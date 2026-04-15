#!/usr/bin/env bash
# Start the MBTiles server in the background, then open the viewer.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting MBTiles server..."
python3 "$SCRIPT_DIR/mbtiles_server.py" --port _Q2VTPORT &
SERVER_PID=$!

# Wait for the server to be ready (poll the port, timeout after 10 s).
for i in $(seq 1 10); do
    sleep 1
    if python3 -c "import socket; s=socket.create_connection(('127.0.0.1',_Q2VTPORT),1); s.close()" 2>/dev/null; then
        break
    fi
done

# Open the viewer — works on macOS (open) and most Linux desktops (xdg-open).
if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:_Q2VTPORT/"
elif command -v open &>/dev/null; then
    open "http://localhost:_Q2VTPORT/"
else
    echo "Server running. Open http://localhost:_Q2VTPORT/ in your browser."
fi

# Keep the script alive so Ctrl-C stops the server cleanly.
wait $SERVER_PID
