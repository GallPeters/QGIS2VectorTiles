@echo off

REM Go to project root (one level above this bat file's location)
cd /d %~dp0

echo Starting MBTiles server...

REM Launch the server in a NEW window (non-blocking)
start "MBTiles Server" python utils\mbtiles_server.py --port _Q2VTPORT

REM Give the server a moment to start before opening the browser
timeout /t 2 /nobreak > nul

REM Open the viewer in the default browser
REM (empty first string "" is a dummy title so the URL is parsed correctly)
start "" "http://localhost:_Q2VTPORT/maplibre_viewer.html"

pause