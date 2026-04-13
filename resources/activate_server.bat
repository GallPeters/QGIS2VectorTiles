@echo off

REM Go to project root (where this bat file is)
cd /d %~dp0

echo Starting MBTiles server...

REM Optional: open browser
start http://localhost:9000

REM Run server from utils folder
python utils\mbtiles_server.py --port 9000

pause