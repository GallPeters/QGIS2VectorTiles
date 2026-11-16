"""Screenshot regression check of the web map: exports projects with the
plugin, takes MapLibre screenshots of fixed views, and compares them with
reference images.

Run it with QGIS's Python (on Windows, OSGeo4W's python-qgis.bat):

    python-qgis check.py <views.json> [--update] [--only NAME]

views.json (see views.example.json) lists projects, each with its export
extent and zooms, and views (centre in EPSG:3857, zoom, size). References
are kept in "references" (a folder outside the repository: they show your
own data), one PNG per view. --update records the current screenshots as
the references; otherwise each view is compared and its difference image
written next to the screenshot. A view fails when more than "tolerance"
(default 0.5) percent of its pixels differ by more than "threshold" (0-255,
default 40) in any channel.

Needs Node.js and puppeteer-core (npm install, in this folder) and Chrome
or Edge ("browser" in views.json, else the usual install paths).

Run it before every release and before upgrading MapLibre: the plugin
depends on how MapLibre draws some things (badges, stretched backgrounds,
variable anchors), which an upgrade can change without an error.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome", "/usr/bin/chromium",
)
NODES = (r"C:\Program Files\nodejs\node.exe", "/usr/local/bin/node", "/opt/homebrew/bin/node", "/usr/bin/node")


def export(project: dict, out_dir: str) -> str:
    """Export a project with the repository's plugin (in its worker
    process, as QGIS runs it); returns the export folder."""
    from qgis.core import QgsApplication
    work = os.path.join(out_dir, f"job_{uuid.uuid4().hex[:8]}")
    os.makedirs(work)
    job = {
        "plugins_dir": os.path.dirname(REPO), "package": os.path.basename(REPO),
        "prefix_path": QgsApplication.prefixPath(), "profile_dir": QgsApplication.qgisSettingsDirPath(),
        "project_path": project["project"], "cancel_file": os.path.join(work, "cancel"),
        "params": {"min_zoom": project.get("min_zoom", 0), "max_zoom": project.get("max_zoom", 16),
                   "extent": project["extent"], "cpu_percent": 100, "output_dir": out_dir,
                   "included_fields": project.get("included_fields", 0)},
    }
    job_path = os.path.join(work, "job.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f)
    worker = os.path.join(REPO, "src", "processing", "worker.py")
    result = None
    process = subprocess.run([sys.executable, "-u", worker, job_path], capture_output=True,
                             encoding="utf-8", errors="replace", cwd=work)
    for line in process.stdout.split("\n"):
        if line.startswith("\x1eQ2VT "):
            message = json.loads(line[6:])
            if message.get("type") == "result":
                result = message.get("temp_dir")
            elif message.get("type") in ("error", "warning"):
                print("   ", message.get("type"), str(message.get("text"))[:300])
    if not result:
        raise RuntimeError(f"export failed: {process.stderr[-1500:]}")
    return result


def lay_out(folder: str) -> None:
    """The web map's files around the export, as the plugin's server lays
    them out (the background: the project's colour)."""
    viewer = os.path.join(REPO, "resources", "viewer")
    for name in ("ml", "ol"):
        target = os.path.join(folder, name)
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(os.path.join(viewer, name), target)
    with open(os.path.join(viewer, "viewer.html"), encoding="utf-8") as f:
        html = f.read()
    for key, value in {"_Q2VT_MINZOOM": "0", "_Q2VT_CENTER": "[0, 0]", "_Q2VT_BOUNDS": "null",
                       "_Q2VT_STORAGE_KEY": "check", "_Q2VT_BASEMAP": "none"}.items():
        html = html.replace(key, value)
    with open(os.path.join(folder, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    style = os.path.join(folder, "style", "style.json")
    with open(style, encoding="utf-8") as f:
        text = f.read().replace("10031993", "check")
    with open(style, "w", encoding="utf-8") as f:
        f.write(text)
    os.makedirs(os.path.join(folder, "server"), exist_ok=True)
    shutil.copy(os.path.join(REPO, "resources", "tiles_server.py"), os.path.join(folder, "server"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def compare(reference: str, current: str, diff_path: str, threshold: int) -> float:
    """Percent of pixels differing by more than ``threshold``; writes
    ``diff_path`` (differing pixels in red over a faded current image)."""
    from qgis.PyQt.QtGui import QColor, QImage
    a, b = QImage(reference), QImage(current)
    if a.size() != b.size():
        return 100.0
    diff = QImage(b.size(), QImage.Format.Format_RGB32)
    count = 0
    for y in range(b.height()):
        for x in range(b.width()):
            p, q = a.pixelColor(x, y), b.pixelColor(x, y)
            if max(abs(p.red() - q.red()), abs(p.green() - q.green()), abs(p.blue() - q.blue())) > threshold:
                count += 1
                diff.setPixelColor(x, y, QColor(255, 0, 0))
            else:
                grey = (q.red() + q.green() + q.blue()) // 3
                diff.setPixelColor(x, y, QColor(grey // 2 + 128, grey // 2 + 128, grey // 2 + 128))
    diff.save(diff_path)
    return 100.0 * count / (b.width() * b.height())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config")
    parser.add_argument("--update", action="store_true", help="record the screenshots as references")
    parser.add_argument("--only", help="only this project (its name)")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    references = config["references"]
    work = config.get("work_dir") or os.path.join(references, "_run")
    os.makedirs(references, exist_ok=True)
    browser = config.get("browser") or next((b for b in BROWSERS if os.path.isfile(b)), None)
    if not browser:
        print("No Chrome or Edge found: set \"browser\" in the config.")
        return 2
    # QGIS's Python environment may leave Node.js off the PATH.
    node = config.get("node") or shutil.which("node") or next(
        (n for n in NODES if os.path.isfile(n)), None)
    if not node:
        print("No Node.js found: set \"node\" in the config.")
        return 2

    from qgis.core import QgsApplication
    app = QgsApplication([], False)
    app.initQgis()

    failures = 0
    for project in config["projects"]:
        if args.only and project["name"] != args.only:
            continue
        print(f"{project['name']}: exporting...")
        out_dir = os.path.join(work, project["name"])
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        started = time.time()
        folder = export(project, out_dir)
        print(f"   exported in {time.time() - started:.0f} s")
        lay_out(folder)
        port = free_port()
        server = subprocess.Popen([sys.executable, os.path.join(folder, "server", "tiles_server.py"),
                                   "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(2)
            for view in project["views"]:
                shot = os.path.join(out_dir, f"{project['name']}_{view['name']}.png")
                command = [node, os.path.join(HERE, "shoot.mjs"), browser, str(port), shot,
                           repr(view["center"][0]), repr(view["center"][1]), repr(view["zoom"]),
                           str(view.get("width", 1000)), str(view.get("height", 700))]
                result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace")
                if result.returncode != 0 or not os.path.isfile(shot):
                    print(f"   {view['name']}: screenshot failed: {result.stderr[-500:]}")
                    failures += 1
                    continue
                if result.stdout.strip():
                    print(f"   {view['name']}: browser messages: {result.stdout.strip()[:300]}")
                reference = os.path.join(references, f"{project['name']}_{view['name']}.png")
                if args.update or not os.path.isfile(reference):
                    shutil.copy(shot, reference)
                    print(f"   {view['name']}: reference recorded")
                    continue
                percent = compare(reference, shot, shot.replace(".png", "_diff.png"),
                                  int(view.get("threshold", config.get("threshold", 40))))
                tolerance = float(view.get("tolerance", config.get("tolerance", 0.5)))
                status = "ok" if percent <= tolerance else "DIFFERS"
                failures += status != "ok"
                print(f"   {view['name']}: {percent:.2f}% of pixels differ ({status})")
        finally:
            server.kill()
    print("all views match" if not failures else f"{failures} view(s) differ: see the _diff images in {work}")
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
