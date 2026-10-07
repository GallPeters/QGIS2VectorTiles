"""
worker.py

Runs one QGIS2VectorTiles conversion in a separate Python process, so the
heavy work never shares QGIS's interpreter, memory or threads with its
interface. Started by subprocess_runner.run_in_subprocess():

    python worker.py <job.json>

The job file holds the project path and the algorithm parameters. Log
messages and the result are written to stdout as JSON lines prefixed with
MESSAGE_PREFIX; anything else on stdout/stderr is ordinary output. Creating
the job's cancel file cancels the run.

This file is a script: it must not be imported by the plugin.
"""

import importlib
import json
import os
import shutil
import sys
import threading
import time
import traceback

MESSAGE_PREFIX = "\x1eQ2VT "
_out_lock = threading.Lock()


def emit(kind: str, **data) -> None:
    """Send one message to the parent (thread-safe)."""
    line = MESSAGE_PREFIX + json.dumps({"type": kind, **data}) + "\n"
    with _out_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def main(job_path: str) -> int:
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    sys.path.insert(0, job["plugins_dir"])

    from qgis.core import (
        QgsApplication,
        QgsProcessingFeedback,
        QgsProcessingUtils,
        QgsProject,
        QgsRectangle,
    )

    QgsApplication.setPrefixPath(job["prefix_path"], True)
    # Same profile as the parent: same plugin folder, auth and settings.
    app = QgsApplication([], False, job["profile_dir"])
    app.initQgis()
    sys.path.append(os.path.join(QgsApplication.pkgDataPath(), "python", "plugins"))
    from processing.core.Processing import Processing

    Processing.initialize()

    class Feedback(QgsProcessingFeedback):
        """Forwards everything the pipeline reports to the parent."""

        def __init__(self):
            super().__init__(False)
            self._last_progress = -1.0
            self.progressChanged.connect(self._on_progress)

        def pushInfo(self, info):
            emit("info", text=info)

        def pushWarning(self, warning):
            emit("warning", text=warning)

        def pushDebugInfo(self, info):
            emit("debug", text=info)

        def pushCommandInfo(self, info):
            emit("debug", text=info)

        def pushConsoleInfo(self, info):
            emit("debug", text=info)

        def reportError(self, error, fatalError=False):
            emit("error", text=error, fatal=bool(fatalError))

        def setProgressText(self, text):
            emit("progress_text", text=text)

        def _on_progress(self, value):
            if abs(value - self._last_progress) >= 1 or value >= 100:
                self._last_progress = value
                emit("progress", value=value)

    feedback = Feedback()

    def watch_cancel():
        while not feedback.isCanceled():
            if os.path.exists(job["cancel_file"]):
                feedback.cancel()
                return
            time.sleep(0.3)

    threading.Thread(target=watch_cancel, daemon=True, name="cancel-watch").start()

    if not QgsProject.instance().read(job["project_path"]):
        emit("error", text=f"Cannot read the project '{job['project_path']}'.", fatal=True)
        return 2

    package = importlib.import_module(f"{job['package']}.src.qgis2vectortiles")
    params = job["params"]
    runner = package.QGIS2VectorTiles(
        min_zoom=params["min_zoom"],
        max_zoom=params["max_zoom"],
        extent=QgsRectangle(*params["extent"]),
        cpu_percent=params["cpu_percent"],
        output_dir=params["output_dir"],
        include_required_fields_only=params["include_required_fields_only"],
        cent_source=params["cent_source"],
        background_type=params["background_type"],
        viewer=params["viewer"],
        feedback=feedback,
    )
    # The parent starts the tile server and changes its own project.
    runner.serve_tiles = lambda temp_dir: None
    runner.convert_project_to_vector_tiles()
    emit("result", temp_dir=runner.temp_dir, min_zoom=runner.min_zoom,
         canceled=feedback.isCanceled())

    # This process's temporary files (materialised sources, rule outputs);
    # the results are in output_dir, which belongs to the parent.
    del runner
    shutil.rmtree(QgsProcessingUtils.tempFolder(), ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        code = main(sys.argv[1])
    except BaseException:  # noqa: BLE001  (report anything to the parent)
        emit("error", text=traceback.format_exc(), fatal=True)
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip QGIS/GDAL teardown: everything is written and closed, and their
    # exit-time cleanup can crash in a standalone process.
    os._exit(code)
