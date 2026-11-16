"""
place_worker.py

Lays out balloon labels' zoom bands (LabelPlacer._layout) in a separate
Python process, so that several bands are laid out on several cores at once
(QGIS's labeling engine runs on one). Started by LabelPlacer._layouts:

    python place_worker.py <job.json>

The job file holds the project to read, the export extent, the bands' zooms
and the labelled layers' points; the result file holds, per zoom, each
label's box: [layer id, provider id, feature id, left, top, right, bottom].
A missing result file means the process failed and the caller lays the
bands out itself. Creating the job's cancel file stops it.

This file is a script: it must not be imported by the plugin.
"""

import faulthandler
import importlib
import json
import os
import sys
import threading
import time


def main(job_path: str) -> int:
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    sys.path.insert(0, job["plugins_dir"])

    from qgis.core import QgsApplication, QgsProcessingFeedback, QgsProject, QgsRectangle

    QgsApplication.setPrefixPath(job["prefix_path"], True)
    app = QgsApplication([], False, job["profile_dir"])
    app.initQgis()
    sys.path.append(os.path.join(QgsApplication.pkgDataPath(), "python", "plugins"))
    faulthandler.enable()

    if not QgsProject.instance().read(job["project_path"]):
        return 2
    placer_module = importlib.import_module(f"{job['package']}.src.core.label_placer")

    feedback = QgsProcessingFeedback()

    def watch_cancel():
        while not feedback.isCanceled():
            if os.path.exists(job["cancel_file"]):
                feedback.cancel()
                return
            time.sleep(0.3)

    threading.Thread(target=watch_cancel, daemon=True, name="cancel-watch").start()

    placer = placer_module.LabelPlacer([], [], QgsRectangle(*job["extent"]), 0, job["work_dir"], feedback)
    coordinates = {layer_id: {int(fid): (x, y) for fid, x, y in points}
                   for layer_id, points in job["coordinates"].items()}
    result = {}
    for zoom in job["zooms"]:
        if feedback.isCanceled():
            return 4
        boxes = placer._layout(zoom + placer_module.BAND_NUDGE, coordinates)  # pylint: disable=protected-access
        result[repr(zoom)] = [[layer_id, provider, fid, *box]
                              for (layer_id, provider), found in boxes.items()
                              for fid, box in found.items()]
    with open(job["result_path"], "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    try:
        code = main(sys.argv[1])
    except BaseException:  # noqa: BLE001  (the caller lays the bands out itself)
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
