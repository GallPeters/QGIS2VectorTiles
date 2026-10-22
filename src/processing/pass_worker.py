"""
pass_worker.py

Runs one rule-export pass (RulesExporter._export_rule_pass) in a separate
Python process: the per-feature Python work of several passes then runs on
several cores at once, which threads sharing one interpreter can't.
Started by RulesExporter (see _run_pass_process):

    python pass_worker.py <job.json>

The job file holds the base layer, the rule groups (plain data), the export
extent, the feature range, the expression variables and where to write the
result. The result file lists each dataset's part file (or null) and the
warnings; a missing result file means the pass failed and the caller runs
it itself. Creating the job's cancel file cancels the pass.

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

    from qgis.core import (
        QgsApplication,
        QgsExpression,
        QgsExpressionContext,
        QgsExpressionContextScope,
        QgsExpressionContextUtils,
        QgsProcessingFeedback,
        QgsRectangle,
    )

    QgsApplication.setPrefixPath(job["prefix_path"], True)
    app = QgsApplication([], False, job["profile_dir"])
    app.initQgis()
    # The plugin's modules import QGIS's processing package.
    sys.path.append(os.path.join(QgsApplication.pkgDataPath(), "python", "plugins"))
    faulthandler.enable()

    exporter_module = importlib.import_module(f"{job['package']}.src.core.rules_exporter")

    warnings = []

    class Feedback(QgsProcessingFeedback):
        """Keeps the pass's warnings for the result file."""

        def pushWarning(self, warning):
            warnings.append(warning)

    feedback = Feedback()

    def watch_cancel():
        while not feedback.isCanceled():
            if os.path.exists(job["cancel_file"]):
                feedback.cancel()
                return
            time.sleep(0.3)

    threading.Thread(target=watch_cancel, daemon=True, name="cancel-watch").start()

    groups = [exporter_module.group_from_dict(data) for data in job["groups"]]
    # An expression this process can't parse (a function only the parent's
    # QGIS has, e.g. from a plugin) fails the pass: the caller runs it.
    for grp in groups:
        for expression in exporter_module.group_expressions(grp):
            if QgsExpression(expression).hasParserError():
                return 3

    exporter = exporter_module.RulesExporter(
        [], QgsRectangle(*job["extent"]), 0, 0, job["utils_dir"], feedback,
    )
    context = QgsExpressionContext()
    context.appendScope(QgsExpressionContextUtils.globalScope())
    scope = QgsExpressionContextScope("project")
    for name, value in job["variables"].items():
        scope.setVariable(name, value)
    context.appendScope(scope)

    fid_range = tuple(job["fid_range"]) if job.get("fid_range") else None
    outputs = exporter._export_rule_pass(  # pylint: disable=protected-access
        groups, job["base_path"], context, fid_range=fid_range, suffix=job["suffix"],
    )
    if feedback.isCanceled():
        return 4
    with open(job["result_path"], "w", encoding="utf-8") as f:
        json.dump({"outputs": outputs, "warnings": warnings}, f)
    return 0


if __name__ == "__main__":
    try:
        code = main(sys.argv[1])
    except BaseException:  # noqa: BLE001  (the caller runs the pass itself)
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip QGIS/GDAL teardown, as worker.py does.
    os._exit(code)
