"""
pass_worker.py

Runs rule-export passes (RulesExporter._export_rule_pass) in a separate
Python process: the per-feature Python work of several passes then runs on
several cores at once, which threads sharing one interpreter can't.
Started by RulesExporter (see _run_pass_processes and _run_pass_batch):

    python pass_worker.py <job.json>

The job file holds the export extent, the expression variables and either
one pass (its base layer, rule groups as plain data, feature range and
result file) or a list of such passes ("passes"), run one after another.
Each pass's result file lists its datasets' files (or null) and its
warnings; a missing result file means the pass failed and the caller runs
it itself. Creating the job's cancel file cancels the passes.

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

    exporter = exporter_module.RulesExporter(
        [], QgsRectangle(*job["extent"]), 0, 0, job["utils_dir"], feedback,
    )
    def new_context() -> QgsExpressionContext:
        """A pass's own expression context: a pass adds its layer's scope
        and fields and leaves its last feature in it, which the next pass's
        expressions would read (a field missing from its layer was looked up
        in the previous layer's feature)."""
        context = QgsExpressionContext()
        context.appendScope(QgsExpressionContextUtils.globalScope())
        scope = QgsExpressionContextScope("project")
        for name, value in job["variables"].items():
            scope.setVariable(name, value)
        context.appendScope(scope)
        return context

    passes = job.get("passes") or [job]
    failed = 0
    for one in passes:
        if feedback.isCanceled():
            return 4
        groups = [exporter_module.group_from_dict(data) for data in one["groups"]]
        # An expression this process can't parse (a function only the
        # parent's QGIS has, e.g. from a plugin) fails the pass: the caller
        # runs it.
        if any(QgsExpression(expression).hasParserError()
               for grp in groups for expression in exporter_module.group_expressions(grp)):
            failed += 1
            continue
        del warnings[:]
        fid_range = tuple(one["fid_range"]) if one.get("fid_range") else None
        try:
            outputs = exporter._export_rule_pass(  # pylint: disable=protected-access
                groups, one["base_path"], new_context(), fid_range=fid_range,
                suffix=one.get("suffix", ""),
            )
        except Exception:  # noqa: BLE001  (the caller runs the pass itself)
            import traceback  # pylint: disable=import-outside-toplevel
            traceback.print_exc()
            failed += 1
            continue
        if feedback.isCanceled():
            return 4
        with open(one["result_path"], "w", encoding="utf-8") as f:
            json.dump({"outputs": outputs, "warnings": list(warnings)}, f)
    if failed and len(passes) == 1:
        return 3
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
