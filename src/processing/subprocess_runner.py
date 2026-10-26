"""
subprocess_runner.py

Runs a QGIS2VectorTiles conversion in a separate Python process (worker.py)
and relays its log, progress and cancellation to a processing feedback.

Running out of process keeps QGIS responsive: the conversion's threads,
Python code and memory never compete with QGIS's interface. The child runs
at below-normal priority.

Depends on: worker.py (started as a script, never imported)
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from os.path import dirname, exists, join
from typing import Optional
from uuid import uuid4

from qgis.core import QgsApplication, QgsProcessingException, QgsProcessingUtils

_WORKER = join(dirname(__file__), "worker.py")
_MESSAGE_PREFIX = "\x1eQ2VT "            # Must match worker.MESSAGE_PREFIX.
_CANCEL_GRACE_S = 30                     # Then the child is killed.
_STDERR_LINES_KEPT = 400
_STDERR_LINES_SHOWN = 60
# Runs after the first that a native crash of the child is retried with.
_CRASH_RETRIES = 1


class _NativeCrash(Exception):
    """The child process died of a native crash (its fault report)."""


def _fault_line(lines) -> Optional[str]:
    """The child's native fault report line, if it crashed."""
    return next((line.strip() for line in lines
                 if "fatal exception" in line.lower() or "Fatal Python error" in line), None)


def _crash_report(lines) -> str:
    """What to show of the child's stderr when it ended without a result.

    After a native crash, the worker's faulthandler writes the fault and
    every thread's stack; show the fault and the crashing thread's stack
    ("Current thread"), which other threads' stacks can precede. Otherwise
    show the end of stderr.
    """
    fault = next((i for i, line in enumerate(lines)
                  if "fatal exception" in line.lower() or "Fatal Python error" in line), None)
    current = next((i for i, line in enumerate(lines) if line.startswith("Current thread")), None)
    if fault is not None and current is not None and current > fault:
        stack = []
        for line in lines[current:]:
            if not line.strip():
                break
            stack.append(line)
        return "".join([lines[fault]] + stack[:_STDERR_LINES_SHOWN]).strip()
    return "".join(lines[-_STDERR_LINES_SHOWN:]).strip()


def python_executable() -> Optional[str]:
    """The Python interpreter QGIS runs on, or None if it can't be found."""
    if os.name == "nt":
        candidates = [join(sys.prefix, "python.exe"), join(sys.prefix, "python3.exe")]
    else:
        candidates = [join(sys.prefix, "bin", "python3"), join(sys.prefix, "bin", "python")]
    return next((c for c in candidates if exists(c)), None)


def run_in_subprocess(project_path: str, package_dir: str, params: dict, feedback) -> dict:
    """Run the conversion in a child process; returns its result message.

    A native crash of the child (a fault in a library it runs, seen once
    in many runs and not reproduced) is retried, the base layers it built
    being cached by then. Raises QgsProcessingException if the child fails.
    """
    for attempt in range(_CRASH_RETRIES + 1):
        try:
            return _run_once(project_path, package_dir, params, feedback)
        except _NativeCrash as crash:
            if attempt == _CRASH_RETRIES or feedback.isCanceled():
                raise QgsProcessingException(str(crash)) from None
            feedback.pushInfo(f". The conversion process stopped unexpectedly ({crash.args[1]}); "
                              "running it again...")
    raise QgsProcessingException("The conversion process failed.")


def _run_once(project_path: str, package_dir: str, params: dict, feedback) -> dict:
    """One run of the conversion in a child process; returns its result
    message. Raises _NativeCrash if the child crashed, QgsProcessingException
    if it failed otherwise."""
    python = python_executable()
    if python is None:
        raise QgsProcessingException("QGIS's Python interpreter was not found.")

    work_dir = join(QgsProcessingUtils.tempFolder(), f"q2vt_job_{uuid4().hex}")
    os.makedirs(work_dir, exist_ok=True)
    job_path = join(work_dir, "job.json")
    cancel_file = join(work_dir, "cancel")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump({
            "plugins_dir": dirname(package_dir),
            "package": os.path.basename(package_dir),
            "prefix_path": QgsApplication.prefixPath(),
            "profile_dir": QgsApplication.qgisSettingsDirPath(),
            "project_path": project_path,
            "cancel_file": cancel_file,
            "params": params,
        }, f)

    env = os.environ.copy()
    # Let the child import qgis/processing exactly as QGIS's interpreter does.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONIOENCODING"] = "utf-8"
    kwargs = {}
    if os.name == "nt":
        # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
        kwargs["creationflags"] = 0x08000000 | 0x00004000
    else:
        kwargs["preexec_fn"] = lambda: os.nice(5)

    process = subprocess.Popen(
        [python, "-u", _WORKER, job_path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=work_dir,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )

    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    stderr_tail: deque = deque(maxlen=_STDERR_LINES_KEPT)

    def read_stdout():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)  # End of output.

    def read_stderr():
        for line in process.stderr:
            stderr_tail.append(line)

    threading.Thread(target=read_stdout, daemon=True).start()
    stderr_reader = threading.Thread(target=read_stderr, daemon=True)
    stderr_reader.start()

    result: Optional[dict] = None
    fatal_errors = []
    cancel_sent_at = None
    stdout_done = False
    while not stdout_done or process.poll() is None:
        if feedback.isCanceled() and cancel_sent_at is None:
            open(cancel_file, "w").close()
            cancel_sent_at = time.monotonic()
            feedback.pushInfo("Cancelling...")
        if cancel_sent_at is not None and time.monotonic() - cancel_sent_at > _CANCEL_GRACE_S:
            process.kill()
        try:
            line = lines.get(timeout=0.3)
        except queue.Empty:
            continue
        if line is None:
            stdout_done = True
            continue
        if not line.startswith(_MESSAGE_PREFIX):
            continue  # Ordinary output (e.g. library prints).
        message = json.loads(line[len(_MESSAGE_PREFIX):])
        kind = message.pop("type")
        if kind == "result":
            result = message
        elif kind == "info":
            feedback.pushInfo(message["text"])
        elif kind == "warning":
            feedback.pushWarning(message["text"])
        elif kind == "debug":
            pass  # Not shown: the log keeps to the steps, warnings and errors.
        elif kind == "progress":
            feedback.setProgress(message["value"])
        elif kind == "progress_text":
            feedback.setProgressText(message["text"])
        elif kind == "error":
            if message.get("fatal"):
                fatal_errors.append(message["text"])
            else:
                feedback.reportError(message["text"])

    process.wait()
    process.stdout.close()
    stderr_reader.join(timeout=5)
    process.stderr.close()
    shutil.rmtree(work_dir, ignore_errors=True)
    if result is not None:
        return result
    if feedback.isCanceled():
        return {"temp_dir": None, "min_zoom": None, "canceled": True}
    details = "\n".join(fatal_errors) or _crash_report(list(stderr_tail)) or "no output"
    message = f"The conversion process failed (exit code {process.returncode}):\n{details}"
    fault = None if fatal_errors else _fault_line(list(stderr_tail))
    if fault:
        raise _NativeCrash(message, fault)
    raise QgsProcessingException(message)
