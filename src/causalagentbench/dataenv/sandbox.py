"""The ``python`` tool: a persistent Python session per episode, in a separate, restricted worker process.

The worker runs ``WORKER`` with ``python -I`` in a fresh directory holding only ``data.csv``, with a cleared
environment. As root (inside Modal containers) it drops to the ``nobody`` user, and on Linux it gets memory
and CPU limits. The model's code is untrusted: a policy that can read answer keys will learn to, so training
and evaluation run it only in containers whose task files are unreadable to ``nobody``
(``docs/environment_v3.md``). Each request and reply is one JSON line on the worker's stdin and stdout.
"""

from __future__ import annotations

import csv
import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence

import numpy as np

CALL_SECONDS = 20.0
"""Wall-clock limit of one tool call; a call that runs longer restarts the session."""
STARTUP_SECONDS = 60.0
"""Wall-clock limit for the worker to import its libraries and load the data."""
MEMORY_BYTES = 2 * 1024**3
"""Address-space limit of the worker (Linux only)."""
CPU_SECONDS = 600
"""Total CPU-time limit of one worker over its whole episode (Linux only)."""
OUTPUT_CHARS = 2000
"""Printed output longer than this is cut in the middle."""
NOBODY = 65534
"""User and group id the worker runs as when started by root."""

WORKER = r'''
import ast, contextlib, io, json, sys, traceback
import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats

OUTPUT_CHARS = int(sys.argv[1])


def partial_corr(df, x, y, given=()):
    """Partial correlation of columns x and y given the columns in `given`, with a Fisher-z test.

    Returns a dict with the partial correlation "r", its two-sided "p_value" for the null of zero partial
    correlation, and the number of rows "n".
    """
    given = [given] if isinstance(given, str) else list(given)
    columns = [x, y, *given]
    precision = np.linalg.inv(np.corrcoef(df[columns].to_numpy(dtype=float), rowvar=False))
    r = float(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1]))
    statistic = np.sqrt(len(df) - len(given) - 3) * abs(np.arctanh(r))
    return {"r": r, "p_value": float(2 * stats.norm.sf(statistic)), "n": len(df)}


def pc(df, alpha=0.01):
    """The PC algorithm (causal-learn, stable, Fisher-z tests at level alpha) over all columns of df.

    Returns a dict: "directed" lists (cause, effect) pairs the data orient, and "undirected" lists pairs that
    are adjacent but whose direction the data cannot determine.
    """
    from causallearn.search.ConstraintBased.PC import pc as causal_learn_pc

    names = list(df.columns)
    graph = causal_learn_pc(df.to_numpy(dtype=float), alpha=alpha, stable=True, show_progress=False).G.graph
    directed, undirected = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = int(graph[i, j]), int(graph[j, i])
            if (a, b) == (-1, 1):
                directed.append((names[i], names[j]))
            elif (a, b) == (1, -1):
                directed.append((names[j], names[i]))
            elif (a, b) != (0, 0):
                undirected.append((names[i], names[j]))
    return {"directed": directed, "undirected": undirected}


def ols(df, y, X):
    """Ordinary least squares of column y on the columns in the list X, with an intercept (statsmodels).

    Returns the fitted results: use .params for coefficients, .bse for standard errors, .summary() for a table.
    """
    X = [X] if isinstance(X, str) else list(X)
    return sm.OLS(df[y], sm.add_constant(df[X], has_constant="add")).fit()


def _cut(text):
    """Keep the head and tail of long output."""
    if len(text) <= OUTPUT_CHARS:
        return text
    half = OUTPUT_CHARS // 2
    return text[:half] + f"\n... [{len(text) - OUTPUT_CHARS} characters cut] ...\n" + text[-half:]


def _run(code, namespace):
    """Execute code; if it ends in an expression, print that expression's value, as a notebook cell does."""
    tree = ast.parse(code, mode="exec")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, "<python>", "exec"), namespace)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), "<python>", "eval"), namespace)
        if value is not None:
            print(repr(value))


namespace = {"__name__": "__main__", "df": pd.read_csv("data.csv"), "np": np, "pd": pd, "scipy": scipy,
             "stats": stats, "sm": sm, "statsmodels": statsmodels, "partial_corr": partial_corr, "pc": pc,
             "ols": ols}
protocol = sys.stdout
print(json.dumps({"ready": True}), file=protocol, flush=True)
for line in sys.stdin:
    code = json.loads(line)["code"]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            _run(code, namespace)
        except BaseException:
            kind, error, trace = sys.exc_info()
            frames = [f for f in traceback.extract_tb(trace) if f.filename == "<python>"]
            print("Traceback (most recent call last):\n" + "".join(traceback.format_list(frames)), end="")
            print("".join(traceback.format_exception_only(kind, error)), end="")
    print(json.dumps({"output": _cut(buffer.getvalue())}), file=protocol, flush=True)
'''


def _restrict(directory: str) -> None:
    """In the worker before exec: resource limits on Linux, and the ``nobody`` user when started as root."""
    if sys.platform.startswith("linux"):
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(NOBODY)
        os.setuid(NOBODY)
    os.chdir(directory)


class PythonSession:
    """A persistent Python session over one dataset, in a restricted worker process."""

    def __init__(self, rows: np.ndarray, columns: Sequence[str]) -> None:
        """Write the data to a fresh directory and start the worker (it loads in the background)."""
        self.directory = tempfile.mkdtemp(prefix="episode-")
        with open(os.path.join(self.directory, "data.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows.tolist())
        if os.geteuid() == 0:
            os.chown(self.directory, NOBODY, NOBODY)
            os.chown(os.path.join(self.directory, "data.csv"), NOBODY, NOBODY)
        self.process: subprocess.Popen[str] | None = None
        self._ready = False
        self._start()

    def _start(self) -> None:
        """Launch a fresh worker."""
        environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": self.directory, "OMP_NUM_THREADS": "1"}
        environment |= {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-c", WORKER, str(OUTPUT_CHARS)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            text=True,
            preexec_fn=lambda: _restrict(self.directory),
        )
        self._ready = False

    def _read(self, seconds: float) -> dict[str, object] | None:
        """One JSON reply from the worker, or None on timeout, exit or a malformed line."""
        assert self.process is not None and self.process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=seconds):
                return None
        line = self.process.stdout.readline()
        try:
            reply = json.loads(line)
        except json.JSONDecodeError:
            return None
        return reply if isinstance(reply, dict) else None

    def _restart(self) -> None:
        """Kill the worker and start a fresh one; its variables are lost."""
        self._kill()
        self._start()

    def run(self, code: object) -> str:
        """Run ``code`` in the session and return its printed output (or the error)."""
        if not isinstance(code, str):
            return f"Error: code must be a string, got {type(code).__name__}."
        assert self.process is not None and self.process.stdin is not None
        if not self._ready:
            if self._read(STARTUP_SECONDS) != {"ready": True}:
                self._restart()
                return "Error: the Python session failed to start. Try again."
            self._ready = True
        try:
            self.process.stdin.write(json.dumps({"code": code}) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            self._restart()
            return "Error: the Python session had exited; it was restarted and its variables were lost."
        reply = self._read(CALL_SECONDS)
        if reply is None or not isinstance(reply.get("output"), str):
            self._restart()
            return (
                f"Error: the code did not finish within {CALL_SECONDS:.0f} seconds, crashed, or broke the session. "
                "The session was restarted and its variables were lost."
            )
        output = str(reply["output"])
        return output if output.strip() else "(no output)"

    def _kill(self) -> None:
        """Stop the worker if it is running and close its pipes."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        self.process = None

    def close(self) -> None:
        """Stop the worker and delete the episode directory."""
        self._kill()
        shutil.rmtree(self.directory, ignore_errors=True)
