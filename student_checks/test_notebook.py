"""Verify the notebook displays child stdout/stderr and preserves failures."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def notebook_runner():
    path = Path(__file__).resolve().parents[1]/"lab_day10.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in nb["cells"]
                  if c["cell_type"] == "code" and "def run_cmd(" in "".join(c["source"]))
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "run_cmd")
    namespace = {"os": os, "subprocess": subprocess}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["run_cmd"]


def test_child_logs_are_printed_in_cell(capsys):
    result = notebook_runner()(sys.executable, "-c", "import sys; print('stdout-marker'); print('stderr-marker', file=sys.stderr)")
    assert result.returncode == 0
    output = capsys.readouterr().out
    assert "stdout-marker\n" in output
    assert "stderr-marker\n" in output


def test_child_failure_stops_cell_and_can_be_inspected(capsys):
    run_cmd = notebook_runner()
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_cmd(sys.executable, "-c", "import sys; print('failure-marker'); sys.exit(3)")
    assert error.value.returncode == 3
    assert "failure-marker\n" in capsys.readouterr().out
    assert run_cmd(sys.executable, "-c", "import sys; sys.exit(3)", check=False).returncode == 3
