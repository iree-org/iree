# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared test utilities for subprocess execution.

This module provides helpers for running Python modules from tools/utils
as subprocesses with proper environment setup.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_python_module(
    module_name: str, args: list[str], **subprocess_kwargs: Any
) -> subprocess.CompletedProcess:
    """Run a Python module from tools/utils with correct environment setup.

    This helper ensures that:
    - The current Python interpreter is used (sys.executable), not hardcoded "python3"
    - PYTHONPATH is set so tools/utils modules are importable
    - Tests work from any directory (repo root or tools/utils)

    Args:
        module_name: Module to run (e.g., "lit_tools.iree_lit_extract",
                     "ci.iree_ci_triage", "common.some_tool")
        args: Command-line arguments to pass to the module
        **subprocess_kwargs: Additional arguments for subprocess.run()
                            (capture_output, text, input, timeout, etc.)

    Returns:
        CompletedProcess from subprocess.run()

    Examples:
        # Extract a test case
        result = run_python_module(
            "lit_tools.iree_lit_extract",
            [str(test_file), "--case", "2"],
            capture_output=True,
            text=True
        )

        # Run with stdin input
        result = run_python_module(
            "lit_tools.iree_lit_replace",
            [str(test_file), "--case", "2"],
            input="new content",
            capture_output=True,
            text=True
        )

        # Run CI triage tool
        result = run_python_module(
            "ci.iree_ci_triage",
            ["--pr", "12345"],
            capture_output=True
        )

    Note:
        This helper is ONLY for Python modules in tools/utils.
        For external binaries (gh, git, iree-opt), use subprocess.run() directly.
    """
    # Calculate tools/utils directory.
    # test/test_helpers.py is in tools/utils/test/, so parent.parent is tools/utils
    tools_utils_dir = Path(__file__).resolve().parent.parent

    # Prepare environment with PYTHONPATH for tools/utils directory.
    # This makes lit_tools, ci, common modules directly importable.
    env = os.environ.copy()
    pythonpath = str(tools_utils_dir)
    if "PYTHONPATH" in env:
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    # Build command with current Python interpreter.
    cmd = [sys.executable, "-m", module_name] + list(args)

    # Inject environment into subprocess kwargs.
    subprocess_kwargs["env"] = env

    return subprocess.run(cmd, **subprocess_kwargs)
