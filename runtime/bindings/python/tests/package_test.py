# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Creates a venv and installs the runtime package, doing some smoke tests.

This is not a perfect approximation of how the runtime package is built
for real since it is run from the build dir and does not do a recursive
invocation of CMake. However, it can detect gross errors in the installation
process, including missing modules, scripts, etc.
"""

import os
import subprocess
import sys
import tempfile
from typing import List

SETUP_PY_DIR = sys.argv[1]
if not os.path.exists(os.path.join(SETUP_PY_DIR, "setup.py")):
  print("ERROR: Expected directory containing setup.py as argument")
print(f"Using setup.py directory: {SETUP_PY_DIR}")

# Figure out where to stage output.
TEMP_DIR = os.getenv("TEST_TMPDIR")
if not TEMP_DIR:
  TEMP_DIR = tempfile.gettempdir()

# Create the venv.
VENV_DIR = os.path.join(TEMP_DIR, "iree_runtime_venv")
print(f"Using venv directory: {VENV_DIR}")
subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
venv_python = None
for venv_bin in [
    os.path.join(VENV_DIR, "bin"),  # Posix.
    os.path.join(VENV_DIR, "Scripts")  # Windows.
]:
  if os.path.exists(os.path.join(venv_bin, "activate")):
    venv_python = os.path.join(venv_bin, "python")
if not venv_python:
  print("ERROR: Could not find venv python")
venv_bin = os.path.dirname(venv_python)
print(f"Running with python: {venv_python}")

# Install the package.
subprocess.check_call([
    venv_python, "-m", "pip", "install", "--force-reinstall", SETUP_PY_DIR + "/"
])

# Run some sanity checks.
if "PYTHONPATH" in os.environ:
  del os.environ["PYTHONPATH"]

print("***** Sanity checking that module loads...")
subprocess.check_call(
    [venv_python, "-c", "import iree.runtime; print('Runtime loaded')"],
    cwd=venv_bin)


# Check tools.
def check_tool(tool_name: str, args: List[str]):
  print(f"**** Checking tool {tool_name} with args {args}")
  subprocess.check_call([os.path.join(venv_bin, tool_name)] + args,
                        cwd=venv_bin)


check_tool("iree-benchmark-module", ["--help"])
check_tool("iree-benchmark-trace", ["--help"])
check_tool("iree-run-module", ["--help"])
check_tool("iree-run-trace", ["--help"])

print("***** All done *****")
