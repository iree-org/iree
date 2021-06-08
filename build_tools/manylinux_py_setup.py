#!/opt/python/cp38-cp38/bin/python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates CMake arguments to build all manylinux python versions.

manylinux containers have all python version linked under /opt/python.
This script scrapes them to get configuration, install deps, etc.

Usage:
  Install dependencies:
    manylinux_py_setup.py deps
  Get CMake arguments to build (typically via $() expansion):
    manylinux_py_setup.py args
"""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def _get_python_exes():
  PYTHON_PARENT_PATH = Path("/opt/python")
  return PYTHON_PARENT_PATH.glob("*/bin/python")


def install_deps():
  for python_exe in _get_python_exes():
    args = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "absl-py",
        "numpy",
    ]
    print("EXEC:", " ".join(args))
    subprocess.run(args, check=True)


def dump_current(identifier):
  print("-DIREE_MULTIPY_{}_EXECUTABLE='{}'".format(identifier, sys.executable))
  print("-DIREE_MULTIPY_{}_INCLUDE_DIRS='{}'".format(
      identifier, sysconfig.get_config_var("INCLUDEPY")))
  # TODO: Print LIBRARIES for Windows and OSX
  print("-DIREE_MULTIPY_{}_EXTENSION='{}'".format(
      identifier, sysconfig.get_config_var("EXT_SUFFIX")))


def dump_all():
  versions_ids = []
  for python_exe in _get_python_exes():
    identifier = python_exe.parent.parent.name
    versions_ids.append(identifier)
    # Invoke ourselves with a different interpreter/args to dump config.
    subprocess.run([str(python_exe), __file__, "_current_args", identifier],
                   check=True)
  print("-DIREE_MULTIPY_VERSIONS='{}'".format(";".join(versions_ids)))


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("SYNTAX: mainlinux_py_setup.py {deps|args}")
    sys.exit(1)
  command = sys.argv[1]
  if command == "args":
    dump_all()
  elif command == "_current_args":
    dump_current(sys.argv[2])
  elif command == "deps":
    install_deps()
  else:
    print("Unexpected command")
    sys.exit(1)
