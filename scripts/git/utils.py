# Lint as: python3
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pylint: disable=missing-docstring

import argparse
import os
import subprocess


def find_git_toplevel():
  """Finds the containing git top-level directory that contains this script."""
  return execute(["git", "rev-parse", "--show-toplevel"],
                 cwd=os.path.dirname(__file__),
                 capture_output=True,
                 silent=True).stdout.strip()


def str2bool(v):
  """Can be used as an argparse type to parse a bool."""
  if v is None:
    return None
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected.")


def execute(args, cwd, capture_output=False, text=True, silent=False, **kwargs):
  """Executes a command.

  Args:
    args: List of command line arguments.
    cwd: Directory to execute in.
    capture_output: Whether to capture the output.
    text: Whether or not to treat std* as text (as opposed to binary streams).
    silent: Whether to skip logging the invocation.
    **kwargs: Extra arguments to pass to subprocess.exec

  Returns:
    A subprocess.CompletedProcess
  """
  if not silent:
    print(f"+{' '.join(args)}  [from {cwd}]")
  if capture_output:
    # TODO(#4131) python>=3.7: Use capture_output=True.
    kwargs["stdout"] = subprocess.PIPE
    kwargs["stderr"] = subprocess.PIPE
  return subprocess.run(
      args,
      cwd=cwd,
      check=True,
      # TODO(#4131) python>=3.7: Replace 'universal_newlines' with 'text'.
      universal_newlines=text,
      **kwargs)
