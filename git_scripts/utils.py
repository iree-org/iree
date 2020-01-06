# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=missing-docstring

import argparse
import os
import subprocess


def find_git_toplevel():
  """Finds the containing git top-level directory that contains this script."""
  return execute(["git", "rev-parse", "--show-toplevel"],
                 cwd=os.path.dirname(__file__),
                 capture_output=True,
                 silent=True).strip().decode("UTF-8")


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


def execute(args, cwd, capture_output=False, silent=False, **kwargs):
  """Executes a command.

  Args:
    args: List of command line arguments.
    cwd: Directory to execute in.
    capture_output: Whether to capture the output.
    silent: Whether to skip logging the invocation.
    **kwargs: Extra arguments to pass to subprocess.exec

  Returns:
    The output if capture_output, otherwise None.
  """
  if not silent:
    print("+", " ".join(args), "  [from %s]" % cwd)
  if capture_output:
    return subprocess.check_output(args, cwd=cwd, **kwargs)
  else:
    return subprocess.check_call(args, cwd=cwd, **kwargs)
