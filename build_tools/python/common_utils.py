# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
from typing import Sequence


def execute_cmd(args: Sequence[str],
                verbose: bool = False,
                **kwargs) -> subprocess.CompletedProcess:
  """Executes a command and returns the completed process.

  A thin wrapper around subprocess.run that sets some useful defaults and
  optionally prints out the command being run.

  Raises:
    CalledProcessError if the command fails.
  """
  cmd = " ".join(args)
  if verbose:
    print(f"cmd: {cmd}")
  try:
    return subprocess.run(args, check=True, text=True, **kwargs)
  except subprocess.CalledProcessError as exc:
    print((f"\n\nThe following command failed:\n\n{cmd}"
           f"\n\nReturn code: {exc.returncode}\n\n"))
    if exc.stdout:
      print(f"Stdout:\n\n{exc.stdout}\n\n")
    if exc.stderr:
      print(f"Stderr:\n\n{exc.stderr}\n\n")
    raise exc


def execute_cmd_and_get_output(args: Sequence[str],
                               verbose: bool = False,
                               **kwargs) -> str:
  """Executes a command and returns its stdout.

  Same as execute_cmd except captures stdout (and not stderr).
  """
  return execute_cmd(args, verbose=verbose, stdout=subprocess.PIPE,
                     **kwargs).stdout.strip()


def get_git_commit_hash(commit: str) -> str:
  return execute_cmd_and_get_output(['git', 'rev-parse', commit],
                                    cwd=os.path.dirname(
                                        os.path.realpath(__file__)))
