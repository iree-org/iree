# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import pathlib
import subprocess
import sys


def run():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--runtime_only",
      help=("Only check the initialization of the submodules for the"
            "runtime-dependent submodules. Default: False"),
      action="store_true",
      default=False)
  args = parser.parse_args()
  # No-op if we're not in a git repository.
  try:
    subprocess.check_call(['git', 'rev-parse', '--is-inside-work-tree'],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
  except:
    return

  output = os.popen("git submodule status")
  submodules = output.readlines()

  runtime_submodules = pathlib.Path(__file__).with_name(
      "runtime_submodules.txt").read_text().split("\n")

  for submodule in submodules:
    prefix = submodule.strip()[0]
    name = submodule.split()[1]
    if prefix == "-" and (not args.runtime_only or name in runtime_submodules):
      print(
          "The git submodule '%s' is not initialized. Please run `git submodule update --init`"
          % (name))
      sys.exit(1)


if __name__ == "__main__":
  run()
