# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
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

  with open(
      os.path.dirname(os.path.realpath(__file__)) + "/runtime_submodules.txt",
      "r") as f:
    runtime_submodules = f.read().split("\n")

  for submodule in submodules:
    if submodule.strip()[0] == "-":
      if (args.runtime_only and
          submodule.split()[1] in runtime_submodules) or not args.runtime_only:
        print(
            "The git submodule '%s' is not initialized. Please run `git submodule update --init`"
            % (submodule.split()[1]))
        sys.exit(1)


if __name__ == "__main__":
  run()
