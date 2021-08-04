# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import sys


def run():
  # No-op if we're not in a git repository.
  try:
    subprocess.check_call(['git', 'rev-parse', '--is-inside-work-tree'],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
  except:
    return

  output = os.popen("git submodule status")
  submodules = output.readlines()

  for submodule in submodules:
    if (submodule.strip()[0] == "-"):
      print(
          "The git submodule '%s' is not initialized. Please run `git submodule update --init`"
          % (submodule.split()[1]))
      sys.exit(1)


if __name__ == "__main__":
  run()
