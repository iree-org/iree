# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys


def run():
  # Checking the "git status"
  git_check = os.popen("git status")
  git_status = git_check.readline()

  # no-op if not a git repository
  if not git_status.startswith("On"):
    sys.exit(1)
  else:
    output = os.popen("git submodule status")
    submodules = output.readlines()

    for submodule in submodules:
      if (submodule.strip()[0] == "-"):
        print(
            f"The git submodule '{submodule.split()[1]}' is not initialized. Please run `git submodule update --init`"
        )
        sys.exit(1)


if __name__ == "__main__":
  run()
