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
  runtime_submodules = [
      "third_party/benchmark", "third_party/cpuinfo", "third_party/flatcc",
      "third_party/googletest", "third_party/liburing", "third_party/libyaml",
      "third_party/musl", "third_party/spirv_cross",
      "third_party/spirv_headers", "third_party/tracy",
      "third_party/vulkan_headers", "third_party/vulkan_memory_allocator",
      "third_party/webgpu-headers"
  ]

  for submodule in submodules:
    if (submodule.strip()[0] == "-"):
      if (args.runtime_only and submodule.split()[1] in runtime_submodules):
        print(
            "The git submodule '%s' is not initialized. "
            "Please run `./build_tools/scripts/git/update_runtime_submodules.sh`"
            % (submodule.split()[1]))
        sys.exit(1)
      elif (not args.runtime_only):
        print(
            "The git submodule '%s' is not initialized. Please run `git submodule update --init`"
            % (submodule.split()[1]))
        sys.exit(1)


if __name__ == "__main__":
  run()
