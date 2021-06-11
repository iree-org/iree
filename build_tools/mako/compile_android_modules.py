#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Compiles the given model (which is configured in configuration.py) to modules.
#
# The scripts is used for benchmarking automation, and it assumes:
#   1) ANDROID_NDK env is set.
#   2) IREE is built for the host in `build-host`, e.g. build with
#      build_tools/cmake/build_android.sh script.

import subprocess

import configuration

IREE_TRANSLATE_PATH = "build-host/iree/tools/iree-translate"


def main() -> None:
  for model_benchmark in configuration.MODEL_BENCHMARKS:
    for phone in model_benchmark.phones:
      for target in phone.targets:
        module_name = configuration.get_module_name(model_benchmark.name,
                                                    phone.name, target.mako_tag)
        print(f"Generating {module_name} ...")
        subprocess.run(args=[
            IREE_TRANSLATE_PATH, model_benchmark.model_path,
            "--iree-mlir-to-vm-bytecode-module",
            f"--iree-hal-target-backends={target.hal_target_backend}", "-o",
            module_name
        ] + target.compilation_flags,
                       check=True)


if __name__ == "__main__":
  main()
