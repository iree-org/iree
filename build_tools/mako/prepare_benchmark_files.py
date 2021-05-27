#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Prepares files for
#   * compile_android_modules.py
#   * benchmark_modules_on_android.py
#
# The script assumes model artifacts are in google bucket and they are zipped in
# tar.gz format.

import subprocess

import configuration


def main() -> None:
  for model_benchmark in configuration.MODEL_BENCHMARKS:
    print(f"Preparing benchmark files for {model_benchmark.name}")
    subprocess.run(args=[
        "gsutil", "cp",
        f"gs://iree-model-artifacts/{model_benchmark.model_artifacts_name}", "."
    ],
                   check=True)
    subprocess.run(args=["tar", "-xvf", model_benchmark.model_artifacts_name],
                   check=True)
    subprocess.run(args=[
        "cp", model_benchmark.flagfile_path,
        configuration.get_flagfile_name(model_benchmark.name)
    ],
                   check=True)


if __name__ == "__main__":
  main()
