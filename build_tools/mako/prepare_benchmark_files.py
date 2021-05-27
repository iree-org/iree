#!/usr/bin/env python3
# Copyright 2021 Google LLC
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
