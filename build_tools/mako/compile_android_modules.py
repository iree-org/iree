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
