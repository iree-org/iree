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

import os
import subprocess


class TargetInfo:

  def __init__(self, name, extra_flags):
    self.name = name
    self.extra_flags = extra_flags


class PhoneInfo:

  def __init__(self, name, targets):
    self.name = name
    self.targets = targets


class ModelInfo:

  def __init__(self, name, bucket_path, model_path, flagfile_path):
    self.name = name
    self.bucket_path = bucket_path
    self.model_path = model_path
    self.flagfile_path = flagfile_path


PHONES = [
    PhoneInfo(
        name="Pixel4",
        targets=[
            TargetInfo("vmla", []),
            TargetInfo(
                "dylib-llvm-aot",
                ["--iree-llvm-target-triple=aarch64-none-linux-android29"]),
            TargetInfo("vulkan-spirv", ["-iree-spirv-enable-vectorization"])
        ]),
    PhoneInfo(
        name="S20",
        targets=[
            TargetInfo("vmla", []),
            TargetInfo(
                "dylib-llvm-aot",
                ["--iree-llvm-target-triple=aarch64-none-linux-android29"]),
            TargetInfo("vulkan-spirv", [
                "-iree-spirv-enable-vectorization",
                "-iree-vulkan-target-triple=valhall-g77-unknown-android10"
            ])
        ])
]

MODELS = [
    ModelInfo(
        name="mobile-bert",
        bucket_path="gs://iree-model-artifacts/iree-mobile-bert-artifacts-6fe4616e0ab9958eb18f368960a31276f1362029.tar.gz",
        model_path="tmp/iree/modules/MobileBertSquad/iree_input.mlir",
        flagfile_path="tmp/iree/modules/MobileBertSquad/iree_vmla/traces/serving_default/flagfile"
    )
]


def main() -> None:
  IREE_TRANSLATE_PATH = "build-tracy/iree/tools/iree-translate"
  for model in MODELS:
    for phone in PHONES:
      for target in phone.targets:
        module_name = "{}_{}_{}.vmfb".format(model.name, phone.name,
                                             target.name)
        print("Generate {} ...".format(module_name))
        subprocess.run([
            IREE_TRANSLATE_PATH, model.model_path,
            "--iree-mlir-to-vm-bytecode-module", "--iree-hal-target-backends={}"
            .format(target.name), "-o", module_name
        ] + target.extra_flags)


if __name__ == "__main__":
  main()
