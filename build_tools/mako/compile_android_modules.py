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
  """Information of a target backend.

  Attributes:
    name: The target name used in iree-translate, e.g., vulkan-spirv.
    mako_tag: The value_key in Mako config. This will be used in Mako metric
      info, which should match to the config.
    extra_flags: Addition compilation flags. This is useful to target different
      hardware.
  """

  def __init__(self, name, mako_tag, extra_flags=[]):
    self.name = name
    self.mako_tag = mako_tag
    self.extra_flags = extra_flags


class PhoneInfo:
  """Information of a phone.

  Attributes:
    name: The name of the phone.
    targets: A list of TargetInfo which indicates the target config to benchmark
      on the phone.
  """

  def __init__(self, name, targets):
    self.name = name
    self.targets = targets


class ModelInfo:
  """Information of a model.

  Attributes:
    name: The name of the model.
    model_path: A path to MLIR input file. This can be a relative path.
  """

  def __init__(self, name, model_path):
    self.name = name
    self.model_path = model_path


PHONES = [
    PhoneInfo(
        name="Pixel4",
        targets=[
            TargetInfo(name="vmla", mako_tag="vmla"),
            TargetInfo(
                name="dylib-llvm-aot",
                mako_tag="cpu",
                extra_flags=[
                    "--iree-llvm-target-triple=aarch64-none-linux-android29"
                ]),
            TargetInfo(
                name="vulkan-spirv",
                mako_tag="vlk",
                extra_flags=[
                    "--iree-spirv-enable-vectorization",
                    "--iree-vulkan-target-triple=qualcomm-adreno640-unknown-android10"
                ])
        ]),
    PhoneInfo(
        name="S20",
        targets=[
            TargetInfo(name="vmla", mako_tag="vmla"),
            TargetInfo(
                name="dylib-llvm-aot",
                mako_tag="cpu",
                extra_flags=[
                    "--iree-llvm-target-triple=aarch64-none-linux-android29"
                ]),
            TargetInfo(
                name="vulkan-spirv",
                mako_tag="vlk",
                extra_flags=[
                    "--iree-spirv-enable-vectorization",
                    "--iree-vulkan-target-triple=valhall-g77-unknown-android10"
                ])
        ])
]

MODELS = [
    ModelInfo(
        name="mobile-bert",
        model_path="tmp/iree/modules/MobileBertSquad/iree_input.mlir",
    ),
    ModelInfo(
        name="mobilenet-v2",
        model_path="mobilenet-v2/iree_input.mlir",
    )
]


def main() -> None:
  IREE_TRANSLATE_PATH = "build-host/iree/tools/iree-translate"
  for model in MODELS:
    for phone in PHONES:
      for target in phone.targets:
        module_name = "{}_{}_{}_{}.vmfb".format(model.name, phone.name,
                                                target.name, target.mako_tag)
        if module_name.count("_") != 3:
          raise ValueError(
              "Expect model name, phone name and target name do not contain '_'"
          )
        print("Generating {} ...".format(module_name))
        subprocess.run([
            IREE_TRANSLATE_PATH, model.model_path,
            "--iree-mlir-to-vm-bytecode-module", "--iree-hal-target-backends={}"
            .format(target.name), "-o", module_name
        ] + target.extra_flags)


if __name__ == "__main__":
  main()
