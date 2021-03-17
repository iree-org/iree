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


class TargetInfo:
  """Information of a target backend.

  Attributes:
    name: The target name used in iree-translate, e.g., vulkan-spirv.
    mako_tag: The value_key in Mako config. This will be used in Mako metric
      info, which should match to the config.
    compilation_flags: Addition compilation flags. This is useful to target
      different hardware.
    runtime_flags: Addition runtime flags. This is useful when benchmarking.
      E.g., CPU can run with single thread or multi thread.
  """

  def __init__(self,
               name,
               mako_tag,
               compilation_flags=None,
               runtime_flags=None):
    if "_" in name:
      raise ValueError("The target name contains invalid char '_'")
    if compilation_flags is None:
      compilation_flags = []
    if runtime_flags is None:
      runtime_flags = []
    self.name = name
    self.mako_tag = mako_tag
    self.compilation_flags = compilation_flags
    self.runtime_flags = runtime_flags

  def get_driver(self) -> str:
    """ Returns a string indicates the driver of the target."""
    return self.name.split("-")[0]

  def add_batch_flag(self, size):
    self.compilation_flags.append(
        f"--iree-hal-benchmark-dispatch-repeat-count={size}")
    self.runtime_flags.append(f"--batch_size={size}")


class PhoneBenchmarkInfo:
  """Information of a phone.

  Attributes:
    name: The name of the phone.
    targets: A list of TargetInfo which indicates the target config to benchmark
      on the phone.
    benchmark_key: Mako benchmark key, which can be found in config/
  """

  def __init__(self, name, targets, benchmark_key):
    if "_" in name:
      raise ValueError("The target name contains invalid char '_'")
    self.name = name
    self.targets = targets
    self.benchmark_key = benchmark_key


class ModelBenchmarkInfo:
  """Information of a model.

  Attributes:
    name: The name of the model.
    model_path: A path to MLIR input file. This can be a relative path.
    phones: A list of PhoneBenchmarkInfo that the benchmarking targets on.
  """

  def __init__(self, name, model_path, phones):
    if "_" in name:
      raise ValueError("The target name contains invalid char '_'")
    self.name = name
    self.model_path = model_path
    self.phones = phones


def get_pixel4_default_target_list(batch_config=None):
  if batch_config is None:
    batch_config = []
  targets = [
      TargetInfo(name="vmla", mako_tag="vmla"),
      TargetInfo(name="dylib-llvm-aot",
                 mako_tag="cpu",
                 compilation_flags=[
                     "--iree-llvm-target-triple=aarch64-none-linux-android29"
                 ],
                 runtime_flags=["--dylib_worker_count=1"]),
      TargetInfo(
          name="vulkan-spirv",
          mako_tag="vlk",
          compilation_flags=[
              "--iree-spirv-enable-vectorization",
              "--iree-vulkan-target-triple=qualcomm-adreno640-unknown-android10"
          ])
  ]
  for target in targets:
    if target.mako_tag in batch_config:
      target.add_batch_flag(batch_config[target.mako_tag])
  return targets


def get_s20_default_target_list(batch_config=None):
  if batch_config is None:
    batch_config = []
  targets = [
      TargetInfo(name="vmla", mako_tag="vmla"),
      TargetInfo(name="dylib-llvm-aot",
                 mako_tag="cpu",
                 compilation_flags=[
                     "--iree-llvm-target-triple=aarch64-none-linux-android29"
                 ],
                 runtime_flags=["--dylib_worker_count=1"]),
      TargetInfo(name="vulkan-spirv",
                 mako_tag="vlk",
                 compilation_flags=[
                     "--iree-spirv-enable-vectorization",
                     "--iree-vulkan-target-triple=valhall-g77-unknown-android10"
                 ])
  ]
  for target in targets:
    if target.mako_tag in batch_config:
      target.add_batch_flag(batch_config[target.mako_tag])
  return targets


# The batch numbers are roughly computed to let it benchmark more than 3
# seconds.
# Do not set batch size on Pixel 4 for GPU targets, because it will get killed
# after 2 seconds. See https://github.com/google/iree/issues/5052
MODEL_BENCHMARKS = [
    ModelBenchmarkInfo(
        name="mobile-bert",
        model_path="tmp/iree/modules/MobileBertSquad/iree_input.mlir",
        phones=[
            PhoneBenchmarkInfo(name="Pixel4",
                               benchmark_key="5538704950034432",
                               targets=get_pixel4_default_target_list(
                                   {'cpu': 8})),
            PhoneBenchmarkInfo(name="S20",
                               benchmark_key="4699630718681088",
                               targets=get_s20_default_target_list({
                                   'cpu': 8,
                                   'vlk': 16
                               })),
        ]),
    ModelBenchmarkInfo(
        name="mobilenet-v2",
        model_path="mobilenet-v2/iree_input.mlir",
        phones=[
            PhoneBenchmarkInfo(name="Pixel4",
                               benchmark_key="6338759231537152",
                               targets=get_pixel4_default_target_list(
                                   {'cpu': 16})),
            PhoneBenchmarkInfo(name="S20",
                               benchmark_key="5618403088793600",
                               targets=get_s20_default_target_list({
                                   'cpu': 16,
                                   'vlk': 64
                               })),
        ])
]


def get_module_name(model_name, phone_name, mako_tag):
  return f"{model_name}_{phone_name}_{mako_tag}.vmfb"
