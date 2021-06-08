#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class TargetInfo:
  """Information of a target backend.

  Attributes:
    driver: The driver used in iree-benchmark-module, e.g., vulkan.
    hal_target_backend: The target name used in iree-translate, e.g., vulkan-spirv.
    taskset: The value used for taskset when benchmarking the IREE module.
    mako_tag: The value_key in Mako config. This will be used in Mako metric
      info, which should match to the config.
    compilation_flags: Addition compilation flags. This is useful to target
      different hardware.
    runtime_flags: Addition runtime flags. This is useful when benchmarking.
      E.g., CPU can run with single thread or multi thread.
  """

  def __init__(self,
               driver,
               hal_target_backend,
               taskset,
               mako_tag,
               compilation_flags=None,
               runtime_flags=None):
    if compilation_flags is None:
      compilation_flags = []
    if runtime_flags is None:
      runtime_flags = []
    self.driver = driver
    self.hal_target_backend = hal_target_backend
    self.taskset = taskset
    self.mako_tag = mako_tag
    # This setup is going to be deprecated. Now we only test mhlo inputs for
    # Mako setup, hardcoded would make it eaiser.
    self.compilation_flags = compilation_flags + ['--iree-input-type=mhlo']
    self.runtime_flags = runtime_flags

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
    model_artifacts_name: The filename of model artifacts which locates in
      Google bucket iree-model-artifacts.
    model_path: A path to MLIR input file. This can be a relative path.
    flagfile_path: A path to flagfile. This can be a relative path.
    phones: A list of PhoneBenchmarkInfo that the benchmarking targets on.
  """

  def __init__(self, name, model_artifacts_name, model_path, flagfile_path,
               phones):
    if "_" in name:
      raise ValueError("The target name contains invalid char '_'")
    self.name = name
    self.model_artifacts_name = model_artifacts_name
    self.model_path = model_path
    self.flagfile_path = flagfile_path
    self.phones = phones


def get_pixel4_default_target_list(skipped_target=None,
                                   batch_config=None,
                                   compilation_flags=None):
  if skipped_target is None:
    skipped_target = []
  if batch_config is None:
    batch_config = []
  if compilation_flags is None:
    compilation_flags = []
  targets = [
      TargetInfo(driver="dylib-sync",
                 hal_target_backend="dylib-llvm-aot",
                 taskset="80",
                 mako_tag="cpu",
                 compilation_flags=[
                     "--iree-llvm-target-triple=aarch64-none-linux-android29",
                     "--iree-flow-inline-constants-max-byte-length=2048",
                 ]),
      TargetInfo(driver="dylib",
                 hal_target_backend="dylib-llvm-aot",
                 taskset="f0",
                 mako_tag="cpu3t",
                 compilation_flags=[
                     "--iree-llvm-target-triple=aarch64-none-linux-android29",
                     "--iree-flow-inline-constants-max-byte-length=2048",
                 ],
                 runtime_flags=[
                     "--task_topology_group_count=3",
                 ]),
      TargetInfo(driver="vmvx",
                 hal_target_backend="vmvx",
                 taskset="f0",
                 mako_tag="vmvx3t",
                 compilation_flags=[
                     "--iree-flow-inline-constants-max-byte-length=2048",
                 ],
                 runtime_flags=[
                     "--task_topology_group_count=3",
                 ]),
      TargetInfo(driver="vulkan",
                 hal_target_backend="vulkan-spirv",
                 taskset="80",
                 mako_tag="vlk",
                 compilation_flags=[
                     "--iree-vulkan-target-triple=adreno-a640-android11",
                     "--iree-flow-inline-constants-max-byte-length=2048",
                     "--iree-flow-dispatch-formation-enable-operand-fusion",
                     "--iree-enable-fusion-with-reduction-ops",
                 ])
  ]
  targets = [elem for elem in targets if elem.mako_tag not in skipped_target]
  for target in targets:
    if target.mako_tag in batch_config:
      target.add_batch_flag(batch_config[target.mako_tag])
    if target.mako_tag in compilation_flags:
      target.compilation_flags += compilation_flags[target.mako_tag]
  return targets


def get_s20_default_target_list(skipped_target=None,
                                batch_config=None,
                                compilation_flags=None):
  if skipped_target is None:
    skipped_target = []
  if batch_config is None:
    batch_config = []
  if compilation_flags is None:
    compilation_flags = []
  targets = [
      TargetInfo(driver="dylib-sync",
                 hal_target_backend="dylib-llvm-aot",
                 taskset="80",
                 mako_tag="cpu",
                 compilation_flags=[
                     "--iree-llvm-target-triple=aarch64-none-linux-android29",
                     "--iree-flow-inline-constants-max-byte-length=2048",
                 ]),
      TargetInfo(driver="dylib",
                 hal_target_backend="dylib-llvm-aot",
                 taskset="f0",
                 mako_tag="cpu3t",
                 compilation_flags=[
                     "--iree-llvm-target-triple=aarch64-none-linux-android29",
                     "--iree-flow-inline-constants-max-byte-length=2048",
                 ],
                 runtime_flags=[
                     "--task_topology_group_count=3",
                 ]),
      TargetInfo(driver="vmvx",
                 hal_target_backend="vmvx",
                 taskset="f0",
                 mako_tag="vmvx3t",
                 compilation_flags=[
                     "--iree-flow-inline-constants-max-byte-length=2048",
                 ],
                 runtime_flags=[
                     "--task_topology_group_count=3",
                 ]),
      TargetInfo(
          driver="vulkan",
          hal_target_backend="vulkan-spirv",
          taskset="80",
          mako_tag="vlk",
          compilation_flags=[
              "--iree-vulkan-target-triple=valhall-g77-android11",
              # TODO(GH-5330): Revisit the number or delete the flag.
              "--iree-flow-inline-constants-max-byte-length=16",
              "--iree-flow-dispatch-formation-enable-operand-fusion"
          ])
  ]
  targets = [elem for elem in targets if elem.mako_tag not in skipped_target]
  for target in targets:
    if target.mako_tag in batch_config:
      target.add_batch_flag(batch_config[target.mako_tag])
    if target.mako_tag in compilation_flags:
      target.compilation_flags += compilation_flags[target.mako_tag]
  return targets


# The batch numbers are roughly computed to let it benchmark more than 3
# seconds.
# Do not set batch size on Pixel 4 for GPU targets, because it will get killed
# after 2 seconds. See https://github.com/google/iree/issues/5052
MODEL_BENCHMARKS = [
    ModelBenchmarkInfo(
        name="mobile-bert",
        model_artifacts_name=
        "iree-mobile-bert-artifacts-6fe4616e0ab9958eb18f368960a31276f1362029.tar.gz",
        model_path="tmp/iree/modules/MobileBertSquad/iree_input.mlir",
        flagfile_path=
        "tmp/iree/modules/MobileBertSquad/iree_vmla/traces/serving_default/flagfile",
        phones=[
            PhoneBenchmarkInfo(name="Pixel4",
                               benchmark_key="5538704950034432",
                               targets=get_pixel4_default_target_list(
                                   skipped_target=["cpu2", "vmvx3t", "vlk2"],)),
            PhoneBenchmarkInfo(name="S20",
                               benchmark_key="4699630718681088",
                               targets=get_s20_default_target_list(
                                   skipped_target=["cpu2", "vmvx3t", "vlk2"],)),
        ]),
    ModelBenchmarkInfo(
        name="mobilenet-v2",
        model_artifacts_name="mobilenet-v2.tar.gz",
        model_path="mobilenet-v2/iree_input.mlir",
        flagfile_path="mobilenet-v2/flagfile",
        phones=[
            PhoneBenchmarkInfo(
                name="Pixel4",
                benchmark_key="6338759231537152",
                targets=get_pixel4_default_target_list(
                    skipped_target=["vlk2"],
                    compilation_flags={
                        'cpu': [
                            # TODO(GH-5857): Enable this after fixing segfault.
                            #"--iree-flow-dispatch-formation-enable-operand-fusion",
                            "-iree-llvm-loop-unrolling=true"
                        ],
                        'cpu3t': [
                            # TODO(GH-5857): Enable this after fixing segfault.
                            #"--iree-flow-dispatch-formation-enable-operand-fusion",
                            "-iree-llvm-loop-unrolling=true"
                        ]
                    })),
            PhoneBenchmarkInfo(
                name="S20",
                benchmark_key="5618403088793600",
                targets=get_s20_default_target_list(
                    compilation_flags={
                        'cpu': [
                            # TODO(GH-5857): Enable this after fixing segfault.
                            #"--iree-flow-dispatch-formation-enable-operand-fusion",
                            "-iree-llvm-loop-unrolling=true"
                        ],
                        'cpu3t': [
                            # TODO(GH-5857): Enable this after fixing segfault.
                            #"--iree-flow-dispatch-formation-enable-operand-fusion",
                            "-iree-llvm-loop-unrolling=true"
                        ]
                    })),
        ]),
    ModelBenchmarkInfo(
        name="mobilebert-f16",
        model_artifacts_name="mobilebert-f16.tar.gz",
        model_path="mobilebert-f16/mobilebert-f16.mlir",
        flagfile_path="mobilebert-f16/flagfile",
        phones=[
            PhoneBenchmarkInfo(
                name="S20",
                benchmark_key="4636549841944576",
                targets=get_s20_default_target_list(
                    skipped_target=['cpu', 'vmvx3t', 'cpu2', 'vlk2'])),
        ])
]


def get_flagfile_name(model_name):
  return f"{model_name}_flagfile"


def get_module_name(model_name, phone_name, mako_tag):
  return f"{model_name}_{phone_name}_{mako_tag}.vmfb"
