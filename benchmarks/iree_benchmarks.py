## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Tuple
from benchmarks.device_library import GCP_C2_STANDARD_16_DEVICE_SPEC
from benchmarks.model_library import MOBILENET_V2_MODEL
from benchmarks.definitions.common import DeviceArchitecture, DevicePlatform, ZEROS_MODEL_INPUT_DATA
from benchmarks.definitions.iree import IreeBenchmarkCompileSpec, IreeBenchmarkRunSpec, IreeCompileTarget, IreeCompileConfig, IreeRunConfig, IreeRuntimeDriver, IreeRuntimeLoader, IreeTargetBackend


class Linux_x86_64_Benchmarks(object):
  CASCADELAKE_CPU_TARGET = IreeCompileTarget(
      target_architecture=DeviceArchitecture.X86_64_CASCADELAKE,
      target_platform=DevicePlatform.LINUX_GNU,
      target_backend=IreeTargetBackend.LLVM_CPU)

  CASCADELAKE_TOSA_DEFAULT_COMPILE_CONFIG = IreeCompileConfig(
      id="abcd",
      tags=["defaults"],
      compile_targets=[CASCADELAKE_CPU_TARGET],
      extra_flags=["--iree-input-type=tosa"])

  ELF_LOCAL_TASK_8_CORES_RUN_CONFIG = IreeRunConfig(
      id="1234",
      tags=["8-cores,defaults"],
      loader=IreeRuntimeLoader.EMBEDDED_ELF,
      driver=IreeRuntimeDriver.LOCAL_TASK,
      benchmark_tool="iree-benchmark-module",
      extra_flags=["--task_topology_group_count=8"])

  MOBILENET_V2_COMPILE_SPEC = IreeBenchmarkCompileSpec(
      compile_config=CASCADELAKE_TOSA_DEFAULT_COMPILE_CONFIG,
      model=MOBILENET_V2_MODEL)

  MOBILENET_V2_8_CORES_RUN_SPEC = IreeBenchmarkRunSpec(
      compile_spec=MOBILENET_V2_COMPILE_SPEC,
      run_config=ELF_LOCAL_TASK_8_CORES_RUN_CONFIG,
      target_device_spec=GCP_C2_STANDARD_16_DEVICE_SPEC,
      input_data=ZEROS_MODEL_INPUT_DATA)

  @classmethod
  def generate(
      cls) -> Tuple[List[IreeBenchmarkCompileSpec], List[IreeBenchmarkRunSpec]]:
    return ([cls.MOBILENET_V2_COMPILE_SPEC],
            [cls.MOBILENET_V2_8_CORES_RUN_SPEC])


def generate_iree_benchmarks(
) -> Tuple[List[IreeBenchmarkCompileSpec], List[IreeBenchmarkRunSpec]]:
  return Linux_x86_64_Benchmarks.generate()
