## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE benchmarks."""

from itertools import product
from typing import List, Tuple
from .device_specs.linux_x86_64 import GCP_C2_STANDARD_16_DEVICE_SPEC
from .models.model_group import MOBILE_MODEL_GROUP
from .definitions.common import DeviceArchitecture, DevicePlatform, RANDOM_MODEL_INPUT_DATA
from .definitions.iree import (IreeBenchmarkCompileSpec, IreeBenchmarkRunSpec,
                               IreeCompileTarget, IreeCompileConfig,
                               IreeRunConfig, IreeRuntimeDriver,
                               IreeRuntimeLoader, IreeTargetBackend)
from .id_defs import (IREE_COMPILE_CONFIG_LINUX_CASCADELAKE_ID,
                      IREE_RUN_CONFIG_LOCAL_SYNC_ID,
                      IREE_RUN_CONFIG_LOCAL_TASK_ID_BASE)

MODULE_BENCHMARK_TOOL = "iree-benchmark-module"


class Linux_x86_64_Benchmarks(object):
  """Benchmarks on x86_64 linux devices."""

  CASCADELAKE_CPU_TARGET = IreeCompileTarget(
      target_architecture=DeviceArchitecture.X86_64_CASCADELAKE,
      target_platform=DevicePlatform.LINUX_GNU,
      target_backend=IreeTargetBackend.LLVM_CPU)

  CASCADELAKE_COMPILE_CONFIG = IreeCompileConfig(
      id=IREE_COMPILE_CONFIG_LINUX_CASCADELAKE_ID,
      tags=["default-flags"],
      compile_targets=[CASCADELAKE_CPU_TARGET])

  LOCAL_SYNC_RUN_CONFIG = IreeRunConfig(
      id=IREE_RUN_CONFIG_LOCAL_SYNC_ID,
      tags=["full-inference", "default-flags"],
      loader=IreeRuntimeLoader.EMBEDDED_ELF,
      driver=IreeRuntimeDriver.LOCAL_SYNC,
      benchmark_tool=MODULE_BENCHMARK_TOOL)

  @classmethod
  def generate(
      cls) -> Tuple[List[IreeBenchmarkCompileSpec], List[IreeBenchmarkRunSpec]]:
    # Generate default run configs.
    default_run_configs = [cls.LOCAL_SYNC_RUN_CONFIG]
    for thread_num in [1, 4, 8]:
      default_run_configs.append(
          IreeRunConfig(
              id=f"{IREE_RUN_CONFIG_LOCAL_TASK_ID_BASE}_{thread_num}",
              tags=[f"{thread_num}-thread", "full-inference", "default-flags"],
              loader=IreeRuntimeLoader.EMBEDDED_ELF,
              driver=IreeRuntimeDriver.LOCAL_TASK,
              benchmark_tool=MODULE_BENCHMARK_TOOL,
              extra_flags=[f"--task_topology_group_count={thread_num}"]))
    # Generate compile specs for mobile models.
    mobile_model_compile_specs = list(
        IreeBenchmarkCompileSpec(compile_config=cls.CASCADELAKE_COMPILE_CONFIG,
                                 model=model) for model in MOBILE_MODEL_GROUP)
    # Generate run specs for mobile models.
    mobile_model_run_specs = []
    for compile_spec, run_config in product(mobile_model_compile_specs,
                                            default_run_configs):
      mobile_model_run_specs.append(
          IreeBenchmarkRunSpec(
              compile_spec=compile_spec,
              run_config=run_config,
              target_device_spec=GCP_C2_STANDARD_16_DEVICE_SPEC,
              input_data=RANDOM_MODEL_INPUT_DATA))

    return (mobile_model_compile_specs, mobile_model_run_specs)
