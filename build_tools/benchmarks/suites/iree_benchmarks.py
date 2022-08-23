## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE benchmarks."""

import itertools
from typing import List, Tuple

from .device_specs import linux_x86_64_specs
from .models import model_groups
from .definitions import common_definitions, iree_definitions
from . import unique_ids

MODULE_BENCHMARK_TOOL = "iree-benchmark-module"


class Linux_x86_64_Benchmarks(object):
  """Benchmarks on x86_64 linux devices."""

  CASCADELAKE_CPU_TARGET = iree_definitions.CompileTarget(
      target_architecture=common_definitions.DeviceArchitecture.
      X86_64_CASCADELAKE,
      target_platform=common_definitions.DevicePlatform.LINUX_GNU,
      target_backend=iree_definitions.TargetBackend.LLVM_CPU)

  CASCADELAKE_COMPILE_CONFIG = iree_definitions.CompileConfig(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CASCADELAKE,
      tags=["default-flags"],
      compile_targets=[CASCADELAKE_CPU_TARGET])

  @classmethod
  def generate(
      cls
  ) -> Tuple[List[iree_definitions.BenchmarkCompileSpec],
             List[iree_definitions.BenchmarkRunSpec]]:
    """Generates IREE compile and run specs."""

    default_run_configs = cls._generate_default_run_configs()

    # Generate compile specs for mobile models.
    mobile_model_compile_specs = [
        iree_definitions.BenchmarkCompileSpec(
            compile_config=cls.CASCADELAKE_COMPILE_CONFIG, model=model)
        for model in model_groups.MOBILE
    ]

    # Generate run specs for mobile models.
    mobile_model_run_specs = []
    for compile_spec, run_config in itertools.product(
        mobile_model_compile_specs, default_run_configs):
      mobile_model_run_specs.append(
          iree_definitions.BenchmarkRunSpec(
              compile_spec=compile_spec,
              run_config=run_config,
              target_device_spec=linux_x86_64_specs.GCP_C2_STANDARD_16,
              input_data=common_definitions.RANDOM_MODEL_INPUT_DATA))

    return (mobile_model_compile_specs, mobile_model_run_specs)

  @staticmethod
  def _generate_default_run_configs() -> List[iree_definitions.RunConfig]:
    run_configs = [
        # Local-sync run config.
        iree_definitions.RunConfig(
            id=unique_ids.IREE_RUN_CONFIG_LOCAL_SYNC,
            tags=["full-inference", "default-flags"],
            loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
            driver=iree_definitions.RuntimeDriver.LOCAL_SYNC,
            benchmark_tool=MODULE_BENCHMARK_TOOL)
    ]
    for thread_num in [1, 4, 8]:
      run_configs.append(
          iree_definitions.RunConfig(
              id=f"{unique_ids.IREE_RUN_CONFIG_LOCAL_TASK_BASE}_{thread_num}",
              tags=[f"{thread_num}-thread", "full-inference", "default-flags"],
              loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
              driver=iree_definitions.RuntimeDriver.LOCAL_TASK,
              benchmark_tool=MODULE_BENCHMARK_TOOL,
              extra_flags=[f"--task_topology_group_count={thread_num}"]))
    return run_configs
