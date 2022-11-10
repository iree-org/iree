## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import List, Sequence
import itertools

from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework import serialization


@serialization.serializable
@dataclass(frozen=True)
class BenchmarkRunConfig(object):
  model_run_config: iree_definitions.E2EModelRunConfig
  target_device_spec: common_definitions.DeviceSpec


def generate_e2e_model_run_configs(
    module_generation_configs: Sequence[
        iree_definitions.ModuleGenerationConfig],
    module_execution_configs: Sequence[iree_definitions.ModuleExecutionConfig],
    device_specs: Sequence[common_definitions.DeviceSpec],
) -> List[BenchmarkRunConfig]:
  """Generates the run specs from the product of compile specs and run configs.
  """
  return [
      BenchmarkRunConfig(model_run_config=iree_definitions.E2EModelRunConfig(
          module_generation_config=module_generation_config,
          module_execution_config=module_execution_config,
          input_data=common_definitions.ZEROS_MODEL_INPUT_DATA),
                         target_device_spec=device_spec)
      for module_generation_config,
      module_execution_config, device_spec in itertools.product(
          module_generation_configs, module_execution_configs, device_specs)
  ]
