## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Sequence, Optional
import itertools

from e2e_test_framework.definitions import common_definitions, iree_definitions


def generate_e2e_model_run_configs(
    module_generation_configs: Sequence[
        iree_definitions.ModuleGenerationConfig],
    module_execution_configs: Sequence[iree_definitions.ModuleExecutionConfig],
    device_specs: Sequence[common_definitions.DeviceSpec],
    tags: Optional[Sequence[str]] = None,
    tool: iree_definitions.E2EModelRunTool = iree_definitions.E2EModelRunTool.
    IREE_BENCHMARK_MODULE
) -> List[iree_definitions.E2EModelRunConfig]:
  """Generates the run specs from the product of compile specs and run configs.
  """
  return [
      iree_definitions.E2EModelRunConfig.build(
          module_generation_config=module_generation_config,
          module_execution_config=module_execution_config,
          target_device_spec=device_spec,
          input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
          tool=tool,
          tags=tags) for module_generation_config,
      module_execution_config, device_spec in itertools.product(
          module_generation_configs, module_execution_configs, device_specs)
  ]
