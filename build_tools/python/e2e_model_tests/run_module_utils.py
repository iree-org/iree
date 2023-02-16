## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utils that help launch iree run module tools."""

from typing import Any, List

from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.definitions.iree_definitions import ModuleExecutionConfig, RuntimeDriver
from e2e_test_framework.device_specs import device_parameters


def build_run_flags_for_model(
    model: common_definitions.Model,
    model_input_data: common_definitions.ModelInputData) -> List[str]:
  """Returns the IREE run module flags for the model and its inputs."""

  run_flags = [f"--function={model.entry_function}"]
  if model_input_data != common_definitions.ZEROS_MODEL_INPUT_DATA:
    raise ValueError("Currently only support all-zeros data.")
  run_flags += [f"--input={input_type}=0" for input_type in model.input_types]
  return run_flags


def build_run_flags_for_execution_config(
    module_execution_config: ModuleExecutionConfig,
    gpu_id: str = "0") -> List[str]:
  """Returns the IREE run module flags of the execution config."""

  run_flags = list(module_execution_config.extra_flags)
  driver = module_execution_config.driver
  if driver == RuntimeDriver.CUDA:
    run_flags.append(f"--device=cuda://{gpu_id}")
  else:
    run_flags.append(f"--device={driver.value}")
  return run_flags


def build_linux_wrapper_cmds_for_device_spec(
    device_spec: common_definitions.DeviceSpec) -> List[str]:
  """Builds the commands with tools to create the execution environment."""

  affinity_mask = None
  for param in device_spec.device_parameters:
    if param == device_parameters.OCTA_CORES:
      affinity_mask = "0xFF"
    else:
      raise ValueError(f"Unsupported device parameter: {param}.")

  cmds = []
  if affinity_mask is not None:
    cmds += ["taskset", affinity_mask]

  return cmds
