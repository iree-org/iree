## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utils that help launch iree run module tools."""

from typing import List

from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.device_specs import device_parameters


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
