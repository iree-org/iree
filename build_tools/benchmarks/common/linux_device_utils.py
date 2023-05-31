#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utils for accessing Linux device information."""

import re
from typing import Optional, Sequence

from .benchmark_definition import (execute_cmd_and_get_stdout, DeviceInfo,
                                   PlatformType)


def _get_lscpu_field(lscpu_output: str, field_name: str) -> str:
  (value,) = re.findall(f"^{field_name}:\s*(.+)", lscpu_output, re.MULTILINE)
  return value


def get_linux_cpu_arch(lscpu_output: str) -> str:
  """Returns CPU Architecture, e.g., 'x86_64'."""
  return _get_lscpu_field(lscpu_output, "Architecture")


def get_linux_cpu_features(lscpu_output: str) -> Sequence[str]:
  """Returns CPU feature lists, e.g., ['mmx', 'fxsr', 'sse', 'sse2']."""
  return _get_lscpu_field(lscpu_output, "Flags").split(" ")


def canonicalize_gpu_name(gpu_name: str) -> str:
  # Replace all consecutive non-word characters with a single hyphen.
  return re.sub(r"\W+", "-", gpu_name)


def get_linux_device_info(device_model: str = "Unknown",
                          cpu_uarch: Optional[str] = None,
                          gpu_id: str = "0",
                          verbose: bool = False) -> DeviceInfo:
  """Returns device info for the Linux device.

    Args:
    - device_model: the device model name, e.g., 'ThinkStation P520'
    - cpu_uarch: the CPU microarchitecture, e.g., 'CascadeLake'
    - gpu_id: the target GPU ID, e.g., '0' or 'GPU-<UUID>'
  """
  lscpu_output = execute_cmd_and_get_stdout(["lscpu"], verbose)

  try:
    gpu_name = execute_cmd_and_get_stdout([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
        f"--id={gpu_id}"
    ], verbose)
  except FileNotFoundError:
    # Set GPU name to Unknown if the tool "nvidia-smi" doesn't exist.
    gpu_name = "Unknown"

  return DeviceInfo(
      PlatformType.LINUX,
      # Includes CPU model as it is the key factor of the device performance.
      model=device_model,
      # Currently we only have x86, so CPU ABI = CPU arch.
      cpu_abi=get_linux_cpu_arch(lscpu_output),
      cpu_uarch=cpu_uarch,
      cpu_features=get_linux_cpu_features(lscpu_output),
      gpu_name=canonicalize_gpu_name(gpu_name))
