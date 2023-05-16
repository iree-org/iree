#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utils for accessing Android devices."""

import json
import re

from typing import Sequence
from .benchmark_definition import (execute_cmd_and_get_stdout, DeviceInfo,
                                   PlatformType)


def get_android_device_model(verbose: bool = False) -> str:
  """Returns the Android device model."""
  model = execute_cmd_and_get_stdout(
      ["adb", "shell", "getprop", "ro.product.model"], verbose=verbose)
  model = re.sub(r"\W+", "-", model)
  return model


def get_android_cpu_abi(verbose: bool = False) -> str:
  """Returns the CPU ABI for the Android device."""
  return execute_cmd_and_get_stdout(
      ["adb", "shell", "getprop", "ro.product.cpu.abi"], verbose=verbose)


def get_android_cpu_features(verbose: bool = False) -> Sequence[str]:
  """Returns the CPU features for the Android device."""
  cpuinfo = execute_cmd_and_get_stdout(["adb", "shell", "cat", "/proc/cpuinfo"],
                                       verbose=verbose)
  features = []
  for line in cpuinfo.splitlines():
    if line.startswith("Features"):
      _, features = line.split(":")
      return features.strip().split()
  return features


def get_android_gpu_name(verbose: bool = False) -> str:
  """Returns the GPU name for the Android device."""
  vkjson = execute_cmd_and_get_stdout(["adb", "shell", "cmd", "gpu", "vkjson"],
                                      verbose=verbose)
  vkjson = json.loads(vkjson)
  name = vkjson["devices"][0]["properties"]["deviceName"]

  # Perform some canonicalization:

  # - Adreno GPUs have raw names like "Adreno (TM) 650".
  name = name.replace("(TM)", "")

  # Replace all consecutive non-word characters with a single hyphen.
  name = re.sub(r"\W+", "-", name)

  return name


def get_android_device_info(verbose: bool = False) -> DeviceInfo:
  """Returns device info for the Android device."""
  return DeviceInfo(platform_type=PlatformType.ANDROID,
                    model=get_android_device_model(verbose),
                    cpu_abi=get_android_cpu_abi(verbose),
                    cpu_uarch=None,
                    cpu_features=get_android_cpu_features(verbose),
                    gpu_name=get_android_gpu_name(verbose))
