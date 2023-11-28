## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines device specs of Pixel 6 Pro."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

DEVICE_NAME = "pixel-6-pro"

BIG_CORES = common_definitions.DeviceSpec.build(
    id=unique_ids.DEVICE_SPEC_MOBILE_PIXEL_6_PRO + "-big-core",
    device_name=DEVICE_NAME,
    architecture=common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
    host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
    device_parameters=common_definitions.DeviceParameters(
        # Cortex-X1
        cpu_params=common_definitions.CPUParameters(pinned_cores=[6, 7])
    ),
    tags=["big-cores"],
)
ALL_CORES = common_definitions.DeviceSpec.build(
    id=unique_ids.DEVICE_SPEC_MOBILE_PIXEL_6_PRO + "-all-core",
    device_name=DEVICE_NAME,
    architecture=common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
    host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
    tags=["all-cores"],
)
GPU = common_definitions.DeviceSpec.build(
    id=unique_ids.DEVICE_SPEC_MOBILE_PIXEL_6_PRO + "-gpu",
    device_name=DEVICE_NAME,
    architecture=common_definitions.DeviceArchitecture.ARM_VALHALL,
    host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
    device_parameters=common_definitions.DeviceParameters(
        # Pin on the fastest CPU core.
        cpu_params=common_definitions.CPUParameters(pinned_cores=[7])
    ),
    tags=["gpu"],
)
