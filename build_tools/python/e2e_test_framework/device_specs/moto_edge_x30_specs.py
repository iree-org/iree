## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines device specs of Moto Edge X30."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.device_specs import device_parameters

VENDOR_NAME = "XT2201-2"

BIG_CORES = common_definitions.DeviceSpec(
    id=unique_ids.DEVICE_SPEC_MOBILE_MOTO_EDGE_X30 + "_big-core",
    vendor_name=VENDOR_NAME,
    architecture=common_definitions.DeviceArchitecture.ARMV9_A_GENERIC,
    platform=common_definitions.DevicePlatform.GENERIC_ANDROID,
    device_parameters=[device_parameters.ARM_BIG_CORES])
LITTLE_CORES = common_definitions.DeviceSpec(
    id=unique_ids.DEVICE_SPEC_MOBILE_MOTO_EDGE_X30 + "_litte-core",
    vendor_name=VENDOR_NAME,
    architecture=common_definitions.DeviceArchitecture.ARMV9_A_GENERIC,
    platform=common_definitions.DevicePlatform.GENERIC_ANDROID,
    device_parameters=[device_parameters.ARM_LITTLE_CORES])
GPU = common_definitions.DeviceSpec(
    id=unique_ids.DEVICE_SPEC_MOBILE_MOTO_EDGE_X30 + "_gpu",
    vendor_name=VENDOR_NAME,
    architecture=common_definitions.DeviceArchitecture.ADRENO_GENERIC,
    platform=common_definitions.DevicePlatform.GENERIC_ANDROID)
