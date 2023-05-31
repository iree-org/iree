## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines device specs of Moto Edge X30."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

DEVICE_NAME = "moto-edge-x30"

GPU = common_definitions.DeviceSpec.build(
    id=unique_ids.DEVICE_SPEC_MOBILE_MOTO_EDGE_X30 + "-gpu",
    device_name=DEVICE_NAME,
    architecture=common_definitions.DeviceArchitecture.QUALCOMM_ADRENO,
    host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
    tags=["gpu"])
