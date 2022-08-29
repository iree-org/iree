## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines x86_64 linux devices."""

from .. import unique_ids
from ..definitions import common_definitions

GCP_C2_STANDARD_16 = common_definitions.DeviceSpec(
    id=unique_ids.DEVICE_SPEC_GCP_C2_STANDARD_16,
    vendor_name="GCP-c2-standard-16",
    architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
    platform=common_definitions.DevicePlatform.LINUX_GNU,
    device_parameters=["all-cores"])
