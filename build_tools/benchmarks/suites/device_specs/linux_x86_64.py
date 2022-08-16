## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines x86_64 linux devices."""

from ..id_defs import DEVICE_SPEC_GCP_C2_STANDARD_16_ID
from ..definitions.common import DeviceArchitecture, DevicePlatform, DeviceSpec

GCP_C2_STANDARD_16_DEVICE_SPEC = DeviceSpec(
    id=DEVICE_SPEC_GCP_C2_STANDARD_16_ID,
    vendor_name="GCP-c2-standard-16",
    architecture=DeviceArchitecture.X86_64_CASCADELAKE,
    platform=DevicePlatform.LINUX_GNU,
    device_parameters=["all-cores"])
