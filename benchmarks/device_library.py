## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from benchmarks.definitions.common import DeviceArchitecture, DevicePlatform, DeviceSpec

GCP_C2_STANDARD_16_DEVICE_SPEC = DeviceSpec(
    id="d1234",
    vendor_name="intel",
    architecture=DeviceArchitecture.X86_64_CASCADELAKE,
    platform=DevicePlatform.LINUX_GNU,
    configs=["any-cores"])
