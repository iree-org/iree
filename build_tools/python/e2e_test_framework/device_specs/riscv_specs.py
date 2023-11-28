## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines device specs of RISC-V devices."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions


EMULATOR_RISCV_64 = common_definitions.DeviceSpec.build(
    id=unique_ids.DEVICE_SPEC_EMULATOR_RISCV_64,
    device_name="emulator-riscv_64",
    host_environment=common_definitions.HostEnvironment.LINUX_RISCV_64,
    architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
    tags=["cpu"],
)

EMULATOR_RISCV_32 = common_definitions.DeviceSpec.build(
    id=unique_ids.DEVICE_SPEC_EMULATOR_RISCV_32,
    device_name="emulator-riscv_32",
    host_environment=common_definitions.HostEnvironment.LINUX_RISCV_32,
    architecture=common_definitions.DeviceArchitecture.RV32_GENERIC,
    tags=["cpu"],
)
