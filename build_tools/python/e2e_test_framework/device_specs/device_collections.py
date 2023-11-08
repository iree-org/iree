# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines the collections of device specs and provides query methods."""

from typing import List, Sequence, Set
from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.device_specs import (
    gcp_specs,
    moto_edge_x30_specs,
    pixel_4_specs,
    pixel_6_pro_specs,
    riscv_specs,
)


class DeviceCollection(object):
    """Class to collect and query device specs."""

    def __init__(self, device_specs: Sequence[common_definitions.DeviceSpec]):
        self.device_specs = device_specs

    def query_device_specs(
        self,
        architecture: common_definitions.DeviceArchitecture,
        host_environment: common_definitions.HostEnvironment,
        device_parameters: Set[str] = set(),
    ) -> List[common_definitions.DeviceSpec]:
        """Query the device specs.

        Args:
          architecture: device architecture to match.
          platform: device platform to match.
          device_parameters: parameters that devices need to have.
        Returns:
          List of matched device specs.
        """

        matched_device_specs = []
        for device_spec in self.device_specs:
            if device_spec.architecture != architecture:
                continue
            if device_spec.host_environment != host_environment:
                continue
            if not device_parameters.issubset(device_spec.device_parameters):
                continue
            matched_device_specs.append(device_spec)

        return matched_device_specs


ALL_DEVICE_SPECS = [
    # Pixel 6 Pro
    pixel_6_pro_specs.LITTLE_CORES,
    pixel_6_pro_specs.BIG_CORES,
    pixel_6_pro_specs.ALL_CORES,
    pixel_6_pro_specs.GPU,
    # Moto Edge X30
    moto_edge_x30_specs.GPU,
    # GCP machines
    gcp_specs.GCP_C2_STANDARD_16,
    gcp_specs.GCP_C2_STANDARD_60,
    gcp_specs.GCP_A2_HIGHGPU_1G,
    # RISCV emulators
    riscv_specs.EMULATOR_RISCV_32,
    riscv_specs.EMULATOR_RISCV_64,
]
DEFAULT_DEVICE_COLLECTION = DeviceCollection(ALL_DEVICE_SPECS)
