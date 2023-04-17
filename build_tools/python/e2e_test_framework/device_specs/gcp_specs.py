## copyright 2022 the iree authors
#
# licensed under the apache license v2.0 with llvm exceptions.
# see https://llvm.org/license.txt for license information.
# spdx-license-identifier: apache-2.0 with llvm-exception
"""Defines device specs for GCP machines."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.device_specs import device_parameters

GCP_C2_STANDARD_16 = common_definitions.DeviceSpec.build(
    id=unique_ids.DEVICE_SPEC_GCP_C2_STANDARD_16,
    device_name="c2-standard-16",
    host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
    architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
    device_parameters=[device_parameters.ALL_CORES],
    tags=["cpu"])

GCP_A2_HIGHGPU_1G = common_definitions.DeviceSpec.build(
    id=unique_ids.DEVICE_SPEC_GCP_A2_HIGHGPU_1G,
    device_name="a2-highgpu-1g",
    host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
    architecture=common_definitions.DeviceArchitecture.NVIDIA_AMPERE,
    tags=["gpu"])
