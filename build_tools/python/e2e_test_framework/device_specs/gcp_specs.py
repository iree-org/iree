## copyright 2022 the iree authors
#
# licensed under the apache license v2.0 with llvm exceptions.
# see https://llvm.org/license.txt for license information.
# spdx-license-identifier: apache-2.0 with llvm-exception
"""Defines device specs for GCP machines."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

GCP_C2_STANDARD_16 = common_definitions.DeviceSpec(
    id=unique_ids.DEVICE_SPEC_GCP_C2_STANDARD_16,
    vendor_name="GCP-c2-standard-16",
    architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
    platform=common_definitions.DevicePlatform.GENERIC_LINUX,
    device_parameters=["all-cores"])

GCP_A2_HIGHGPU_1G = common_definitions.DeviceSpec(
    id=unique_ids.DEVICE_SPEC_GCP_A2_HIGHGPU_1G,
    vendor_name="GCP-a2-highgpu-1g",
    architecture=common_definitions.DeviceArchitecture.CUDA_SM80,
    platform=common_definitions.DevicePlatform.GENERIC_LINUX)
