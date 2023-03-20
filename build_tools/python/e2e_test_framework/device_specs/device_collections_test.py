## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.device_specs import device_collections


class DeviceCollectionTest(unittest.TestCase):

  def test_query_device_specs(self):
    linux_x86_device_spec = common_definitions.DeviceSpec.build(
        id="linux_x86",
        device_name="a",
        architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
        tags=[])
    android_x86_device_spec = common_definitions.DeviceSpec.build(
        id="android_x86",
        device_name="b",
        architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        tags=[])
    little_cores_device_spec = common_definitions.DeviceSpec.build(
        id="android_little",
        device_name="c",
        architecture=common_definitions.DeviceArchitecture.ARMV9_A_GENERIC,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        device_parameters=["little-cores"],
        tags=[])
    big_cores_device_spec = common_definitions.DeviceSpec.build(
        id="android_big",
        device_name="d",
        architecture=common_definitions.DeviceArchitecture.ARMV9_A_GENERIC,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        device_parameters=["big-cores"],
        tags=[])
    devices = device_collections.DeviceCollection(device_specs=[
        linux_x86_device_spec, android_x86_device_spec,
        little_cores_device_spec, big_cores_device_spec
    ])

    linux_x86_devices = devices.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64)
    android_x86_devices = devices.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A)
    little_cores_devices = devices.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.ARMV9_A_GENERIC,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        device_parameters={"little-cores"})
    big_cores_devices = devices.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.ARMV9_A_GENERIC,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        device_parameters={"big-cores"})
    all_arm_devices = devices.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.ARMV9_A_GENERIC,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A)
    no_matched_device = devices.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.ARMV9_A_GENERIC,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64)

    self.assertEqual(linux_x86_devices, [linux_x86_device_spec])
    self.assertEqual(android_x86_devices, [android_x86_device_spec])
    self.assertEqual(little_cores_devices, [little_cores_device_spec])
    self.assertEqual(big_cores_devices, [big_cores_device_spec])
    self.assertEqual(all_arm_devices,
                     [little_cores_device_spec, big_cores_device_spec])
    self.assertEqual(no_matched_device, [])


if __name__ == "__main__":
  unittest.main()
