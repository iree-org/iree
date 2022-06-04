# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

from iree.runtime import system_setup as ss


class DeviceSetupTest(unittest.TestCase):

  def testQueryDriversDevices(self):
    driver_names = ss.query_available_drivers()
    print(f"Drivers: {driver_names}")
    self.assertIn("vmvx", driver_names)
    self.assertIn("dylib", driver_names)

    for driver_name in ["vmvx", "dylib"]:
      driver = ss.get_driver(driver_name)
      print(f"Driver {driver_name}: {driver}")
      device_infos = driver.query_available_devices()
      print(f"DeviceInfos: {device_infos}")
      if driver_name == "vmvx":
        # We happen to know that this should have one device_info
        self.assertEqual(device_infos, [(0, "default")])

  def testCreateBadDeviceId(self):
    driver = ss.get_driver("vmvx")
    with self.assertRaises(
        ValueError,
        msg="Device id 5555 not found. Available devices: [(0, 'default')]"):
      _ = driver.create_device(5555)

  def testCreateDevice(self):
    driver = ss.get_driver("vmvx")
    infos = driver.query_available_devices()
    # Each info record is (device_id, name)
    device1 = driver.create_device(infos[0][0])
    # Should also take the info tuple directly for convenience.
    device2 = driver.create_device(infos[0])

  def testCreateDeviceByName(self):
    device1 = ss.get_device_by_name("vmvx")
    device2 = ss.get_device_by_name("vmvx:0")
    device3 = ss.get_device_by_name("vmvx:0")
    self.assertIsNot(device1, device2)
    self.assertIs(device2, device3)

    with self.assertRaises(
        ValueError,
        msg="Device index 5555 is out of range. Found devices [(0, 'default')]"
    ):
      _ = ss.get_device_by_name("vmvx:5555")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  unittest.main()
