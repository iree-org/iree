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
    self.assertIn("local-sync", driver_names)
    self.assertIn("local-task", driver_names)

    for driver_name in ["local-sync", "local-task"]:
      driver = ss.get_driver(driver_name)
      print(f"Driver {driver_name}: {driver}")
      device_infos = driver.query_available_devices()
      print(f"DeviceInfos: {device_infos}")
      if driver_name == "local-sync":
        # We happen to know that this should have one device_info
        self.assertEqual(device_infos, [(0, "default")])

  def testCreateBadDeviceId(self):
    driver = ss.get_driver("local-sync")
    with self.assertRaises(
        ValueError,
        msg="Device id 5555 not found. Available devices: [(0, 'default')]"):
      _ = driver.create_device(5555)

  def testCreateDevice(self):
    driver = ss.get_driver("local-sync")
    infos = driver.query_available_devices()
    # Each info record is (device_id, name)
    device1 = driver.create_device(infos[0][0])
    # Should also take the info tuple directly for convenience.
    device2 = driver.create_device(infos[0])

  def testCreateDeviceByName(self):
    device1 = ss.get_device("local-task")
    device2 = ss.get_device("local-sync")
    device3 = ss.get_device("local-sync")
    device4 = ss.get_device("local-sync", cache=False)
    self.assertIsNot(device1, device2)
    self.assertIsNot(device3, device4)
    self.assertIs(device2, device3)

    with self.assertRaises(ValueError, msg="Device not found: local-sync://1"):
      _ = ss.get_device("local-sync://1")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  unittest.main()
