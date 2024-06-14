# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

from iree.runtime import system_setup as ss
from iree.runtime._binding import create_hal_driver, clear_hal_driver_cache


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
                self.assertEqual(
                    device_infos, [{"device_id": 0, "path": "", "name": "default"}]
                )

    def testCreateBadDeviceId(self):
        driver = ss.get_driver("local-sync")
        with self.assertRaises(
            ValueError,
            msg="Device id 5555 not found. Available devices: [{ device_id:0, path:'', name:'default'}]",
        ):
            _ = driver.create_device(5555)

    def testCreateDevice(self):
        driver = ss.get_driver("local-sync")
        infos = driver.query_available_devices()
        # Each record is a dict:
        # {"device_id": obj, "path": str, "name": str}.
        device1 = driver.create_device(infos[0]["device_id"])
        # Should also take the info dict directly for convenience.
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

    def testCreateDeviceWithAllocators(self):
        driver = ss.get_driver("local-sync")
        infos = driver.query_available_devices()
        device1 = driver.create_device(infos[0]["device_id"], allocators=[])
        device2 = driver.create_device(
            infos[0]["device_id"], allocators=["caching", "debug"]
        )

    def testDriverCacheInternals(self):
        # Two drivers created with the same URI using the caching get_driver
        # should return the same driver
        driver1 = ss.get_driver("local-sync")
        driver2 = ss.get_driver("local-sync")
        self.assertIs(driver1, driver2)

        # A driver created using the non-caching create_hal_driver should be
        # unique from cached drivers of the same URI
        driver3 = create_hal_driver("local-sync")
        self.assertIsNot(driver3, driver1)

        # Drivers created with create_hal_driver should all be unique from
        # one another
        driver4 = create_hal_driver("local-sync")
        self.assertIsNot(driver4, driver3)

        # Clearing the cache should make any new driver unique from previously
        # cached ones
        clear_hal_driver_cache()
        driver5 = ss.get_driver("local-sync")
        self.assertIsNot(driver5, driver1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
