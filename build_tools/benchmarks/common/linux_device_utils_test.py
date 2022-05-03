#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from unittest import mock

from common.benchmark_definition import DeviceInfo, PlatformType
from common.linux_device_utils import get_linux_cpu_arch, get_linux_cpu_features, get_linux_cpu_model, get_linux_device_info


class LinuxDeviceUtilsTest(unittest.TestCase):

  def setUp(self):
    self.execute_cmd_patch = mock.patch(
        "common.linux_device_utils.execute_cmd_and_get_output")
    self.execute_cmd_mock = self.execute_cmd_patch.start()
    self.execute_cmd_mock.return_value = (
        "Architecture:                    x86_64\n"
        "Vendor ID:                       AuthenticAMD\n"
        "Model name:                      AMD EPYC 7B12\n"
        "Flags:                           fpu vme de pse tsc\n")

  def tearDown(self):
    self.execute_cmd_patch.stop()

  def test_get_linux_cpu_arch(self):
    self.assertEqual(get_linux_cpu_arch(), "x86_64")

  def test_get_linux_cpu_features(self):
    self.assertEqual(get_linux_cpu_features(),
                     ["fpu", "vme", "de", "pse", "tsc"])

  def test_get_linux_cpu_model(self):
    self.assertEqual(get_linux_cpu_model(), "AMD EPYC 7B12")

  def test_get_linux_device_info(self):
    self.assertEqual(
        get_linux_device_info("Dummy", "Zen2"),
        DeviceInfo(platform_type=PlatformType.LINUX,
                   model="Dummy(AMD EPYC 7B12)",
                   cpu_abi="x86_64",
                   cpu_uarch="Zen2",
                   cpu_features=["fpu", "vme", "de", "pse", "tsc"],
                   gpu_name="Unknown"))


if __name__ == "__main__":
  unittest.main()
