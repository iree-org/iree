#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from unittest import mock

from common.benchmark_definition import DeviceInfo, PlatformType
from common.linux_device_utils import canonicalize_gpu_name, get_linux_cpu_arch, get_linux_cpu_features

LSCPU_OUTPUT = ("Architecture:                    x86_64\n"
                "Vendor ID:                       AuthenticAMD\n"
                "Flags:                           fpu vme de pse tsc\n")


class LinuxDeviceUtilsTest(unittest.TestCase):

  def test_get_linux_cpu_arch(self):
    self.assertEqual(get_linux_cpu_arch(LSCPU_OUTPUT), "x86_64")

  def test_get_linux_cpu_features(self):
    self.assertEqual(get_linux_cpu_features(LSCPU_OUTPUT),
                     ["fpu", "vme", "de", "pse", "tsc"])

  def test_canonicalize_gpu_name(self):
    self.assertEqual(canonicalize_gpu_name("Tesla  V100-SXM2-16GB"),
                     "Tesla-V100-SXM2-16GB")


if __name__ == "__main__":
  unittest.main()
