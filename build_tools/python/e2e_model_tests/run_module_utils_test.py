## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from e2e_model_tests import run_module_utils
from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.device_specs import device_parameters


class RunModuleUtilsTest(unittest.TestCase):

  def test_build_linux_wrapper_cmds_for_device_spec(self):
    device_spec = common_definitions.DeviceSpec(
        id="abc",
        device_name="test-device",
        architecture=common_definitions.DeviceArchitecture.VMVX_GENERIC,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
        device_parameters=[device_parameters.OCTA_CORES])

    flags = run_module_utils.build_linux_wrapper_cmds_for_device_spec(
        device_spec)

    self.assertEqual(flags, ["taskset", "0xFF"])


if __name__ == "__main__":
  unittest.main()
