## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from e2e_model_tests import run_module_utils
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import device_parameters


class RunModuleTuilsTest(unittest.TestCase):

  def test_build_run_flags_for_model(self):
    model = common_definitions.Model(
        id="1234",
        name="tflite_m",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="https://example.com/xyz.tflite",
        entry_function="main",
        input_types=["1xf32", "2x2xf32"])

    flags = run_module_utils.build_run_flags_for_model(
        model, common_definitions.ZEROS_MODEL_INPUT_DATA)

    self.assertEqual(
        flags, ["--function=main", "--input=1xf32=0", "--input=2x2xf32=0"])

  def test_build_run_flags_for_execution_config(self):
    execution_config = iree_definitions.ModuleExecutionConfig(
        id="123",
        tags=["test"],
        loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
        driver=iree_definitions.RuntimeDriver.LOCAL_TASK,
        extra_flags=["--task=10"])

    flags = run_module_utils.build_run_flags_for_execution_config(
        execution_config)

    self.assertEqual(
        flags,
        ["--task=10", "--device_allocator=caching", "--device=local-task"])

  def test_build_run_flags_for_execution_config_with_cuda(self):
    execution_config = iree_definitions.ModuleExecutionConfig(
        id="123",
        tags=["test"],
        loader=iree_definitions.RuntimeLoader.NONE,
        driver=iree_definitions.RuntimeDriver.CUDA,
        extra_flags=[])

    flags = run_module_utils.build_run_flags_for_execution_config(
        execution_config, gpu_id="3")

    self.assertEqual(flags, ["--device_allocator=caching", "--device=cuda://3"])

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
