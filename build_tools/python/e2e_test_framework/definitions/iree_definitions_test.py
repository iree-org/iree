## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from e2e_test_framework.definitions import common_definitions, iree_definitions


class IreeDefinitionsTest(unittest.TestCase):

  def test_generate_run_flags(self):
    imported_model = iree_definitions.ImportedModel.from_model(
        common_definitions.Model(
            id="1234",
            name="tflite_m",
            tags=[],
            source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
            source_url="https://example.com/xyz.tflite",
            entry_function="main",
            input_types=["1xf32", "2x2xf32"]))
    execution_config = iree_definitions.ModuleExecutionConfig(
        id="123",
        tags=["test"],
        loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
        driver=iree_definitions.RuntimeDriver.LOCAL_TASK,
        extra_flags=["--task=10"])

    flags = iree_definitions.generate_run_flags(
        imported_model=imported_model,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        module_execution_config=execution_config)

    self.assertEqual(flags, [
        "--function=main", "--input=1xf32=0", "--input=2x2xf32=0", "--task=10",
        "--device=local-task"
    ])

  def test_generate_run_flags_with_cuda(self):
    imported_model = iree_definitions.ImportedModel.from_model(
        common_definitions.Model(
            id="1234",
            name="tflite_m",
            tags=[],
            source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
            source_url="https://example.com/xyz.tflite",
            entry_function="main",
            input_types=["1xf32"]))
    execution_config = iree_definitions.ModuleExecutionConfig(
        id="123",
        tags=["test"],
        loader=iree_definitions.RuntimeLoader.NONE,
        driver=iree_definitions.RuntimeDriver.CUDA,
        extra_flags=[])

    flags = iree_definitions.generate_run_flags(
        imported_model=imported_model,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        module_execution_config=execution_config,
        gpu_id="3")

    self.assertEqual(
        flags, ["--function=main", "--input=1xf32=0", "--device=cuda://3"])

  def test_generate_run_flags_without_driver(self):
    imported_model = iree_definitions.ImportedModel.from_model(
        common_definitions.Model(
            id="1234",
            name="tflite_m",
            tags=[],
            source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
            source_url="https://example.com/xyz.tflite",
            entry_function="main",
            input_types=["1xf32"]))
    execution_config = iree_definitions.ModuleExecutionConfig(
        id="123",
        tags=["test"],
        loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
        driver=iree_definitions.RuntimeDriver.LOCAL_TASK,
        extra_flags=["--task=10"])

    flags = iree_definitions.generate_run_flags(
        imported_model=imported_model,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        module_execution_config=execution_config,
        with_driver=False)

    self.assertEqual(flags, ["--function=main", "--input=1xf32=0", "--task=10"])


if __name__ == "__main__":
  unittest.main()
