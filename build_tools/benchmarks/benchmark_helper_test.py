#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import benchmark_helper
from e2e_test_artifacts import model_artifacts, iree_artifacts
from e2e_test_framework.definitions import common_definitions, iree_definitions


class BenchmarkHelper(unittest.TestCase):

  def test_dump_flags_of_generation_config(self):
    model = common_definitions.Model(
        id="abcd",
        name="test-name",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="",
        entry_function="predict",
        input_types=["1xf32"])
    compile_target = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_architecture=common_definitions.DeviceArchitecture.
        X86_64_CASCADELAKE,
        target_abi=iree_definitions.TargetABI.LINUX_GNU)
    compile_config = iree_definitions.CompileConfig(
        id="comp_a",
        tags=[],
        compile_targets=[compile_target],
        extra_flags=["--test-flag=abcd"])
    imported_model = iree_definitions.ImportedModel.from_model(model)
    gen_config = iree_definitions.ModuleGenerationConfig.with_flag_generation(
        imported_model=imported_model, compile_config=compile_config)

    output = benchmark_helper.dump_flags_of_generation_config(
        module_generation_config=gen_config)

    model_path = model_artifacts.get_model_path(model=imported_model.model)
    imported_model_path = iree_artifacts.get_imported_model_path(
        imported_model=imported_model)
    self.assertEquals(output["composite_id"], gen_config.composite_id)
    self.assertIn(str(imported_model_path), output["compile_flags"])
    self.assertIn("--test-flag=abcd", output["compile_flags"])
    self.assertEquals(output["import_tool"],
                      imported_model.import_config.tool.value)
    self.assertIn(str(model_path), output["import_flags"])

  def test_dump_flags_from_run_config(self):
    model = common_definitions.Model(
        id="abcd",
        name="test-name",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="",
        entry_function="predict",
        input_types=["1xf32"])
    compile_target = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_architecture=common_definitions.DeviceArchitecture.
        X86_64_CASCADELAKE,
        target_abi=iree_definitions.TargetABI.LINUX_GNU)
    compile_config = iree_definitions.CompileConfig(
        id="comp_a",
        tags=[],
        compile_targets=[compile_target],
        extra_flags=["--comp=abcd"])
    imported_model = iree_definitions.ImportedModel.from_model(model)
    gen_config = iree_definitions.ModuleGenerationConfig.with_flag_generation(
        imported_model=imported_model, compile_config=compile_config)
    exec_config = iree_definitions.ModuleExecutionConfig(
        id="exec_a",
        tags=[],
        loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
        driver=iree_definitions.RuntimeDriver.LOCAL_SYNC,
        extra_flags=["--test-flag=abcd"])
    device_spec = common_definitions.DeviceSpec(
        id="dev_a",
        device_name="dev_a",
        architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64)
    run_config = iree_definitions.E2EModelRunConfig.with_flag_generation(
        module_generation_config=gen_config,
        module_execution_config=exec_config,
        target_device_spec=device_spec,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA)

    output = benchmark_helper.dump_flags_from_run_config(
        e2e_model_run_config=run_config)

    module_path = iree_artifacts.get_module_dir_path(
        module_generation_config=gen_config) / iree_artifacts.MODULE_FILENAME
    self.assertEqual(output["composite_id"], run_config.composite_id)
    self.assertIn(f"--module={str(module_path)}", output["run_flags"])
    self.assertIn(f"--test-flag=abcd", output["run_flags"])
    self.assertIn("module_generation_config", output)


if __name__ == "__main__":
  unittest.main()
