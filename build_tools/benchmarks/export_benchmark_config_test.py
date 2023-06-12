#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from e2e_test_framework.definitions import common_definitions, iree_definitions
import export_benchmark_config

COMMON_MODEL = common_definitions.Model(
    id="tflite",
    name="model_tflite",
    tags=[],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    source_url="",
    entry_function="predict",
    input_types=["1xf32"])
COMMON_GEN_CONFIG = iree_definitions.ModuleGenerationConfig.build(
    imported_model=iree_definitions.ImportedModel.from_model(COMMON_MODEL),
    compile_config=iree_definitions.CompileConfig.build(
        id="1",
        tags=[],
        compile_targets=[
            iree_definitions.CompileTarget(
                target_backend=iree_definitions.TargetBackend.LLVM_CPU,
                target_architecture=common_definitions.DeviceArchitecture.
                RV64_GENERIC,
                target_abi=iree_definitions.TargetABI.LINUX_GNU)
        ]))
COMMON_EXEC_CONFIG = iree_definitions.ModuleExecutionConfig.build(
    id="exec",
    tags=[],
    loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
    driver=iree_definitions.RuntimeDriver.LOCAL_SYNC)


class ExportBenchmarkConfigTest(unittest.TestCase):

  def test_filter_and_group_run_configs_set_all_filters(self):
    device_spec_a = common_definitions.DeviceSpec.build(
        id="dev_a_cpu",
        device_name="dev_a_cpu",
        architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        tags=[])
    device_spec_b = common_definitions.DeviceSpec.build(
        id="dev_a_gpu",
        device_name="dev_a_gpu",
        architecture=common_definitions.DeviceArchitecture.ARM_VALHALL,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        tags=[])
    device_spec_c = common_definitions.DeviceSpec.build(
        id="dev_c_accel",
        device_name="dev_c_accel",
        architecture=common_definitions.DeviceArchitecture.CUDA_SM80,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
        tags=[])
    matched_run_config_a = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=COMMON_GEN_CONFIG,
        module_execution_config=COMMON_EXEC_CONFIG,
        target_device_spec=device_spec_a,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        presets=["default"])
    unmatched_run_config_b = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=COMMON_GEN_CONFIG,
        module_execution_config=COMMON_EXEC_CONFIG,
        target_device_spec=device_spec_b,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        presets=["default"])
    matched_run_config_c = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=COMMON_GEN_CONFIG,
        module_execution_config=COMMON_EXEC_CONFIG,
        target_device_spec=device_spec_c,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        presets=["large", "z80"])

    run_config_map = export_benchmark_config.filter_and_group_run_configs(
        run_configs=[
            matched_run_config_a, unmatched_run_config_b, matched_run_config_c
        ],
        target_device_names={"dev_a_cpu", "dev_c_accel"},
        presets=["default", "z80"])

    self.assertEqual(run_config_map, {
        "dev_a_cpu": [matched_run_config_a],
        "dev_c_accel": [matched_run_config_c],
    })

  def test_filter_and_group_run_configs_include_all(self):
    device_spec_a = common_definitions.DeviceSpec.build(
        id="dev_a_cpu",
        device_name="dev_a_cpu",
        architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        tags=[])
    device_spec_b = common_definitions.DeviceSpec.build(
        id="dev_a_gpu",
        device_name="dev_a_gpu",
        architecture=common_definitions.DeviceArchitecture.ARM_VALHALL,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        tags=[])
    device_spec_c = common_definitions.DeviceSpec.build(
        id="dev_a_second_gpu",
        device_name="dev_a_gpu",
        architecture=common_definitions.DeviceArchitecture.QUALCOMM_ADRENO,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        tags=[])
    run_config_a = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=COMMON_GEN_CONFIG,
        module_execution_config=COMMON_EXEC_CONFIG,
        target_device_spec=device_spec_a,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)
    run_config_b = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=COMMON_GEN_CONFIG,
        module_execution_config=COMMON_EXEC_CONFIG,
        target_device_spec=device_spec_b,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)
    run_config_c = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=COMMON_GEN_CONFIG,
        module_execution_config=COMMON_EXEC_CONFIG,
        target_device_spec=device_spec_c,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)

    run_config_map = export_benchmark_config.filter_and_group_run_configs(
        run_configs=[run_config_a, run_config_b, run_config_c])

    self.assertEqual(run_config_map, {
        "dev_a_cpu": [run_config_a],
        "dev_a_gpu": [run_config_b, run_config_c],
    })

  def test_filter_and_group_run_configs_set_target_device_names(self):
    device_spec_a = common_definitions.DeviceSpec.build(
        id="dev_a",
        device_name="dev_a",
        architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        tags=[])
    device_spec_b = common_definitions.DeviceSpec.build(
        id="dev_b",
        device_name="dev_b",
        architecture=common_definitions.DeviceArchitecture.ARM_VALHALL,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        tags=[])
    run_config_a = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=COMMON_GEN_CONFIG,
        module_execution_config=COMMON_EXEC_CONFIG,
        target_device_spec=device_spec_a,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)
    run_config_b = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=COMMON_GEN_CONFIG,
        module_execution_config=COMMON_EXEC_CONFIG,
        target_device_spec=device_spec_b,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)

    run_config_map = export_benchmark_config.filter_and_group_run_configs(
        run_configs=[run_config_a, run_config_b],
        target_device_names={"dev_a", "dev_b"})

    self.assertEqual(run_config_map, {
        "dev_a": [run_config_a],
        "dev_b": [run_config_b],
    })

  def test_filter_and_group_run_configs_with_presets(self):
    small_model = common_definitions.Model(
        id="small_model",
        name="small_model",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="",
        entry_function="predict",
        input_types=["1xf32"])
    big_model = common_definitions.Model(
        id="big_model",
        name="big_model",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="",
        entry_function="predict",
        input_types=["1xf32"])
    compile_target = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
        target_abi=iree_definitions.TargetABI.LINUX_GNU)
    compile_config = iree_definitions.CompileConfig.build(
        id="1", tags=[], compile_targets=[compile_target])
    small_gen_config = iree_definitions.ModuleGenerationConfig.build(
        imported_model=iree_definitions.ImportedModel.from_model(small_model),
        compile_config=compile_config)
    big_gen_config = iree_definitions.ModuleGenerationConfig.build(
        imported_model=iree_definitions.ImportedModel.from_model(big_model),
        compile_config=compile_config)
    device_spec_a = common_definitions.DeviceSpec.build(
        id="dev_a",
        device_name="dev_a",
        architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        tags=[])
    device_spec_b = common_definitions.DeviceSpec.build(
        id="dev_b",
        device_name="dev_b",
        architecture=common_definitions.DeviceArchitecture.ARM_VALHALL,
        host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        tags=[])
    run_config_default = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=small_gen_config,
        module_execution_config=COMMON_EXEC_CONFIG,
        target_device_spec=device_spec_a,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        presets=["riscv_64", "default"])
    run_config_large = iree_definitions.E2EModelRunConfig.build(
        module_generation_config=big_gen_config,
        module_execution_config=COMMON_EXEC_CONFIG,
        target_device_spec=device_spec_b,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        presets=["riscv_64", "large"])

    run_config_map = export_benchmark_config.filter_and_group_run_configs(
        run_configs=[run_config_default, run_config_large], presets=["default"])

    self.assertEqual(run_config_map, {
        "dev_a": [run_config_default],
    })


if __name__ == "__main__":
  unittest.main()
