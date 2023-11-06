#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import unittest
from common.benchmark_definition import IREE_DRIVERS_INFOS
from common.benchmark_suite import BenchmarkCase, BenchmarkSuite
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_artifacts import iree_artifacts


class BenchmarkSuiteTest(unittest.TestCase):
    def test_filter_benchmarks(self):
        model = common_definitions.Model(
            id="model",
            name="model",
            tags=[],
            source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
            source_url="",
            entry_function="predict",
            input_types=["1xf32"],
        )
        exec_config = iree_definitions.ModuleExecutionConfig.build(
            id="exec",
            tags=[],
            loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
            driver=iree_definitions.RuntimeDriver.LOCAL_SYNC,
        )
        device_spec = common_definitions.DeviceSpec.build(
            id="dev",
            device_name="dev",
            architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
            host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
            device_parameters=[],
            tags=[],
        )
        compile_target = iree_definitions.CompileTarget(
            target_backend=iree_definitions.TargetBackend.LLVM_CPU,
            target_architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
            target_abi=iree_definitions.TargetABI.LINUX_GNU,
        )
        dummy_run_config = iree_definitions.E2EModelRunConfig.build(
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=iree_definitions.ImportedModel.from_model(model),
                compile_config=iree_definitions.CompileConfig.build(
                    id="1", tags=[], compile_targets=[compile_target]
                ),
            ),
            module_execution_config=exec_config,
            target_device_spec=device_spec,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        )

        case1 = BenchmarkCase(
            model_name="deepnet",
            model_tags=[],
            bench_mode=["1-thread", "full-inference"],
            target_arch=common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
            driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu"],
            module_dir=pathlib.Path("case1"),
            benchmark_tool_name="tool",
            run_config=dummy_run_config,
        )
        case2 = BenchmarkCase(
            model_name="deepnetv2",
            model_tags=["f32"],
            bench_mode=["full-inference"],
            target_arch=common_definitions.DeviceArchitecture.ARM_VALHALL,
            driver_info=IREE_DRIVERS_INFOS["iree-vulkan"],
            module_dir=pathlib.Path("case2"),
            benchmark_tool_name="tool",
            run_config=dummy_run_config,
        )
        case3 = BenchmarkCase(
            model_name="deepnetv3",
            model_tags=["f32"],
            bench_mode=["full-inference"],
            target_arch=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
            driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu-sync"],
            module_dir=pathlib.Path("case3"),
            benchmark_tool_name="tool",
            run_config=dummy_run_config,
        )
        suite = BenchmarkSuite([case1, case2, case3])

        cpu_and_gpu_benchmarks = suite.filter_benchmarks(
            available_drivers=["local-task", "vulkan"],
            available_loaders=["embedded-elf"],
            target_architectures=[
                common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
                common_definitions.DeviceArchitecture.ARM_VALHALL,
            ],
            driver_filter=None,
            mode_filter=".*full-inference.*",
            model_name_filter="deepnet.*",
        )
        gpu_benchmarks = suite.filter_benchmarks(
            available_drivers=["local-task", "vulkan"],
            available_loaders=["embedded-elf"],
            target_architectures=[
                common_definitions.DeviceArchitecture.ARM_VALHALL,
            ],
            driver_filter="vulkan",
            mode_filter=".*full-inference.*",
            model_name_filter="deepnet.*",
        )
        all_benchmarks = suite.filter_benchmarks(
            available_drivers=None,
            target_architectures=None,
            driver_filter=None,
            mode_filter=None,
            model_name_filter=None,
        )

        self.assertEqual(cpu_and_gpu_benchmarks, [case1, case2])
        self.assertEqual(gpu_benchmarks, [case2])
        self.assertEqual(all_benchmarks, [case1, case2, case3])

    def test_load_from_run_configs(self):
        model_tflite = common_definitions.Model(
            id="tflite",
            name="model_tflite",
            tags=[],
            source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
            source_url="",
            entry_function="predict",
            input_types=["1xf32"],
        )
        model_tf = common_definitions.Model(
            id="tf",
            name="model_tf",
            tags=["fp32"],
            source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
            source_url="",
            entry_function="predict",
            input_types=["1xf32"],
            input_url="https://abc/inputs_npy.tgz",
            expected_output_url="https://abc/outputs_npy.tgz",
        )
        exec_config_a = iree_definitions.ModuleExecutionConfig.build(
            id="exec_a",
            tags=["defaults"],
            loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
            driver=iree_definitions.RuntimeDriver.LOCAL_SYNC,
        )
        exec_config_b = iree_definitions.ModuleExecutionConfig.build(
            id="exec_b",
            tags=["experimental"],
            loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
            driver=iree_definitions.RuntimeDriver.LOCAL_TASK,
        )
        device_spec_a = common_definitions.DeviceSpec.build(
            id="dev_a",
            device_name="a",
            architecture=common_definitions.DeviceArchitecture.RV32_GENERIC,
            host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
            device_parameters=[],
            tags=[],
        )
        device_spec_b = common_definitions.DeviceSpec.build(
            id="dev_b",
            device_name="b",
            architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
            host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
            device_parameters=[],
            tags=[],
        )
        compile_target = iree_definitions.CompileTarget(
            target_backend=iree_definitions.TargetBackend.LLVM_CPU,
            target_architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
            target_abi=iree_definitions.TargetABI.LINUX_GNU,
        )
        run_config_a = iree_definitions.E2EModelRunConfig.build(
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=iree_definitions.ImportedModel.from_model(model_tflite),
                compile_config=iree_definitions.CompileConfig.build(
                    id="1", tags=[], compile_targets=[compile_target]
                ),
            ),
            module_execution_config=exec_config_a,
            target_device_spec=device_spec_a,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        )
        run_config_b = iree_definitions.E2EModelRunConfig.build(
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=iree_definitions.ImportedModel.from_model(model_tflite),
                compile_config=iree_definitions.CompileConfig.build(
                    id="2", tags=[], compile_targets=[compile_target]
                ),
            ),
            module_execution_config=exec_config_b,
            target_device_spec=device_spec_b,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        )
        run_config_c = iree_definitions.E2EModelRunConfig.build(
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=iree_definitions.ImportedModel.from_model(model_tf),
                compile_config=iree_definitions.CompileConfig.build(
                    id="3", tags=[], compile_targets=[compile_target]
                ),
            ),
            module_execution_config=exec_config_a,
            target_device_spec=device_spec_a,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        )
        run_configs = [run_config_a, run_config_b, run_config_c]
        root_dir = pathlib.Path("root")

        suite = BenchmarkSuite.load_from_run_configs(
            run_configs=run_configs, root_benchmark_dir=root_dir
        )

        loaded_run_configs = [case.run_config for case in suite.filter_benchmarks()]
        self.assertEqual(
            loaded_run_configs,
            [
                run_config_a,
                run_config_b,
                run_config_c,
            ],
        )
        run_config_c_case_dir = pathlib.Path(
            iree_artifacts.get_module_dir_path(
                run_config_c.module_generation_config, root_dir
            )
        )
        self.assertEqual(
            suite.filter_benchmarks(
                target_architectures=[
                    common_definitions.DeviceArchitecture.RV32_GENERIC
                ],
                model_name_filter="model_tf.*fp32",
                mode_filter="defaults",
            ),
            [
                BenchmarkCase(
                    model_name=model_tf.name,
                    model_tags=model_tf.tags,
                    bench_mode=exec_config_a.tags,
                    target_arch=common_definitions.DeviceArchitecture.RV32_GENERIC,
                    driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu-sync"],
                    benchmark_tool_name="iree-benchmark-module",
                    module_dir=run_config_c_case_dir,
                    input_uri=model_tf.input_url,
                    expected_output_uri=model_tf.expected_output_url,
                    run_config=run_config_c,
                )
            ],
        )

    def test_load_from_run_configs_with_root_url(self):
        model_tflite = common_definitions.Model(
            id="tflite",
            name="model",
            tags=[],
            source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
            source_url="",
            entry_function="predict",
            input_types=["1xf32"],
        )
        exec_config_a = iree_definitions.ModuleExecutionConfig.build(
            id="exec_a",
            tags=["defaults"],
            loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
            driver=iree_definitions.RuntimeDriver.LOCAL_SYNC,
        )
        device_spec_a = common_definitions.DeviceSpec.build(
            id="dev_a",
            device_name="a",
            architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
            host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
            device_parameters=[],
            tags=[],
        )
        compile_target = iree_definitions.CompileTarget(
            target_backend=iree_definitions.TargetBackend.LLVM_CPU,
            target_architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
            target_abi=iree_definitions.TargetABI.LINUX_GNU,
        )
        run_config_a = iree_definitions.E2EModelRunConfig.build(
            module_generation_config=iree_definitions.ModuleGenerationConfig.build(
                imported_model=iree_definitions.ImportedModel.from_model(model_tflite),
                compile_config=iree_definitions.CompileConfig.build(
                    id="1", tags=[], compile_targets=[compile_target]
                ),
            ),
            module_execution_config=exec_config_a,
            target_device_spec=device_spec_a,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        )

        suite = BenchmarkSuite.load_from_run_configs(
            run_configs=[run_config_a],
            root_benchmark_dir="https://example.com/testdata",
        )

        self.assertEqual(
            suite.filter_benchmarks(),
            [
                BenchmarkCase(
                    model_name=model_tflite.name,
                    model_tags=model_tflite.tags,
                    bench_mode=exec_config_a.tags,
                    target_arch=common_definitions.DeviceArchitecture.RV64_GENERIC,
                    driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu-sync"],
                    benchmark_tool_name="iree-benchmark-module",
                    module_dir="https://example.com/testdata/iree_module_model_tflite___riscv_64-generic-linux_gnu-llvm_cpu___/",
                    run_config=run_config_a,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
