#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import pathlib
import tempfile
import unittest
from typing import Sequence
from common.benchmark_definition import IREE_DRIVERS_INFOS
from common.benchmark_suite import BenchmarkCase, BenchmarkSuite
from e2e_test_framework.definitions import common_definitions, iree_definitions


class BenchmarkSuiteTest(unittest.TestCase):

  def test_list_categories(self):
    suite = BenchmarkSuite({
        pathlib.Path("suite/TFLite"): [],
        pathlib.Path("suite/PyTorch"): [],
    })

    self.assertEqual(suite.list_categories(),
                     [("PyTorch", pathlib.Path("suite/PyTorch")),
                      ("TFLite", pathlib.Path("suite/TFLite"))])

  def test_filter_benchmarks_for_category(self):
    case1 = BenchmarkCase(model_name="deepnet",
                          model_tags=[],
                          bench_mode=["1-thread", "full-inference"],
                          target_arch="CPU-ARMv8",
                          driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu"],
                          benchmark_case_dir=pathlib.Path("case1"),
                          benchmark_tool_name="tool")
    case2 = BenchmarkCase(model_name="deepnetv2",
                          model_tags=["f32"],
                          bench_mode=["full-inference"],
                          target_arch="GPU-Mali",
                          driver_info=IREE_DRIVERS_INFOS["iree-vulkan"],
                          benchmark_case_dir=pathlib.Path("case2"),
                          benchmark_tool_name="tool")
    case3 = BenchmarkCase(model_name="deepnetv3",
                          model_tags=["f32"],
                          bench_mode=["full-inference"],
                          target_arch="CPU-x86_64",
                          driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu-sync"],
                          benchmark_case_dir=pathlib.Path("case3"),
                          benchmark_tool_name="tool")
    suite = BenchmarkSuite({
        pathlib.Path("suite/TFLite"): [case1, case2, case3],
    })

    cpu_and_gpu_benchmarks = suite.filter_benchmarks_for_category(
        category="TFLite",
        available_drivers=["local-task", "vulkan"],
        available_loaders=["embedded-elf"],
        cpu_target_arch_filter="cpu-armv8",
        gpu_target_arch_filter="gpu-mali",
        driver_filter=None,
        mode_filter=".*full-inference.*",
        model_name_filter="deepnet.*")
    gpu_benchmarks = suite.filter_benchmarks_for_category(
        category="TFLite",
        available_drivers=["local-task", "vulkan"],
        available_loaders=["embedded-elf"],
        cpu_target_arch_filter="cpu-unknown",
        gpu_target_arch_filter="gpu-mali",
        driver_filter="vulkan",
        mode_filter=".*full-inference.*",
        model_name_filter="deepnet.*/case2")
    all_benchmarks = suite.filter_benchmarks_for_category(
        category="TFLite",
        available_drivers=None,
        cpu_target_arch_filter=None,
        gpu_target_arch_filter=None,
        driver_filter=None,
        mode_filter=None,
        model_name_filter=None)

    self.assertEqual(cpu_and_gpu_benchmarks, [case1, case2])
    self.assertEqual(gpu_benchmarks, [case2])
    self.assertEqual(all_benchmarks, [case1, case2, case3])

  def test_filter_benchmarks_for_nonexistent_category(self):
    suite = BenchmarkSuite({
        pathlib.Path("suite/TFLite"): [],
    })

    benchmarks = suite.filter_benchmarks_for_category(
        category="PyTorch",
        available_drivers=[],
        available_loaders=[],
        cpu_target_arch_filter="ARMv8",
        gpu_target_arch_filter="Mali-G78")

    self.assertEqual(benchmarks, [])

  def test_load_from_benchmark_suite_dir(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_dir = pathlib.Path(tmp_dir)
      tflite_dir = tmp_dir / "TFLite"
      pytorch_dir = tmp_dir / "PyTorch"
      BenchmarkSuiteTest.__create_bench(tflite_dir,
                                        model_name="DeepNet",
                                        model_tags=["f32"],
                                        bench_mode=["4-thread", "full"],
                                        target_arch="CPU-ARMv8",
                                        config="iree-llvm-cpu",
                                        tool="run-cpu-bench")
      case2 = BenchmarkSuiteTest.__create_bench(pytorch_dir,
                                                model_name="DeepNetv2",
                                                model_tags=[],
                                                bench_mode=["full-inference"],
                                                target_arch="GPU-Mali",
                                                config="iree-vulkan",
                                                tool="run-gpu-bench")

      suite = BenchmarkSuite.load_from_benchmark_suite_dir(tmp_dir)

      self.assertEqual(suite.list_categories(), [("PyTorch", pytorch_dir),
                                                 ("TFLite", tflite_dir)])
      self.assertEqual(
          suite.filter_benchmarks_for_category(
              category="PyTorch",
              available_drivers=["vulkan"],
              available_loaders=[],
              cpu_target_arch_filter="cpu-armv8",
              gpu_target_arch_filter="gpu-mali"), [case2])

  def test_load_from_run_configs(self):
    model_tflite = common_definitions.Model(
        id="tflite",
        name="model_tflite",
        tags=[],
        source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
        source_url="",
        entry_function="predict",
        input_types=["1xf32"])
    model_tf = common_definitions.Model(
        id="tf",
        name="model_tf",
        tags=["fp32"],
        source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
        source_url="",
        entry_function="predict",
        input_types=["1xf32"])
    exec_config_a = iree_definitions.ModuleExecutionConfig(
        id="exec_a",
        tags=["defaults"],
        loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
        driver=iree_definitions.RuntimeDriver.LOCAL_SYNC)
    exec_config_b = iree_definitions.ModuleExecutionConfig(
        id="exec_b",
        tags=["experimental"],
        loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
        driver=iree_definitions.RuntimeDriver.LOCAL_TASK)
    device_spec_a = common_definitions.DeviceSpec(
        id="dev_a",
        device_name="a",
        architecture=common_definitions.DeviceArchitecture.RV32_GENERIC,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
        device_parameters=[])
    device_spec_b = common_definitions.DeviceSpec(
        id="dev_b",
        device_name="b",
        architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
        device_parameters=[])
    compile_target = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
        target_abi=iree_definitions.TargetABI.LINUX_GNU)
    run_config_a = iree_definitions.E2EModelRunConfig.with_flag_generation(
        module_generation_config=iree_definitions.ModuleGenerationConfig.
        with_flag_generation(
            imported_model=iree_definitions.ImportedModel.from_model(
                model_tflite),
            compile_config=iree_definitions.CompileConfig(
                id="1", tags=[], compile_targets=[compile_target])),
        module_execution_config=exec_config_a,
        target_device_spec=device_spec_a,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)
    run_config_b = iree_definitions.E2EModelRunConfig.with_flag_generation(
        module_generation_config=iree_definitions.ModuleGenerationConfig.
        with_flag_generation(
            imported_model=iree_definitions.ImportedModel.from_model(
                model_tflite),
            compile_config=iree_definitions.CompileConfig(
                id="2", tags=[], compile_targets=[compile_target])),
        module_execution_config=exec_config_b,
        target_device_spec=device_spec_b,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)
    run_config_c = iree_definitions.E2EModelRunConfig.with_flag_generation(
        module_generation_config=iree_definitions.ModuleGenerationConfig.
        with_flag_generation(
            imported_model=iree_definitions.ImportedModel.from_model(model_tf),
            compile_config=iree_definitions.CompileConfig(
                id="3", tags=[], compile_targets=[compile_target])),
        module_execution_config=exec_config_a,
        target_device_spec=device_spec_a,
        input_data=common_definitions.ZEROS_MODEL_INPUT_DATA,
        tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE)
    run_configs = [run_config_a, run_config_b, run_config_c]

    suite = BenchmarkSuite.load_from_run_configs(run_configs=run_configs)

    self.assertEqual(suite.list_categories(),
                     [("exported_tf_v2", pathlib.Path("exported_tf_v2")),
                      ("exported_tflite", pathlib.Path("exported_tflite"))])
    self.assertEqual(
        suite.filter_benchmarks_for_category(category="exported_tflite"), [
            BenchmarkCase(model_name=model_tflite.name,
                          model_tags=model_tflite.tags,
                          bench_mode=exec_config_a.tags,
                          target_arch="cpu-riscv_32-generic",
                          driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu-sync"],
                          benchmark_tool_name="iree-benchmark-module",
                          benchmark_case_dir=None,
                          run_config=run_config_a),
            BenchmarkCase(model_name=model_tflite.name,
                          model_tags=model_tflite.tags,
                          bench_mode=exec_config_b.tags,
                          target_arch="cpu-riscv_64-generic",
                          driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu"],
                          benchmark_tool_name="iree-benchmark-module",
                          benchmark_case_dir=None,
                          run_config=run_config_b)
        ])
    self.assertEqual(
        suite.filter_benchmarks_for_category(
            category="exported_tf_v2",
            cpu_target_arch_filter="cpu-riscv_32-generic",
            model_name_filter="model_tf.*fp32",
            mode_filter="defaults"),
        [
            BenchmarkCase(model_name=model_tf.name,
                          model_tags=model_tf.tags,
                          bench_mode=exec_config_a.tags,
                          target_arch="cpu-riscv_32-generic",
                          driver_info=IREE_DRIVERS_INFOS["iree-llvm-cpu-sync"],
                          benchmark_tool_name="iree-benchmark-module",
                          benchmark_case_dir=None,
                          run_config=run_config_c)
        ])
    self.assertEqual(
        suite.filter_benchmarks_for_category(
            category="exported_tf_v2",
            cpu_target_arch_filter="cpu-riscv_32-generic",
            mode_filter="experimental"), [])

  @staticmethod
  def __create_bench(dir_path: pathlib.Path, model_name: str,
                     model_tags: Sequence[str], bench_mode: Sequence[str],
                     target_arch: str, config: str, tool: str):
    case_name = f"{config}__{target_arch}__{','.join(bench_mode)}"
    model_name_with_tags = model_name
    if len(model_tags) > 0:
      model_name_with_tags += f"-{','.join(model_tags)}"
    bench_path = dir_path / model_name_with_tags / case_name
    bench_path.mkdir(parents=True)
    (bench_path / "tool").write_text(tool)

    return BenchmarkCase(model_name=model_name,
                         model_tags=model_tags,
                         bench_mode=bench_mode,
                         target_arch=target_arch,
                         driver_info=IREE_DRIVERS_INFOS[config],
                         benchmark_case_dir=bench_path,
                         benchmark_tool_name=tool)


if __name__ == "__main__":
  unittest.main()
