#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import stat
import unittest
import os
import tempfile
from common.common_arguments import build_common_argument_parser
from common.benchmark_definition import BenchmarkInfo, DeviceInfo, PlatformType
from common.benchmark_suite import BenchmarkCase, BenchmarkConfig, BenchmarkHelper


class BenchmarkConfigTest(unittest.TestCase):

  def setUp(self):
    self.build_dir = tempfile.TemporaryDirectory()
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.normal_tool_dir = os.path.join(self.build_dir.name, "normal_tool")
    os.mkdir(self.normal_tool_dir)
    self.traced_tool_dir = os.path.join(self.build_dir.name, "traced_tool")
    os.mkdir(self.traced_tool_dir)
    self.trace_capture_tool = tempfile.NamedTemporaryFile()
    os.chmod(self.trace_capture_tool.name, stat.S_IEXEC)

  def tearDown(self):
    self.tmp_dir.cleanup()
    self.build_dir.cleanup()

  def test_build_from_args(self):
    args = build_common_argument_parser().parse_args([
        f"--tmp_dir={self.tmp_dir.name}",
        f"--normal_benchmark_tool_dir={self.normal_tool_dir}",
        f"--traced_benchmark_tool_dir={self.traced_tool_dir}",
        f"--trace_capture_tool={self.trace_capture_tool.name}",
        f"--capture_tarball=tarball_dir", f"--driver_filter_regex=a",
        f"--model_name_regex=b", f"--mode_regex=c", f"--keep_going",
        f"--benchmark_min_time=10", self.build_dir.name
    ])

    config = BenchmarkConfig.build(args=args,
                                   git_commit_hash="abcd",
                                   skip_benchmarks={"1"},
                                   skip_captures={"2"})

    per_commit_tmp_dir = os.path.join(self.tmp_dir.name, "abcd")
    self.assertEqual(
        config,
        BenchmarkConfig(tmp_dir=per_commit_tmp_dir,
                        root_benchmark_dir=os.path.join(self.build_dir.name,
                                                        "benchmark_suites"),
                        benchmark_results_dir=os.path.join(
                            per_commit_tmp_dir, "benchmark-results"),
                        capture_dir=os.path.join(per_commit_tmp_dir,
                                                 "captures"),
                        normal_benchmark_tool_dir=self.normal_tool_dir,
                        traced_benchmark_tool_dir=self.traced_tool_dir,
                        trace_capture_tool=self.trace_capture_tool.name,
                        skip_benchmarks={"1"},
                        skip_captures={"2"},
                        driver_filter="a",
                        model_name_filter="b",
                        mode_filter="c",
                        do_capture=True,
                        keep_going=True,
                        benchmark_min_time=10))

  def test_build_from_args_benchmark_only(self):
    args = build_common_argument_parser().parse_args([
        f"--tmp_dir={self.tmp_dir.name}",
        f"--normal_benchmark_tool_dir={self.normal_tool_dir}",
        self.build_dir.name
    ])

    config = BenchmarkConfig.build(args=args, git_commit_hash="abcd")

    self.assertFalse(config.do_capture)

  def test_build_from_args_invalid_capture_args(self):
    args = build_common_argument_parser().parse_args([
        f"--tmp_dir={self.tmp_dir.name}",
        f"--normal_benchmark_tool_dir={self.normal_tool_dir}",
        f"--traced_benchmark_tool_dir={self.traced_tool_dir}",
        self.build_dir.name
    ])

    self.assertRaises(
        ValueError,
        lambda: BenchmarkConfig.build(args=args, git_commit_hash="abcd"))


class BenchmarkHelperTest(unittest.TestCase):

  @staticmethod
  def _create_bench_dir(dir_path: str, compilation_flag: str, flag: str,
                        tool: str):
    os.makedirs(dir_path)
    with open(os.path.join(dir_path, "compilation_flagfile"), "w") as f:
      f.write(compilation_flag)
    with open(os.path.join(dir_path, "flag"), "w") as f:
      f.write(flag)
    with open(os.path.join(dir_path, "tool"), "w") as f:
      f.write(tool)

  def setUp(self):
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.root_dir = tempfile.TemporaryDirectory()

    os.mkdir(os.path.join(self.root_dir.name, "TFLite"))
    os.mkdir(os.path.join(self.root_dir.name, "PyTorch"))

    self.bench_dir1 = os.path.join(
        self.root_dir.name, "TFLite", "DeepLabV3-fp32",
        "iree-dylib__CPU-ARM64-v8A__1-thread,big-core,full-inference,default-flags"
    )
    BenchmarkHelperTest._create_bench_dir(self.bench_dir1,
                                          compilation_flag="test",
                                          flag="abcd",
                                          tool="iree-benchmark-module")
    self.bench_dir2 = os.path.join(
        self.root_dir.name, "TFLite", "DeepLabV3-fp32",
        "iree-dylib__CPU-x86__1-thread,big-core,full-inference,default-flags")
    BenchmarkHelperTest._create_bench_dir(self.bench_dir2,
                                          compilation_flag="test",
                                          flag="abcd",
                                          tool="iree-benchmark-module")
    self.bench_dir3 = os.path.join(
        self.root_dir.name, "PyTorch", "MobileBertSquad-fp32",
        "iree-dylib__CPU-ARM64-v8A__4-thread,big-core,full-inference,default-flags"
    )
    BenchmarkHelperTest._create_bench_dir(self.bench_dir3,
                                          compilation_flag="test",
                                          flag="abcd",
                                          tool="iree-benchmark-module")
    self.bench_dir4 = os.path.join(
        self.root_dir.name, "PyTorch", "MobileBertSquad-fp32",
        "iree-vulkan__GPU-Mali-G78__full-inference,default-flags")
    BenchmarkHelperTest._create_bench_dir(self.bench_dir4,
                                          compilation_flag="test",
                                          flag="abcd",
                                          tool="iree-benchmark-module")

    self.normal_benchmark_tool_dir = os.path.join(self.tmp_dir.name,
                                                  "normal_tool")
    os.makedirs(self.normal_benchmark_tool_dir)
    with open(
        os.path.join(self.normal_benchmark_tool_dir, "iree-benchmark-module"),
        "w") as f:
      f.write("dummy")

    self.traced_benchmark_tool_dir = os.path.join(self.tmp_dir.name,
                                                  "traced_tool")
    os.makedirs(self.traced_benchmark_tool_dir)
    with open(
        os.path.join(self.traced_benchmark_tool_dir, "iree-benchmark-module"),
        "w") as f:
      f.write("dummy")

    self.config = BenchmarkConfig(
        tmp_dir=self.tmp_dir.name,
        root_benchmark_dir=self.root_dir.name,
        benchmark_results_dir=os.path.join(self.tmp_dir.name, "results"),
        capture_dir=os.path.join(self.tmp_dir.name, "captures"),
        normal_benchmark_tool_dir=self.normal_benchmark_tool_dir,
        traced_benchmark_tool_dir=self.traced_benchmark_tool_dir,
        trace_capture_tool="",
        driver_filter=None,
        model_name_filter=None,
        mode_filter=None,
        skip_benchmarks=set(),
        skip_captures=set(),
        do_capture=False,
        keep_going=False,
        benchmark_min_time=0)

    self.device_info = DeviceInfo(PlatformType.LINUX, "Unknown", "arm64-v8a",
                                  ["sha2"], "Mali-G78")

  def tearDown(self) -> None:
    self.root_dir.cleanup()
    self.tmp_dir.cleanup()

  def test_list_benchmark_categories(self):
    helper = BenchmarkHelper(self.config, self.device_info)
    self.assertEqual(helper.list_benchmark_categories(), ["PyTorch", "TFLite"])

  def test_generate_benchmark_cases_cpu(self):
    helper = BenchmarkHelper(self.config, self.device_info)

    cases = helper.generate_benchmark_cases(category="TFLite",
                                            cpu_target_arch="cpu-arm64",
                                            gpu_target_arch="none",
                                            available_drivers=["dylib"])

    benchmark_key = "DeepLabV3 [fp32] (TFLite) 1-thread,big-core,full-inference,default-flags with IREE-Dylib @ Unknown (CPU-ARMv8-A)"
    self.assertEqual(cases, [
        BenchmarkCase(
            benchmark_info=BenchmarkInfo(model_name="DeepLabV3",
                                         model_tags=["fp32"],
                                         model_source="TFLite",
                                         bench_mode=[
                                             "1-thread", "big-core",
                                             "full-inference", "default-flags"
                                         ],
                                         runner="iree-dylib",
                                         device_info=self.device_info),
            benchmark_key=benchmark_key,
            benchmark_case_dir=self.bench_dir1,
            normal_benchmark_tool_path=os.path.join(
                self.normal_benchmark_tool_dir, "iree-benchmark-module"),
            traced_benchmark_tool_path=None,
            flagfile_path=os.path.join(self.bench_dir1, "flagfile"),
            benchmark_results_filename=os.path.join(
                self.tmp_dir.name, "results", f"{benchmark_key}.json"),
            capture_filename=os.path.join(self.tmp_dir.name, "captures",
                                          f"{benchmark_key}.tracy"),
            skip_normal_benchmark=False,
            skip_traced_benchmark=True)
    ])

  def test_generate_benchmark_cases_gpu(self):
    helper = BenchmarkHelper(self.config, self.device_info)

    cases = helper.generate_benchmark_cases(
        category="PyTorch",
        cpu_target_arch="none",
        gpu_target_arch="gpu-mali-g78",
        available_drivers=["dylib", "vulkan"])

    benchmark_key = "MobileBertSquad [fp32] (PyTorch) full-inference,default-flags with IREE-Vulkan @ Unknown (GPU-Mali-G78)"
    self.assertEqual(cases, [
        BenchmarkCase(
            benchmark_info=BenchmarkInfo(
                model_name="MobileBertSquad",
                model_tags=["fp32"],
                model_source="PyTorch",
                bench_mode=["full-inference", "default-flags"],
                runner="iree-vulkan",
                device_info=self.device_info),
            benchmark_key=benchmark_key,
            benchmark_case_dir=self.bench_dir4,
            normal_benchmark_tool_path=os.path.join(
                self.normal_benchmark_tool_dir, "iree-benchmark-module"),
            traced_benchmark_tool_path=None,
            flagfile_path=os.path.join(self.bench_dir4, "flagfile"),
            benchmark_results_filename=os.path.join(
                self.tmp_dir.name, "results", f"{benchmark_key}.json"),
            capture_filename=os.path.join(self.tmp_dir.name, "captures",
                                          f"{benchmark_key}.tracy"),
            skip_normal_benchmark=False,
            skip_traced_benchmark=True)
    ])

  def test_generate_benchmark_cases_capture(self):
    self.config.do_capture = True
    helper = BenchmarkHelper(self.config, self.device_info)

    cases = helper.generate_benchmark_cases(category="TFLite",
                                            cpu_target_arch="cpu-arm64",
                                            gpu_target_arch="none",
                                            available_drivers=["dylib"])

    benchmark_key = "DeepLabV3 [fp32] (TFLite) 1-thread,big-core,full-inference,default-flags with IREE-Dylib @ Unknown (CPU-ARMv8-A)"
    self.assertEqual(cases, [
        BenchmarkCase(
            benchmark_info=BenchmarkInfo(model_name="DeepLabV3",
                                         model_tags=["fp32"],
                                         model_source="TFLite",
                                         bench_mode=[
                                             "1-thread", "big-core",
                                             "full-inference", "default-flags"
                                         ],
                                         runner="iree-dylib",
                                         device_info=self.device_info),
            benchmark_key=benchmark_key,
            benchmark_case_dir=self.bench_dir1,
            normal_benchmark_tool_path=os.path.join(
                self.normal_benchmark_tool_dir, "iree-benchmark-module"),
            traced_benchmark_tool_path=os.path.join(
                self.traced_benchmark_tool_dir, "iree-benchmark-module"),
            flagfile_path=os.path.join(self.bench_dir1, "flagfile"),
            benchmark_results_filename=os.path.join(
                self.tmp_dir.name, "results", f"{benchmark_key}.json"),
            capture_filename=os.path.join(self.tmp_dir.name, "captures",
                                          f"{benchmark_key}.tracy"),
            skip_normal_benchmark=False,
            skip_traced_benchmark=False)
    ])

  def test_generate_benchmark_cases_skip_bench(self):
    benchmark_key = "DeepLabV3 [fp32] (TFLite) 1-thread,big-core,full-inference,default-flags with IREE-Dylib @ Unknown (CPU-ARMv8-A)"
    self.config.do_capture = True
    self.config.skip_benchmarks = {benchmark_key}
    helper = BenchmarkHelper(self.config, self.device_info)

    cases = helper.generate_benchmark_cases(category="TFLite",
                                            cpu_target_arch="cpu-arm64",
                                            gpu_target_arch="none",
                                            available_drivers=["dylib"])

    self.assertEqual(len(cases), 1)
    self.assertEqual(cases[0].skip_normal_benchmark, True)
    self.assertEqual(cases[0].skip_traced_benchmark, False)

  def test_generate_benchmark_cases_cpu_and_gpu(self):
    helper = BenchmarkHelper(self.config, self.device_info)

    cases = helper.generate_benchmark_cases(
        category="PyTorch",
        cpu_target_arch="cpu-arm64",
        gpu_target_arch="gpu-mali-g78",
        available_drivers=["dylib", "vulkan"])

    self.assertEqual(len(cases), 2)

  def test_generate_benchmark_cases_with_filter(self):
    self.config.driver_filter = "dylib"
    self.config.model_name_filter = "MobileBertSquad.+"
    self.config.mode_filter = ".*full-inference.*"
    helper = BenchmarkHelper(self.config, self.device_info)

    cases = helper.generate_benchmark_cases(
        category="PyTorch",
        cpu_target_arch="cpu-arm64",
        gpu_target_arch="gpu-mali-g78",
        available_drivers=["dylib", "vulkan"])

    self.assertEqual(len(cases), 1)
    self.assertEqual(
        cases[0].benchmark_key,
        "MobileBertSquad [fp32] (PyTorch) 4-thread,big-core,full-inference,default-flags with IREE-Dylib @ Unknown (CPU-ARMv8-A)"
    )


if __name__ == "__main__":
  unittest.main()
