#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Collect compilation statistics from benchmark suites.

The benchmark suites need to be built with ninja and enable the CMake option
IREE_ENABLE_COMPILATION_BENCHMARKS.
"""

import argparse
import json
import os
import re
import zipfile

from dataclasses import asdict
from typing import BinaryIO, Dict, Optional, TextIO
from pathlib import PurePath

from common.benchmark_definition import CompilationInfo, CompilationResults, CompilationStatistics, ModuleComponentSizes, get_git_commit_hash
from common.benchmark_suite import BENCHMARK_SUITE_REL_PATH, BenchmarkSuite

BENCHMARK_FLAGFILE = "flagfile"
MODULE_DIR = "vmfb"
MODULE_FILE_EXTENSION = ".vmfb"
NINJA_LOG_HEADER = "ninja log v5"
NINJA_BUILD_LOG = ".ninja_log"
COMPILATION_STATS_MODULE_SUFFIX = "compile-stats"

VM_COMPONENT_NAME = "module.fb"
CONST_COMPONENT_NAME = "_const.bin"
DISPATCH_COMPONENT_PATTERNS = [
    r".+_embedded_elf_.+\.so",
    r".+_vulkan_spirv_fb\.fb",
    r".+_vmvx_bytecode_fb\.bin",
]


def match_module_cmake_target(module_path: str) -> Optional[str]:
  # Get the last 4 parts of module path. They are expected to be:
  # benchmark_suites/<category>/vmfb/<module filename>.vmfb
  path_parts = PurePath(module_path).parts[-4:]
  if len(path_parts) < 4:
    return None
  if path_parts[0] != BENCHMARK_SUITE_REL_PATH:
    return None
  if path_parts[2] != MODULE_DIR:
    return None
  if os.path.splitext(path_parts[3])[1] != MODULE_FILE_EXTENSION:
    return None
  return os.path.join(*path_parts)


def parse_compilation_time_from_ninja_log(log: TextIO) -> Dict[str, int]:
  """Retrieve the compilation time (ms) from the Ninja build log.
  
  Returns:
    Map of target name and compilation time in ms.
  """

  target_build_time_map = {}
  header = log.readline()
  if NINJA_LOG_HEADER not in header:
    raise NotImplementedError(f"Unsupported ninja log version: {header}")

  for line in log:
    start_time, end_time, _, target, _ = line.strip().split("\t")
    cmake_target = match_module_cmake_target(target)
    if cmake_target is None:
      continue

    start_time = int(start_time)
    end_time = int(end_time)
    target_build_time_map[cmake_target] = end_time - start_time

  return target_build_time_map


def get_module_component_info(module: BinaryIO,
                              module_file_size: int) -> ModuleComponentSizes:
  with zipfile.ZipFile(module) as module_zipfile:
    size_map = dict(
        (info.filename, info.file_size) for info in module_zipfile.infolist())

  vm_component_size = size_map[VM_COMPONENT_NAME]
  const_component_size = size_map[CONST_COMPONENT_NAME]
  identified_names = {VM_COMPONENT_NAME, CONST_COMPONENT_NAME}
  total_dispatch_component_size = 0
  for filename, size in size_map.items():
    for pattern in DISPATCH_COMPONENT_PATTERNS:
      if re.match(pattern, filename):
        total_dispatch_component_size += size
        identified_names.add(filename)
        break

  if identified_names != set(size_map.keys()):
    raise RuntimeError(
        f"Unrecognized components in the module: {size_map.keys()}.")

  return ModuleComponentSizes(
      file_size=module_file_size,
      vm_component_size=vm_component_size,
      const_component_size=const_component_size,
      total_dispatch_component_size=total_dispatch_component_size)


def get_module_path(flag_file: TextIO) -> Optional[str]:
  """Retrieve the module path for compilation statistics from the flag file."""

  module_path = None
  for line in flag_file:
    match = re.match("--module_file=(.+)", line.strip())
    if match:
      module_name, module_ext = os.path.splitext(match.group(1))
      module_path = f"{module_name}-{COMPILATION_STATS_MODULE_SUFFIX}{module_ext}"
      break

  return module_path


def parse_arguments():
  """Returns an argument parser with common options."""

  def check_dir_path(path):
    if os.path.isdir(path):
      return path
    else:
      raise argparse.ArgumentTypeError(path)

  parser = argparse.ArgumentParser()
  parser.add_argument("--output",
                      required=True,
                      help="Path to output JSON file.")
  parser.add_argument(
      "build_dir",
      metavar="<build-dir>",
      type=check_dir_path,
      help="Path to the build directory containing benchmark suites.")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution.")

  return parser.parse_args()


def main(args: argparse.Namespace):
  benchmark_suite_dir = os.path.join(args.build_dir, BENCHMARK_SUITE_REL_PATH)
  benchmark_suite = BenchmarkSuite.load_from_benchmark_suite_dir(
      benchmark_suite_dir)

  with open(os.path.join(args.build_dir, NINJA_BUILD_LOG), "r") as log_file:
    target_build_time_map = parse_compilation_time_from_ninja_log(log_file)

  compilation_statistics_list = []
  for category, _ in benchmark_suite.list_categories():
    benchmark_cases = benchmark_suite.filter_benchmarks_for_category(
        category=category)
    for benchmark_case in benchmark_cases:
      flag_file_path = os.path.join(benchmark_case.benchmark_case_dir,
                                    BENCHMARK_FLAGFILE)
      with open(flag_file_path, "r") as flag_file:
        module_path = get_module_path(flag_file)

      if module_path is None:
        raise RuntimeError(
            f"Can't find the module file in the flagfile: {flag_file_path}")
      module_path = os.path.abspath(
          os.path.join(benchmark_case.benchmark_case_dir, module_path))

      with open(module_path, "rb") as module_file:
        module_component_sizes = get_module_component_info(
            module_file,
            os.stat(module_path).st_size)

      cmake_target = match_module_cmake_target(module_path)
      if cmake_target is None:
        raise RuntimeError(
            f"Module path isn't a module cmake target: {module_path}")
      compilation_time = target_build_time_map[cmake_target]

      compilation_info = CompilationInfo(model_name=benchmark_case.model_name,
                                         model_tags=benchmark_case.model_tags,
                                         model_source=category,
                                         target_arch=benchmark_case.target_arch,
                                         bench_mode=benchmark_case.bench_mode)
      compilation_statistics = CompilationStatistics(
          compilation_info=compilation_info,
          module_component_sizes=module_component_sizes,
          compilation_time=compilation_time)
      compilation_statistics_list.append(compilation_statistics)

  commit = get_git_commit_hash("HEAD")
  compilation_results = CompilationResults(
      commit=commit, compilation_statistics=compilation_statistics_list)

  json_object = asdict(compilation_results)
  with open(args.output, "w") as f:
    json.dump(json_object, f)

  if args.verbose:
    print(json.dumps(json_object, indent=4))


if __name__ == "__main__":
  main(parse_arguments())
