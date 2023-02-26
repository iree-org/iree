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

import pathlib
import sys

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import argparse
import json
import os
import re
import zipfile

from dataclasses import asdict
from typing import BinaryIO, Dict, List, Optional, TextIO

from common.benchmark_definition import CompilationInfo, CompilationResults, CompilationStatistics, ModuleComponentSizes, get_git_commit_hash
from common.benchmark_suite import BenchmarkSuite
from common import benchmark_config
from e2e_test_artifacts import iree_artifacts
from e2e_test_framework import serialization
from e2e_test_framework.definitions import iree_definitions

BENCHMARK_FLAGFILE = "flagfile"
MODULE_DIR = "vmfb"
MODULE_FILE_EXTENSION = "vmfb"
NINJA_LOG_HEADER = "ninja log v5"
NINJA_BUILD_LOG = ".ninja_log"
COMPILATION_STATS_MODULE_SUFFIX = "compile-stats"

VM_COMPONENT_NAME = "module.fb"
CONST_COMPONENT_NAME = "_const.bin"
DISPATCH_COMPONENT_PATTERNS = [
    r".+_embedded_elf_.+\.so",
    r".+_vulkan_spirv_fb\.fb",
    r".+_cuda_nvptx_fb\.fb",
    r".+_vmvx_bytecode_fb\.fb",
]


def match_module_cmake_target(module_path: pathlib.PurePath) -> Optional[str]:
  if module_path.match(f"{benchmark_config.E2E_TEST_ARTIFACTS_REL_PATH}/iree_*/"
                       f"{iree_artifacts.MODULE_FILENAME}"):
    # <e2e test artifacts dir>/iree_<module dir>/<module filename>
    path_parts = module_path.parts[-3:]
  elif module_path.match(
      f"{benchmark_config.BENCHMARK_SUITE_REL_PATH}/*/{MODULE_DIR}/"
      f"*.{MODULE_FILE_EXTENSION}"):
    # <benchmark_suites dir>/<category>/vmfb/<module filename>.vmfb
    path_parts = module_path.parts[-4:]
  else:
    return None
  # Join to get the CMake target name. This is *not* a filesystem path, so we
  # don't want \ separators on Windows that we would get with os.path.join().
  return '/'.join(path_parts)


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
    cmake_target = match_module_cmake_target(pathlib.PurePath(target))
    if cmake_target is None:
      continue

    start_time = int(start_time)
    end_time = int(end_time)
    target_build_time_map[cmake_target] = end_time - start_time

  return target_build_time_map


def get_module_component_info(module: BinaryIO,
                              module_file_bytes: int) -> ModuleComponentSizes:
  with zipfile.ZipFile(module) as module_zipfile:
    size_map = dict(
        (info.filename, info.file_size) for info in module_zipfile.infolist())

  vm_component_bytes = size_map[VM_COMPONENT_NAME]
  const_component_bytes = size_map[CONST_COMPONENT_NAME]
  identified_names = {VM_COMPONENT_NAME, CONST_COMPONENT_NAME}
  total_dispatch_component_bytes = 0
  for filename, size in size_map.items():
    for pattern in DISPATCH_COMPONENT_PATTERNS:
      if re.match(pattern, filename):
        total_dispatch_component_bytes += size
        identified_names.add(filename)
        break

  if identified_names != set(size_map.keys()):
    raise RuntimeError(
        f"Unrecognized components in the module: {size_map.keys()}.")

  return ModuleComponentSizes(
      file_bytes=module_file_bytes,
      vm_component_bytes=vm_component_bytes,
      const_component_bytes=const_component_bytes,
      total_dispatch_component_bytes=total_dispatch_component_bytes)


def get_module_path(flag_file: TextIO) -> Optional[str]:
  """Retrieve the module path for compilation statistics from the flag file."""

  module_path = None
  for line in flag_file:
    match = re.match("--module=(.+)", line.strip())
    if match:
      module_name, module_ext = os.path.splitext(match.group(1))
      module_path = f"{module_name}-{COMPILATION_STATS_MODULE_SUFFIX}{module_ext}"
      break

  return module_path


def get_module_map_from_generation_config(
    serialized_gen_config: TextIO, e2e_test_artifacts_dir: pathlib.PurePath
) -> Dict[CompilationInfo, pathlib.Path]:
  gen_configs = serialization.unpack_and_deserialize(
      data=json.load(serialized_gen_config),
      root_type=List[iree_definitions.ModuleGenerationConfig])
  module_map = {}
  for gen_config in gen_configs:
    model = gen_config.imported_model.model
    compile_config = gen_config.compile_config
    target_archs = []
    for compile_target in compile_config.compile_targets:
      arch = compile_target.target_architecture
      target_archs.append(
          (f"{arch.type.value}-{arch.architecture}-{arch.microarchitecture}-"
           f"{compile_target.target_abi.value}"))
    compilation_info = CompilationInfo(
        model_name=model.name,
        model_tags=tuple(model.tags),
        model_source=model.source_type.value,
        target_arch=f"[{','.join(target_archs)}]",
        compile_tags=tuple(compile_config.tags),
        gen_config_id=gen_config.composite_id)
    module_dir_path = iree_artifacts.get_module_dir_path(
        module_generation_config=gen_config, root_path=e2e_test_artifacts_dir)
    module_path = module_dir_path / iree_artifacts.MODULE_FILENAME
    module_map[compilation_info] = pathlib.Path(module_path)

  return module_map


def get_module_map_from_benchmark_suite(
    benchmark_suite_dir: pathlib.Path) -> Dict[CompilationInfo, pathlib.Path]:
  benchmark_suite = BenchmarkSuite.load_from_benchmark_suite_dir(
      benchmark_suite_dir)
  module_map = {}
  for category, _ in benchmark_suite.list_categories():
    benchmark_cases = benchmark_suite.filter_benchmarks_for_category(
        category=category)
    for benchmark_case in benchmark_cases:
      if benchmark_case.benchmark_case_dir is None:
        raise ValueError("benchmark_case_dir can't be None.")
      benchmark_case_dir = benchmark_case.benchmark_case_dir

      flag_file_path = benchmark_case_dir / BENCHMARK_FLAGFILE
      with flag_file_path.open("r") as flag_file:
        module_path = get_module_path(flag_file)

      if module_path is None:
        raise RuntimeError(
            f"Can't find the module file in the flagfile: {flag_file_path}")
      compilation_info = CompilationInfo(
          model_name=benchmark_case.model_name,
          model_tags=tuple(benchmark_case.model_tags),
          model_source=category,
          target_arch=benchmark_case.target_arch,
          compile_tags=tuple(benchmark_case.bench_mode))
      module_map[compilation_info] = (benchmark_case_dir /
                                      module_path).resolve()

  return module_map


def _legacy_get_module_map_and_build_log(args: argparse.Namespace):
  module_map = get_module_map_from_benchmark_suite(
      args.build_dir / benchmark_config.BENCHMARK_SUITE_REL_PATH)
  return module_map, args.build_dir / NINJA_BUILD_LOG


def _alpha_get_module_map_and_build_log(args: argparse.Namespace):
  module_map = get_module_map_from_generation_config(
      serialized_gen_config=args.generation_config.open("r"),
      e2e_test_artifacts_dir=args.e2e_test_artifacts_dir)
  return module_map, args.build_log


def _check_dir_path(path_str: str) -> pathlib.Path:
  path = pathlib.Path(path_str)
  if not path.is_dir():
    raise argparse.ArgumentTypeError(f"{path} is not a directory.")
  return path


def _check_file_path(path_str: str) -> pathlib.Path:
  path = pathlib.Path(path_str)
  if not path.is_file():
    raise argparse.ArgumentTypeError(f"{path} is not a file.")
  return path


def _parse_arguments():
  """Returns an argument parser with common options."""

  # Makes global options come *after* command.
  # See https://stackoverflow.com/q/23296695
  subparser_base = argparse.ArgumentParser(add_help=False)
  subparser_base.add_argument("--output",
                              type=pathlib.Path,
                              help="Path to output JSON file.")
  subparser_base.add_argument(
      "--verbose",
      action="store_true",
      help="Print internal information during execution.")

  parser = argparse.ArgumentParser(
      description="Collect compilation statistics from benchmark suites.")

  subparser = parser.add_subparsers(title="tool version", required=True)
  legacy_parser = subparser.add_parser("legacy",
                                       parents=[subparser_base],
                                       help="Use with legacy benchmark suites.")
  legacy_parser.set_defaults(
      get_module_map_and_build_log=_legacy_get_module_map_and_build_log)
  legacy_parser.add_argument(
      "build_dir",
      type=_check_dir_path,
      help="Path to the build directory containing benchmark suites.")

  alpha_parser = subparser.add_parser("alpha",
                                      parents=[subparser_base],
                                      help="Use with e2e test artifacts.")
  alpha_parser.set_defaults(
      get_module_map_and_build_log=_alpha_get_module_map_and_build_log)
  alpha_parser.add_argument(
      "--generation_config",
      type=_check_file_path,
      required=True,
      help="Exported module generation config of e2e test artifacts.")
  alpha_parser.add_argument("--build_log",
                            type=_check_file_path,
                            required=True,
                            help="Path to the ninja build log.")
  alpha_parser.add_argument("--e2e_test_artifacts_dir",
                            type=_check_dir_path,
                            required=True,
                            help="Path to the e2e test artifacts directory.")

  return parser.parse_args()


def main(args: argparse.Namespace):
  module_map, build_log_path = args.get_module_map_and_build_log(args)
  with build_log_path.open("r") as log_file:
    target_build_time_map = parse_compilation_time_from_ninja_log(log_file)

  compilation_statistics_list = []
  for compilation_info, module_path in module_map.items():
    with module_path.open("rb") as module_file:
      module_component_sizes = get_module_component_info(
          module_file,
          module_path.stat().st_size)

    cmake_target = match_module_cmake_target(module_path)
    if cmake_target is None:
      raise RuntimeError(
          f"Module path isn't a module cmake target: {module_path}")
    compilation_time_ms = target_build_time_map[cmake_target]
    compilation_statistics = CompilationStatistics(
        compilation_info=compilation_info,
        module_component_sizes=module_component_sizes,
        compilation_time_ms=compilation_time_ms)
    compilation_statistics_list.append(compilation_statistics)

  commit = get_git_commit_hash("HEAD")
  compilation_results = CompilationResults(
      commit=commit, compilation_statistics=compilation_statistics_list)

  json_output = json.dumps(asdict(compilation_results), indent=2)
  if args.output is None:
    print(json_output)
  else:
    args.output.write_text(json_output)


if __name__ == "__main__":
  main(_parse_arguments())
