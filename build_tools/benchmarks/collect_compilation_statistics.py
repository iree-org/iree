#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Collect compilation statistics from benchmark suites.

See https://iree.dev/developers/performance/benchmark-suites/ for how to build
the benchmark suites.
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

from dataclasses import asdict, dataclass
from typing import BinaryIO, Dict, List, Optional, TextIO

from common import benchmark_definition
from common.benchmark_definition import (
    CompilationInfo,
    CompilationResults,
    CompilationStatistics,
    ModuleComponentSizes,
    get_git_commit_hash,
)
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
E2E_TEST_ARTIFACTS_REL_PATH = "e2e_test_artifacts"

VM_COMPONENT_NAME = "module.fb"
CONST_COMPONENT_NAME = "_const.bin"
DISPATCH_COMPONENT_PATTERNS = [
    r".+_embedded_elf_.+\.so",
    r".+_vulkan_spirv_fb\.fb",
    r".+_cuda_nvptx_fb\.fb",
    r".+_vmvx_bytecode_fb\.fb",
]


@dataclass(frozen=True)
class ModuleInfo(object):
    module_path: pathlib.Path
    stream_stats_path: pathlib.Path


def match_module_cmake_target(module_path: pathlib.PurePath) -> Optional[str]:
    if module_path.match(
        f"{E2E_TEST_ARTIFACTS_REL_PATH}/iree_*/" f"{iree_artifacts.MODULE_FILENAME}"
    ):
        # <e2e test artifacts dir>/iree_<module dir>/<module filename>
        path_parts = module_path.parts[-3:]
        # Join to get the CMake target name. This is *not* a filesystem path, so we
        # don't want \ separators on Windows that we would get with os.path.join().
        return "/".join(path_parts)

    return None


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


def get_module_component_info(
    module: BinaryIO, module_file_bytes: int
) -> ModuleComponentSizes:
    with zipfile.ZipFile(module) as module_zipfile:
        size_map = dict(
            (info.filename, info.file_size) for info in module_zipfile.infolist()
        )

    identified_names = set()
    if VM_COMPONENT_NAME in size_map:
        vm_component_bytes = size_map[VM_COMPONENT_NAME]
        identified_names.add(VM_COMPONENT_NAME)
    else:
        vm_component_bytes = 0

    if CONST_COMPONENT_NAME in size_map:
        const_component_bytes = size_map[CONST_COMPONENT_NAME]
        identified_names.add(CONST_COMPONENT_NAME)
    else:
        const_component_bytes = 0

    total_dispatch_component_bytes = 0
    for filename, size in size_map.items():
        for pattern in DISPATCH_COMPONENT_PATTERNS:
            if re.match(pattern, filename):
                total_dispatch_component_bytes += size
                identified_names.add(filename)
                break

    actual_key_set = set(size_map.keys())
    if identified_names != actual_key_set:
        # With consteval, we invoke the compiler within the compiler, which
        # can yield additional temporaries.
        print(
            f"Ignoring extra components in the module: {actual_key_set-identified_names}",
            file=sys.stderr,
        )

    return ModuleComponentSizes(
        file_bytes=module_file_bytes,
        vm_component_bytes=vm_component_bytes,
        const_component_bytes=const_component_bytes,
        total_dispatch_component_bytes=total_dispatch_component_bytes,
    )


def get_module_map_from_compilation_benchmark_config(
    compilation_benchmark_config_data: TextIO, e2e_test_artifacts_dir: pathlib.PurePath
) -> Dict[CompilationInfo, ModuleInfo]:
    benchmark_config = json.load(compilation_benchmark_config_data)
    gen_configs = serialization.unpack_and_deserialize(
        data=benchmark_config["generation_configs"],
        root_type=List[iree_definitions.ModuleGenerationConfig],
    )
    module_map = {}
    for gen_config in gen_configs:
        model = gen_config.imported_model.model
        compile_config = gen_config.compile_config
        target_archs = []
        for compile_target in compile_config.compile_targets:
            arch = compile_target.target_architecture
            target_archs.append(
                (
                    f"{arch.type.value}-{arch.architecture}-{arch.microarchitecture}-"
                    f"{compile_target.target_abi.value}"
                )
            )
        compilation_info = CompilationInfo(
            name=gen_config.name,
            model_name=model.name,
            model_tags=tuple(model.tags),
            model_source=model.source_type.value,
            target_arch=f"[{','.join(target_archs)}]",
            compile_tags=tuple(compile_config.tags),
            gen_config_id=gen_config.composite_id,
        )
        module_dir_path = pathlib.Path(
            iree_artifacts.get_module_dir_path(
                module_generation_config=gen_config, root_path=e2e_test_artifacts_dir
            )
        )
        module_path = module_dir_path / iree_artifacts.MODULE_FILENAME
        stream_stats_path = module_dir_path / iree_artifacts.SCHEDULING_STATS_FILENAME
        module_map[compilation_info] = ModuleInfo(
            module_path=module_path, stream_stats_path=stream_stats_path
        )

    return module_map


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

    parser = argparse.ArgumentParser(
        description="Collect compilation statistics from benchmark suites."
    )
    parser.add_argument(
        "--compilation_benchmark_config",
        type=_check_file_path,
        required=True,
        help="Exported compilation benchmark config of e2e test artifacts.",
    )
    parser.add_argument(
        "--build_log",
        type=_check_file_path,
        required=True,
        help="Path to the ninja build log.",
    )
    parser.add_argument(
        "--e2e_test_artifacts_dir",
        type=_check_dir_path,
        required=True,
        help="Path to the e2e test artifacts directory.",
    )
    parser.add_argument("--output", type=pathlib.Path, help="Path to output JSON file.")

    return parser.parse_args()


def main(args: argparse.Namespace):
    config_data = args.compilation_benchmark_config.open("r")
    module_map = get_module_map_from_compilation_benchmark_config(
        compilation_benchmark_config_data=config_data,
        e2e_test_artifacts_dir=args.e2e_test_artifacts_dir,
    )
    build_log_path = args.build_log

    with build_log_path.open("r") as log_file:
        target_build_time_map = parse_compilation_time_from_ninja_log(log_file)

    compilation_statistics_list = []
    for compilation_info, module_info in module_map.items():
        module_path = module_info.module_path
        with module_path.open("rb") as module_file:
            module_component_sizes = get_module_component_info(
                module_file, module_path.stat().st_size
            )

        cmake_target = match_module_cmake_target(module_path)
        if cmake_target is None:
            raise RuntimeError(
                f"Module path isn't a module cmake target: {module_path}"
            )
        compilation_time_ms = target_build_time_map[cmake_target]

        stream_stats_json = json.loads(module_info.stream_stats_path.read_text())
        exec_stats_json = stream_stats_json["stream-aggregate"]["execution"]
        ir_stats = benchmark_definition.IRStatistics(
            stream_dispatch_count=exec_stats_json["dispatch-count"]
        )

        compilation_statistics = CompilationStatistics(
            compilation_info=compilation_info,
            module_component_sizes=module_component_sizes,
            compilation_time_ms=compilation_time_ms,
            ir_stats=ir_stats,
        )
        compilation_statistics_list.append(compilation_statistics)

    commit = get_git_commit_hash("HEAD")
    compilation_results = CompilationResults(
        commit=commit, compilation_statistics=compilation_statistics_list
    )

    json_output = json.dumps(asdict(compilation_results), indent=2)
    if args.output is None:
        print(json_output)
    else:
        args.output.write_text(json_output)


if __name__ == "__main__":
    main(_parse_arguments())
