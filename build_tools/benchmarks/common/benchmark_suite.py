# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for handling the benchmark suite.

See docs/developers/developing_iree/benchmark_suites.md for how to build the
benchmark suite.
"""

import pathlib
import re

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from common.benchmark_definition import IREE_DRIVERS_INFOS, DriverInfo
from e2e_test_artifacts import iree_artifacts
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework import serialization

MODEL_FLAGFILE_NAME = "flagfile"
MODEL_TOOLFILE_NAME = "tool"


@dataclass
class BenchmarkCase:
    """Represents a benchmark case.

    model_name: the source model, e.g., 'MobileSSD'.
    model_tags: the source model tags, e.g., ['f32'].
    bench_mode: the benchmark mode, e.g., '1-thread,big-core'.
    target_arch: the target CPU/GPU architature.
    driver_info: the IREE driver configuration.
    benchmark_tool_name: the benchmark tool, e.g., 'iree-benchmark-module'.
    benchmark_case_dir: the path to benchmark case directory.
    run_config: the run config from e2e test framework.
    input_uri: URI to find the input npy.
    expected_output_uri: URI to find the expected output npy.
    """

    model_name: str
    model_tags: Sequence[str]
    bench_mode: Sequence[str]
    target_arch: common_definitions.DeviceArchitecture
    driver_info: DriverInfo
    benchmark_tool_name: str
    benchmark_case_dir: pathlib.Path
    run_config: iree_definitions.E2EModelRunConfig
    input_uri: Optional[str] = None
    expected_output_uri: Optional[str] = None
    verify_params: List[str] = dataclasses.field(default_factory=list)


# A map from execution config to driver info. This is temporary during migration
# before we can drop the DriverInfo.
EXECUTION_CONFIG_TO_DRIVER_INFO_KEY_MAP: Dict[
    Tuple[iree_definitions.RuntimeDriver, iree_definitions.RuntimeLoader], str
] = {
    (
        iree_definitions.RuntimeDriver.LOCAL_TASK,
        iree_definitions.RuntimeLoader.EMBEDDED_ELF,
    ): "iree-llvm-cpu",
    (
        iree_definitions.RuntimeDriver.LOCAL_SYNC,
        iree_definitions.RuntimeLoader.EMBEDDED_ELF,
    ): "iree-llvm-cpu-sync",
    (
        iree_definitions.RuntimeDriver.LOCAL_TASK,
        iree_definitions.RuntimeLoader.VMVX_MODULE,
    ): "iree-vmvx",
    (
        iree_definitions.RuntimeDriver.LOCAL_SYNC,
        iree_definitions.RuntimeLoader.VMVX_MODULE,
    ): "iree-vmvx-sync",
    (
        iree_definitions.RuntimeDriver.VULKAN,
        iree_definitions.RuntimeLoader.NONE,
    ): "iree-vulkan",
    (
        iree_definitions.RuntimeDriver.CUDA,
        iree_definitions.RuntimeLoader.NONE,
    ): "iree-cuda",
}


class BenchmarkSuite(object):
    """Represents the benchmarks in benchmark suite directory."""

    def __init__(self, benchmark_cases: Sequence[BenchmarkCase]):
        """Construct a benchmark suite.

        Args:
          benchmark_cases: list of benchmark cases.
        """
        self.benchmark_cases = list(benchmark_cases)

    def filter_benchmarks(
        self,
        available_drivers: Optional[Sequence[str]] = None,
        available_loaders: Optional[Sequence[str]] = None,
        target_architectures: Optional[
            Sequence[common_definitions.DeviceArchitecture]
        ] = None,
        benchmark_name_filter: str = ".*",
        driver_filter: str = ".*",
        mode_filter: str = ".*",
        model_name_filter: str = ".*",
    ) -> Sequence[BenchmarkCase]:
        """Filters benchmarks.
        Args:
          available_drivers: list of drivers supported by the tools. None means to
            match any driver.
          available_loaders: list of executable loaders supported by the tools.
            None means to match any loader.
          target_architectures: list of target architectures to be included. None
            means no filter.
          driver_filter: driver filter regex.
          mode_filter: benchmark mode regex.
          model_name_filter: model name regex.
        Returns:
          A list of matched benchmark cases.
        """

        chosen_cases = []
        for benchmark_case in self.benchmark_cases:
            driver_info = benchmark_case.driver_info

            driver_name = driver_info.driver_name
            if available_drivers is not None and driver_name not in available_drivers:
                continue

            if re.match(driver_filter, driver_name) is None:
                continue

            if (
                driver_info.loader_name
                and available_loaders is not None
                and driver_info.loader_name not in available_loaders
            ):
                continue

            if (
                target_architectures is not None
                and benchmark_case.target_arch not in target_architectures
            ):
                continue

            bench_mode = ",".join(benchmark_case.bench_mode)
            if re.match(mode_filter, bench_mode) is None:
                continue

            model_name_with_tags = benchmark_case.model_name
            if len(benchmark_case.model_tags) > 0:
                model_name_with_tags += f"-{','.join(benchmark_case.model_tags)}"
            if re.match(model_name_filter, model_name_with_tags) is None:
                continue

            if re.match(benchmark_name_filter, benchmark_case.run_config.name) is None:
                continue

            chosen_cases.append(benchmark_case)

        return chosen_cases

    @staticmethod
    def load_from_run_configs(
        run_configs: Sequence[iree_definitions.E2EModelRunConfig],
        root_benchmark_dir: pathlib.Path,
    ):
        """Loads the benchmarks from the run configs.

        Args:
          run_configs: list of benchmark run configs.
        Returns:
          A benchmark suite.
        """

        benchmark_cases = []
        for run_config in run_configs:
            module_gen_config = run_config.module_generation_config
            module_exec_config = run_config.module_execution_config
            target_device_spec = run_config.target_device_spec

            driver_info_key = EXECUTION_CONFIG_TO_DRIVER_INFO_KEY_MAP.get(
                (module_exec_config.driver, module_exec_config.loader)
            )
            if driver_info_key is None:
                raise ValueError(
                    f"Can't map execution config to driver info: {module_exec_config}."
                )
            driver_info = IREE_DRIVERS_INFOS[driver_info_key]

            target_arch = target_device_spec.architecture
            model = module_gen_config.imported_model.model

            module_dir_path = iree_artifacts.get_module_dir_path(
                module_generation_config=module_gen_config, root_path=root_benchmark_dir
            )
            module_dir_path = pathlib.Path(module_dir_path)

            benchmark_case = BenchmarkCase(
                model_name=model.name,
                model_tags=model.tags,
                bench_mode=module_exec_config.tags,
                target_arch=target_arch,
                driver_info=driver_info,
                benchmark_tool_name=run_config.tool.value,
                benchmark_case_dir=module_dir_path,
                input_uri=model.input_data,
                expected_output_uri=model.expected_output,
                verify_params=model.verify_params,
                run_config=run_config,
            )
            benchmark_cases.append(benchmark_case)

        return BenchmarkSuite(benchmark_cases=benchmark_cases)


def get_run_configs_by_target_and_shard(
    benchmark_groups: Dict, target_device_name: str, shard_index: Optional[int] = None
):
    """Returns a flat list of run_configs from `benchmark_groups`, filtered by the given `target_device_name`.
    If a `shard_index` is given, only the run configs for the given shard are returned, otherwise all the run configs are returned.
    """
    benchmark_group = benchmark_groups.get(target_device_name)
    if benchmark_group is None:
        raise ValueError(
            "Target device '{}' not found in the benchmark config.".format(
                target_device_name
            )
        )

    if shard_index is None:
        # In case no shard index was given we will run ALL benchmarks from ALL shards
        packed_run_configs = [
            shard["run_configs"] for shard in benchmark_group["shards"]
        ]
    else:
        # Otherwise we will only run the benchmarks from the given shard
        benchmark_shard = next(
            (
                shard
                for shard in benchmark_group["shards"]
                if shard["index"] == shard_index
            ),
            None,
        )
        if benchmark_shard is None:
            raise ValueError(
                "Given shard (index={}) not found in the benchmark config group. Available indexes: [{}].".format(
                    shard_index,
                    ", ".join(
                        str(shard["index"]) for shard in benchmark_group["shards"]
                    ),
                )
            )
        packed_run_configs = [benchmark_shard["run_configs"]]

    # When no `shard_index` is given we might have more than one shard to process.
    # We do this by deserializing the `run_config` field from each shard separately
    # and then merge the unpacked flat lists of `E2EModelRunConfig`.
    return [
        run_config
        for packed_run_config in packed_run_configs
        for run_config in serialization.unpack_and_deserialize(
            data=packed_run_config,
            root_type=List[iree_definitions.E2EModelRunConfig],
        )
    ]
