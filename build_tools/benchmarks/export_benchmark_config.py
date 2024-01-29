#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exports JSON config for benchmarking and compilation statistics.

Export type: "execution" outputs:
[
  <target device name>: {
    host_environment: HostEnvironment,
    module_dir_paths: [<paths of dependent module directories>],
    run_configs: serialized [E2EModelRunConfig]
  },
  ...
]
to be used in build_tools/benchmarks/run_benchmarks_on_*.py

Export type: "compilation" outputs:
{
  module_dir_paths: [<paths of dependent module directories>],
  generation_configs: serialized [ModuleGenerationConfig]
}
of generation configs defined for compilation statistics, to be used in
build_tools/benchmarks/collect_compilation_statistics.py
"""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

from typing import Dict, Iterable, List, Optional, Set, Sequence
import argparse
import collections
import dataclasses
import json
import textwrap

from benchmark_suites.iree import benchmark_collections, benchmark_presets
from e2e_test_artifacts import iree_artifacts
from e2e_test_framework import serialization
from e2e_test_framework.definitions import iree_definitions


def filter_and_group_run_configs(
    run_configs: List[iree_definitions.E2EModelRunConfig],
    target_device_names: Optional[Set[str]] = None,
    presets: Optional[Set[str]] = None,
) -> Dict[str, List[iree_definitions.E2EModelRunConfig]]:
    """Filters run configs and groups by target device name.

    Args:
      run_configs: source e2e model run configs.
      target_device_names: list of target device names, includes all if not set.
      presets: set of presets, matches all if not set.

    Returns:
      A map of e2e model run configs keyed by target device name.
    """
    grouped_run_config_map = collections.defaultdict(list)

    for run_config in run_configs:
        device_name = run_config.target_device_spec.device_name
        if target_device_names is not None and device_name not in target_device_names:
            continue
        if presets is not None and not presets.intersection(run_config.presets):
            continue
        grouped_run_config_map[device_name].append(run_config)

    return grouped_run_config_map


def _get_distinct_module_dir_paths(
    module_generation_configs: Iterable[iree_definitions.ModuleGenerationConfig],
    root_path: pathlib.PurePath = pathlib.PurePath(),
) -> List[str]:
    module_dir_paths = (
        str(iree_artifacts.get_module_dir_path(config, root_path=root_path))
        for config in module_generation_configs
    )
    return sorted(set(module_dir_paths))


def _export_execution_handler(
    presets: Optional[Sequence[str]] = None,
    target_device_names: Optional[Sequence[str]] = None,
    shard_count: Optional[Dict[str, int]] = None,
    **_unused_args,
):
    _, all_run_configs = benchmark_collections.generate_benchmarks()
    target_device_name_set = (
        None if target_device_names is None else set(target_device_names)
    )
    grouped_run_config_map = filter_and_group_run_configs(
        all_run_configs,
        target_device_names=target_device_name_set,
        presets=None if presets is None else set(presets),
    )

    shard_count = {} if shard_count is None else shard_count
    default_shard_count = shard_count.get("default", 1)

    output_map = {}
    for device_name, run_configs in grouped_run_config_map.items():
        host_environments = set(
            run_config.target_device_spec.host_environment for run_config in run_configs
        )
        if len(host_environments) > 1:
            raise ValueError(
                "Device specs of the same device should have the same host environment."
            )
        host_environment = host_environments.pop()

        current_shard_count = int(shard_count.get(device_name, default_shard_count))
        # This splits the `run_configs` list into `current_shard_count` sub-lists in a round-robin way.
        # Example: current_shard_count = 3; run_configs = range(10); assert(sharded_run_configs == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
        sharded_run_configs = [
            run_configs[shard_idx::current_shard_count]
            for shard_idx in range(current_shard_count)
        ]

        for index, shard in enumerate(sharded_run_configs):
            distinct_module_dir_paths = _get_distinct_module_dir_paths(
                config.module_generation_config for config in shard
            )

            serialized_run_configs = serialization.serialize_and_pack(shard)
            output_map.setdefault(
                device_name,
                {
                    "host_environment": dataclasses.asdict(host_environment),
                    "shards": [],
                },
            )
            output_map[device_name]["shards"].append(
                {
                    "index": index,
                    "module_dir_paths": distinct_module_dir_paths,
                    "run_configs": serialized_run_configs,
                }
            )

    return output_map


def _export_compilation_handler(
    presets: Optional[Sequence[str]] = None, **_unused_args
):
    all_gen_configs, _ = benchmark_collections.generate_benchmarks()

    if presets is None:
        presets = benchmark_presets.ALL_COMPILATION_PRESETS
    preset_set = set(presets)

    compile_stats_gen_configs = [
        gen_config
        for gen_config in all_gen_configs
        if preset_set.intersection(gen_config.presets)
    ]

    distinct_module_dir_paths = _get_distinct_module_dir_paths(
        compile_stats_gen_configs
    )

    return {
        "module_dir_paths": distinct_module_dir_paths,
        "generation_configs": serialization.serialize_and_pack(
            compile_stats_gen_configs
        ),
    }


def _parse_and_strip_list_argument(arg: str) -> List[str]:
    return [part.strip() for part in arg.split(",") if part != ""]


def _parse_benchmark_presets(arg: str, available_presets: Sequence[str]) -> List[str]:
    presets = []
    for preset in _parse_and_strip_list_argument(arg):
        if preset not in available_presets:
            raise argparse.ArgumentTypeError(
                f"Unrecognized benchmark preset: '{preset}'."
            )
        presets.append(preset)
    return presets


def _parse_shard_count(arg: str):
    return dict(map(str.strip, el.split("=", 1)) for el in arg.split(","))


def _parse_arguments():
    """Parses command-line options."""

    # Makes global options come *after* command.
    # See https://stackoverflow.com/q/23296695
    subparser_base = argparse.ArgumentParser(add_help=False)
    subparser_base.add_argument(
        "--output", type=pathlib.Path, help="Path to write the JSON output."
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
      Export type: "execution" outputs:
      [
        <target device name>: {
          host_environment: HostEnvironment,
          module_dir_paths: [<paths of dependent module directories>],
          run_configs: serialized [E2EModelRunConfig]
        },
        ...
      ]
      to be used in build_tools/benchmarks/run_benchmarks_on_*.py

      Export type: "compilation" outputs:
      {
        module_dir_paths: [<paths of dependent module directories>],
        generation_configs: serialized [ModuleGenerationConfig]
      }
      of generation configs defined for compilation statistics, to be used in
      build_tools/benchmarks/collect_compilation_statistics.py
      """
        ),
    )

    subparser = parser.add_subparsers(required=True, title="export type")
    execution_parser = subparser.add_parser(
        "execution",
        parents=[subparser_base],
        help="Export execution config to run benchmarks.",
    )
    execution_parser.set_defaults(handler=_export_execution_handler)
    execution_parser.add_argument(
        "--target_device_names",
        type=_parse_and_strip_list_argument,
        help=(
            "Target device names, separated by comma, not specified means "
            "including all devices."
        ),
    )
    execution_parser.add_argument(
        "--presets",
        "--benchmark_presets",
        type=lambda arg: _parse_benchmark_presets(
            arg, benchmark_presets.ALL_EXECUTION_PRESETS
        ),
        help=(
            "Presets that select a bundle of benchmarks, separated by comma, "
            "multiple presets will be union. Available options: "
            f"{','.join(benchmark_presets.ALL_EXECUTION_PRESETS)}"
        ),
    )
    execution_parser.add_argument(
        "--shard_count",
        type=_parse_shard_count,
        default={},
        help="Accepts a comma-separated list of device-name to shard-count mappings. Use reserved keyword 'default' for setting a default shard count: c2-standard-60=3,default=2",
    )

    compilation_parser = subparser.add_parser(
        "compilation",
        parents=[subparser_base],
        help=(
            "Export serialized list of module generation configs defined for "
            "compilation statistics."
        ),
    )
    compilation_parser.set_defaults(handler=_export_compilation_handler)
    compilation_parser.add_argument(
        "--presets",
        "--benchmark_presets",
        type=lambda arg: _parse_benchmark_presets(
            arg, benchmark_presets.ALL_COMPILATION_PRESETS
        ),
        help=(
            "Presets `comp-stats*` that select a bundle of compilation"
            " benchmarks, separated by comma, multiple presets will be union."
            " Available options: "
            f"{','.join(benchmark_presets.ALL_COMPILATION_PRESETS)}"
        ),
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    output_obj = args.handler(**vars(args))
    json_data = json.dumps(output_obj, indent=2)
    if args.output is None:
        print(json_data)
    else:
        args.output.write_text(json_data)


if __name__ == "__main__":
    main(_parse_arguments())
