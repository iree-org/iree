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

from typing import Callable, Dict, Iterable, List, Optional, Set
import argparse
import collections
import dataclasses
import json
import textwrap

from benchmark_suites.iree import benchmark_collections
from e2e_test_artifacts import iree_artifacts
from e2e_test_framework import serialization
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.definitions import iree_definitions

PresetMatcher = Callable[[iree_definitions.E2EModelRunConfig], bool]
BENCHMARK_PRESET_MATCHERS: Dict[str, PresetMatcher] = {
    "x86_64":
        lambda config: config.target_device_spec.architecture.architecture ==
        "x86_64",
    "cuda":
        lambda config: config.target_device_spec.architecture.architecture ==
        "cuda" and "long-running" not in config.tags,
    "cuda-long":
        lambda config: config.target_device_spec.architecture.architecture ==
        "cuda" and "long-running" in config.tags,
    # TODO(#9855): Enable benchmarks on Pixel-6-Pro and XT2201-2.
    "experimental-android-cpu":
        lambda config:
        (config.target_device_spec.architecture.type == common_definitions.
         ArchitectureType.CPU and config.target_device_spec.host_environment.
         platform == "android" and config.target_device_spec.device_name in
         ["Pixel-4"]),
    "experimental-android-gpu":
        lambda config:
        (config.target_device_spec.architecture.type == common_definitions.
         ArchitectureType.GPU and config.target_device_spec.host_environment.
         platform == "android"),
    # Not a preset for execution benchmarks.
    "comp-stats":
        lambda _config: False,
}


def filter_and_group_run_configs(
    run_configs: List[iree_definitions.E2EModelRunConfig],
    target_device_names: Optional[Set[str]] = None,
    preset_matchers: Optional[List[PresetMatcher]] = None
) -> Dict[str, List[iree_definitions.E2EModelRunConfig]]:
  """Filters run configs and groups by target device name.
  
  Args:
    run_configs: source e2e model run configs.
    target_device_names: list of target device names, includes all if not set.
    preset_matchers: list of preset matcher, matches all if not set.

  Returns:
    A map of e2e model run configs keyed by target device name.
  """
  grouped_run_config_map = collections.defaultdict(list)

  for run_config in run_configs:
    device_name = run_config.target_device_spec.device_name
    if (target_device_names is not None and
        device_name not in target_device_names):
      continue
    if (preset_matchers is not None and
        not any(matcher(run_config) for matcher in preset_matchers)):
      continue
    grouped_run_config_map[device_name].append(run_config)

  return grouped_run_config_map


def _get_distinct_module_dir_paths(
    module_generation_configs: Iterable[
        iree_definitions.ModuleGenerationConfig],
    root_path: pathlib.PurePath = pathlib.PurePath()
) -> List[str]:
  module_dir_paths = (str(
      iree_artifacts.get_module_dir_path(config, root_path=root_path))
                      for config in module_generation_configs)
  return sorted(set(module_dir_paths))


def _export_execution_handler(args: argparse.Namespace):
  _, all_run_configs = benchmark_collections.generate_benchmarks()
  target_device_names = (set(args.target_device_names)
                         if args.target_device_names is not None else None)
  grouped_run_config_map = filter_and_group_run_configs(
      all_run_configs,
      target_device_names=target_device_names,
      preset_matchers=args.benchmark_presets)

  output_map = {}
  for device_name, run_configs in grouped_run_config_map.items():
    host_environments = set(run_config.target_device_spec.host_environment
                            for run_config in run_configs)
    if len(host_environments) > 1:
      raise ValueError(
          "Device specs of the same device should have the same host environment."
      )
    host_environment = host_environments.pop()

    distinct_module_dir_paths = _get_distinct_module_dir_paths(
        config.module_generation_config for config in run_configs)

    output_map[device_name] = {
        "host_environment": dataclasses.asdict(host_environment),
        "module_dir_paths": distinct_module_dir_paths,
        "run_configs": serialization.serialize_and_pack(run_configs),
    }

  return output_map


def _export_compilation_handler(_args: argparse.Namespace):
  all_gen_configs, _ = benchmark_collections.generate_benchmarks()
  compile_stats_gen_configs = [
      config for config in all_gen_configs
      if benchmark_collections.COMPILE_STATS_TAG in config.compile_config.tags
  ]

  distinct_module_dir_paths = _get_distinct_module_dir_paths(
      compile_stats_gen_configs)

  return {
      "module_dir_paths":
          distinct_module_dir_paths,
      "generation_configs":
          serialization.serialize_and_pack(compile_stats_gen_configs)
  }


def _parse_and_strip_list_argument(arg) -> List[str]:
  return [part.strip() for part in arg.split(",")]


def _parse_benchmark_presets(arg) -> List[PresetMatcher]:
  matchers = []
  for preset in _parse_and_strip_list_argument(arg):
    matcher = BENCHMARK_PRESET_MATCHERS.get(preset)
    if matcher is None:
      raise argparse.ArgumentTypeError(
          f"Unrecognized benchmark preset: '{preset}'.")
    matchers.append(matcher)
  return matchers


def _parse_arguments():
  """Parses command-line options."""

  # Makes global options come *after* command.
  # See https://stackoverflow.com/q/23296695
  subparser_base = argparse.ArgumentParser(add_help=False)
  subparser_base.add_argument("--output",
                              type=pathlib.Path,
                              help="Path to write the JSON output.")

  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description=textwrap.dedent("""
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
      """))

  subparser = parser.add_subparsers(required=True, title="export type")
  execution_parser = subparser.add_parser(
      "execution",
      parents=[subparser_base],
      help="Export execution config to run benchmarks.")
  execution_parser.set_defaults(handler=_export_execution_handler)
  execution_parser.add_argument(
      "--target_device_names",
      type=_parse_and_strip_list_argument,
      help=("Target device names, separated by comma, not specified means "
            "including all devices."))
  execution_parser.add_argument(
      "--benchmark_presets",
      type=_parse_benchmark_presets,
      help=("Presets that select a bundle of benchmarks, separated by comma, "
            "multiple presets will be union. Available options: "
            f"{','.join(BENCHMARK_PRESET_MATCHERS.keys())}"))

  compilation_parser = subparser.add_parser(
      "compilation",
      parents=[subparser_base],
      help=("Export serialized list of module generation configs defined for "
            "compilation statistics."))
  compilation_parser.set_defaults(handler=_export_compilation_handler)

  return parser.parse_args()


def main(args: argparse.Namespace):
  output_obj = args.handler(args)
  json_data = json.dumps(output_obj, indent=2)
  if args.output is None:
    print(json_data)
  else:
    args.output.write_text(json_data)


if __name__ == "__main__":
  main(_parse_arguments())
