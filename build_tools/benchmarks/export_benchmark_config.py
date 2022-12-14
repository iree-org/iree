#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exports JSON config for benchmarking.

The exported JSON is a list of object:
[
  <target device name>: {
    host_environment: HostEnvironment,
    module_dir_paths: [<paths of dependent module directories>],
    run_configs: [E2EModelRunConfig]
  },
  ...
]
"""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

from typing import Callable, Dict, List
import argparse
import collections
import dataclasses
import json

from benchmark_suites.iree import benchmark_collections
from e2e_test_framework.definitions import common_definitions
from e2e_test_framework import serialization
from e2e_test_artifacts import iree_artifacts

DeviceSpecMatcher = Callable[[common_definitions.DeviceSpec], bool]
BENCHMARK_PRESET_DEVICE_SPEC_MATCHERS: Dict[str, DeviceSpecMatcher] = {
    "x86_64":
        lambda device_spec: device_spec.architecture.architecture == "x86_64",
    "cuda":
        lambda device_spec: device_spec.architecture.architecture == "cuda",
    "android-cpu":
        lambda device_spec:
        (device_spec.architecture.type == common_definitions.ArchitectureType.
         CPU and device_spec.host_environment.platform == "android"),
    "android-gpu":
        lambda device_spec:
        (device_spec.architecture.type == common_definitions.ArchitectureType.
         GPU and device_spec.host_environment.platform == "android"),
}


def parse_arguments():
  """Parses command-line options."""

  def parse_and_strip_list_argument(arg) -> List[str]:
    return [part.strip() for part in arg.split(",")]

  parser = argparse.ArgumentParser(
      description="Exports JSON config for benchmarking. Filters can be "
      "specified jointly to select a subset of benchmarks.")
  parser.add_argument(
      "--target_device_names",
      type=parse_and_strip_list_argument,
      help=("Target device names, separated by comma, not specified means "
            "including all devices."))
  parser.add_argument(
      "--benchmark_presets",
      type=parse_and_strip_list_argument,
      help=("Presets that select a bundle of benchmarks, separated by comma, "
            "multiple presets will be union."))
  parser.add_argument("--output",
                      type=pathlib.Path,
                      help="Path to write the JSON output.")

  return parser.parse_args()


def main(args: argparse.Namespace):
  _, all_run_configs = benchmark_collections.generate_benchmarks()

  target_device_names = (set(args.target_device_names)
                         if args.target_device_names is not None else None)
  device_spec_matchers = None
  if args.benchmark_presets is not None:
    device_spec_matchers = []
    for preset in args.benchmark_presets:
      matcher = BENCHMARK_PRESET_DEVICE_SPEC_MATCHERS.get(preset)
      if matcher is None:
        raise ValueError(f"Unrecognized benchmark preset: '{preset}'.")
      device_spec_matchers.append(matcher)

  grouped_run_config_map = collections.defaultdict(list)
  for run_config in all_run_configs:
    device_name = run_config.target_device_spec.device_name
    if (target_device_names is not None and
        device_name not in target_device_names):
      continue
    if (device_spec_matchers is not None and
        not any(matcher(run_config.target_device_spec)
            for matcher in device_spec_matchers)):
      continue
    grouped_run_config_map[device_name].append(run_config)

  output_map = {}
  for device_name, run_configs in grouped_run_config_map.items():
    host_environments = set(run_config.target_device_spec.host_environment
                            for run_config in run_configs)
    if len(host_environments) > 1:
      raise ValueError(
          "Device specs of the same device should have the same host environment."
      )
    host_environment = host_environments.pop()

    all_module_dir_paths = (str(
        iree_artifacts.get_module_dir_path(config.module_generation_config))
                            for config in run_configs)
    module_dir_paths = sorted(set(all_module_dir_paths))

    output_map[device_name] = {
        "host_environment": dataclasses.asdict(host_environment),
        "module_dir_paths": module_dir_paths,
        "run_configs": serialization.serialize_and_pack(run_configs),
    }

  json_data = json.dumps(output_map, indent=2)
  if args.output is None:
    print(json_data)
  else:
    args.output.write_text(json_data)


if __name__ == "__main__":
  main(parse_arguments())
