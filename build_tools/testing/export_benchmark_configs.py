#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""TODO"""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

from typing import List
import argparse
import json
import itertools

from benchmark_suites.iree import benchmark_collections
from e2e_test_framework import serialization
from e2e_test_artifacts import iree_artifacts


def sort_and_dedup_paths(
    values: List[pathlib.PurePath]) -> List[pathlib.PurePath]:
  values.sort()
  return list(value for value, _ in itertools.groupby(values))


def parse_arguments():
  """Parses command-line options."""

  parser = argparse.ArgumentParser()
  parser.add_argument("--target_device_name",
                      type=str,
                      help="Target device name.")
  parser.add_argument("--output_dir",
                      required=True,
                      type=pathlib.Path,
                      help="Directory to write the run config and assets list.")
  parser.add_argument("--dump_metadata",
                      default=None,
                      type=pathlib.Path,
                      help="Path to write the JSON metadata of all configs.")

  return parser.parse_args()


def main(args: argparse.Namespace):
  _, benchmark_run_configs = benchmark_collections.generate_benchmarks()

  if args.target_device_names is None:
    run_configs = benchmark_run_configs
  else:
    run_configs = [
        run_config for run_config in benchmark_run_configs
        if run_config.target_device_spec.vendor_name in args.target_device_names
    ]

  imported_model_paths = sort_and_dedup_paths([
      iree_artifacts.get_imported_model_path(
          config.module_generation_config.imported_model)
      for config in run_configs
  ])
  module_dir_paths = sort_and_dedup_paths([
      iree_artifacts.get_module_dir_path(config.module_generation_config)
      for config in run_configs
  ])

  host_environments = set(run_config.target_device_spec.host_environment
                          for run_config in run_configs)
  if len(host_environments) == 0:
    raise ValueError("No device spec is found.")
  if len(host_environments) > 1:
    raise ValueError(
        "Device specs on the same device should have the same host platform.")
  host_environment = host_environments.pop()

  metadata = {
      "platform": f"{host_environment.platform}-{host_environment.architecture}"
  }

  data = json.dumps({
      "run_configs": serialization.serialize_and_pack(run_configs),
      "imported_model_assets": [str(path) for path in imported_model_paths],
      "module_dir_assets": [str(path) for path in module_dir_paths],
  })


if __name__ == "__main__":
  main(parse_arguments())
