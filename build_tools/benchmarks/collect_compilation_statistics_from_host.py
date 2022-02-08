#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Collects compilation statistics from the host machine."""

import argparse
import os
import json

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Sequence

from common.benchmark_definition import (BenchmarkOrStatisticInfo,
                                         StatisticInstance, StatisticResults,
                                         execute_cmd_and_get_output)
from common.benchmark_suite import (BENCHMARK_SUITE_REL_PATH,
                                    compose_info_object,
                                    filter_benchmarks_for_category)


class StatisticTrait(Enum):
  # This statistic won't change with different IREE drivers
  DRIVER_AGONISTIC = 0


# A class representing a statistic.
@dataclass
class Stat:
  # Hierarchical description of the current statistic.
  breadcrumb: Sequence[str]
  # Traits of the current statistic.
  trait: StatisticTrait


# A list controlling which statistics to collect and publish.
# This also serves as another level of control in case name changes. If so
# we can turn this into a map to make sure they still match the old ones.
KNOWN_STATISTICS = [
    Stat(("stream-aggregate", "synchronization", "await-count"),
         StatisticTrait.DRIVER_AGONISTIC),
    Stat(("stream-aggregate", "execution", "transient-memory-size"),
         StatisticTrait.DRIVER_AGONISTIC),
    Stat(("stream-aggregate", "execution", "dispatch-count"),
         StatisticTrait.DRIVER_AGONISTIC),
    Stat(("stream-aggregate", "executable", "executable-count"),
         StatisticTrait.DRIVER_AGONISTIC),
]


def get_git_commit_hash(commit: str) -> str:
  return execute_cmd_and_get_output(['git', 'rev-parse', commit],
                                    cwd=os.path.dirname(
                                        os.path.realpath(__file__)))


def recursively_get_from_dict(json_object: Dict[str, Any],
                              keys: Sequence[str]) -> Any:
  """Gets the value from the given dict in a recursive way using keys."""
  if len(keys) == 1:
    return json_object.get(keys[0])
  if keys[0] not in json_object:
    return None
  return recursively_get_from_dict(json_object.get(keys[0]), keys[1:])


def collect_compilation_statistics(root_build_dir: str,
                                   verbose: bool = False
                                  ) -> List[StatisticInstance]:
  """Collects compilation statistics from the benchmark suite.

  Args:
    root_build_dir: the root build directory containing the built benchmark
      suites.
    verbose: whether to print additional debug information.

  Returns:
    A list of statistic dictionaries.
  """
  root_benchmark_dir = os.path.join(root_build_dir, BENCHMARK_SUITE_REL_PATH)

  statistics = []

  for directory in sorted(os.listdir(root_benchmark_dir)):
    benchmark_category_dir = os.path.join(root_benchmark_dir, directory)
    matched_benchmarks = filter_benchmarks_for_category(
        benchmark_category_dir=benchmark_category_dir,
        cpu_target_arch_filter=".*",
        gpu_target_arch_filter=".*",
        driver_filter=None,
        verbose=verbose)

    for benchmark_case_dir in matched_benchmarks:
      stats_file = os.path.join(benchmark_case_dir,
                                "compilation_statistics.json")
      if not os.path.isfile(stats_file):
        continue
      with open(stats_file) as f:
        json_object = json.loads(f.read())

      for statistic in KNOWN_STATISTICS:
        value = recursively_get_from_dict(json_object, statistic.breadcrumb)
        if value is None:
          continue
        info = compose_info_object(
            benchmark_category_dir=benchmark_category_dir,
            benchmark_case_dir=benchmark_case_dir,
            device_info=None,
            statistic=statistic.breadcrumb,
            ignore_driver=(statistic.trait == StatisticTrait.DRIVER_AGONISTIC))
        statistics.append(StatisticInstance(info, value))

  return statistics


def parse_arguments():
  """Parses command-line options."""

  def check_dir_path(path):
    if os.path.isdir(path):
      return path
    else:
      raise argparse.ArgumentTypeError(path)

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "build_dir",
      metavar="<build-dir>",
      type=check_dir_path,
      help="Path to the build directory containing benchmark suites")
  parser.add_argument("--output",
                      "-o",
                      default=None,
                      help="Path to the output file")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")

  args = parser.parse_args()

  return args


def main(args):

  results = StatisticResults()
  commit = get_git_commit_hash("HEAD")
  results.set_commit(commit)

  stats = collect_compilation_statistics(root_build_dir=args.build_dir,
                                         verbose=args.verbose)
  for stat in stats:
    results.statistics.append(stat)

  if args.output is not None:
    with open(args.output, "w") as f:
      f.write(results.to_json_str())

  if args.verbose:
    print(results.commit)
    print(results.statistics)


if __name__ == "__main__":
  main(parse_arguments())
