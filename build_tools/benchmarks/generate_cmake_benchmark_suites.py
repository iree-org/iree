#!/usr/bin/env python3
## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates a CMake file to build the benchmark suites."""

import argparse
import pathlib
import string
import suites.cmake_rule_generator

TEMPLATE_DIR = pathlib.Path(__file__).parent
GENERATED_BENCHMARK_SUITES_CMAKE_TEMPLATE = string.Template(
    open(TEMPLATE_DIR / "iree_generated_benchmark_suites_template.cmake",
         "r").read())


def parse_arguments():
  """Parses command-line options."""

  parser = argparse.ArgumentParser()
  parser.add_argument("--output",
                      required=True,
                      help="Path to write the generated cmake file.")

  return parser.parse_args()


def main(args: argparse.Namespace):
  benchmark_rules = suites.cmake_rule_generator.generate_benchmark_rules(
      model_artifacts_dir="${_MODEL_ARTIFACTS_DIR}",
      iree_artifacts_dir="${_IREE_ARTIFACTS_DIR}")
  cmake_file = GENERATED_BENCHMARK_SUITES_CMAKE_TEMPLATE.substitute(
      __BENCHMARK_RULES='\n'.join(benchmark_rules))
  with open(args.output, "w") as output_file:
    output_file.write(cmake_file)


if __name__ == "__main__":
  main(parse_arguments())
