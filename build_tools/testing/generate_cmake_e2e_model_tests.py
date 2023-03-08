#!/usr/bin/env python3
## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates a CMake file to define e2e mdoel tests."""

import string
import sys
import pathlib
import argparse

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent / ".." / "python"))

import benchmark_suites.iree.benchmark_collections
import e2e_model_tests.cmake_generator

TEMPLATE_DIR = pathlib.Path(__file__).parent
GENERATED_E2E_MODEL_TESTS_CMAKE_TEMPLATE = string.Template(
    (TEMPLATE_DIR / "generated_e2e_model_tests_template.cmake").read_text())


def parse_arguments():
  """Parses command-line options."""

  parser = argparse.ArgumentParser()
  parser.add_argument("--output",
                      required=True,
                      help="Path to write the generated cmake file.")

  return parser.parse_args()


def main(args: argparse.Namespace):
  (gen_configs,
   _) = benchmark_suites.iree.benchmark_collections.generate_benchmarks()
  cmake_rules = e2e_model_tests.cmake_generator.generate_rules(
      module_generation_configs=gen_configs)
  output = GENERATED_E2E_MODEL_TESTS_CMAKE_TEMPLATE.substitute(
      __TEST_RULES="\n".join(cmake_rules))
  with open(args.output, "w") as output_file:
    output_file.write(output)


if __name__ == "__main__":
  main(parse_arguments())
