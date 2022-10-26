#!/usr/bin/env python3
## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates a CMake file to build the benchmark suites."""

import string
import sys
import pathlib
import argparse

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent / ".." / "python"))

from e2e_test_artifacts.cmake_generator import rule_generator
import e2e_test_artifacts.artifacts

TEMPLATE_DIR = pathlib.Path(__file__).parent
GENERATED_BENCHMARK_SUITES_CMAKE_TEMPLATE = string.Template(
    (TEMPLATE_DIR /
     "iree_generated_benchmark_suites_template.cmake").read_text())
# CMake variable name to store IREE package name.
PACKAGE_NAME_CMAKE_VARIABLE = "_PACKAGE_NAME"
ROOT_ARTIFACTS_DIR_CMAKE_VARIABLE = "_ROOT_ARTIFACTS_DIR"


def parse_arguments():
  """Parses command-line options."""

  parser = argparse.ArgumentParser()
  parser.add_argument("--output",
                      required=True,
                      help="Path to write the generated cmake file.")

  return parser.parse_args()


def main(args: argparse.Namespace):
  artifacts_root = (
      e2e_test_artifacts.artifacts.generate_default_artifacts_root())
  cmake_rules = rule_generator.generate_rules(
      package_name=f"${{{PACKAGE_NAME_CMAKE_VARIABLE}}}",
      root_path=pathlib.PurePath(f"${{{ROOT_ARTIFACTS_DIR_CMAKE_VARIABLE}}}"),
      artifacts_root=artifacts_root)

  cmake_file = GENERATED_BENCHMARK_SUITES_CMAKE_TEMPLATE.substitute(
      __ROOT_ARTIFACTS_DIR_VARIABLE=ROOT_ARTIFACTS_DIR_CMAKE_VARIABLE,
      __PACKAGE_NAME_VARIABLE=PACKAGE_NAME_CMAKE_VARIABLE,
      __BENCHMARK_RULES='\n'.join(cmake_rules))
  with open(args.output, "w") as output_file:
    output_file.write(cmake_file)


if __name__ == "__main__":
  main(parse_arguments())
