#!/usr/bin/env python3
## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates a CMake file to build test artifacts."""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import argparse
import itertools

from e2e_test_artifacts.cmake_generator import model_rule_generator, iree_rule_generator
from e2e_test_artifacts import iree_artifacts
import benchmark_suites.iree.benchmark_collections

# CMake variable name to store IREE package name.
PACKAGE_NAME_CMAKE_VARIABLE = "PACKAGE_NAME"
ROOT_ARTIFACTS_DIR_CMAKE_VARIABLE = "ROOT_ARTIFACTS_DIR"

GENERATED_E2E_TEST_FETCH_MODELS_CMAKE_FILENAMAE = "generated_e2e_test_fetch_models.cmake"
GENERATED_E2E_TEST_IREE_ARTIFACTS_CMAKE_FILENAME = "generated_e2e_test_iree_artifacts.cmake"


def parse_arguments():
  """Parses command-line options."""

  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir",
                      required=True,
                      help="Dir path to write the generated cmake files.")

  return parser.parse_args()


def main(args: argparse.Namespace):
  # Currently benchmark is the only source of module generation configs.
  (iree_module_generation_configs, iree_e2e_model_run_configs
  ) = benchmark_suites.iree.benchmark_collections.generate_benchmarks()

  dependent_model_map = iree_artifacts.get_dependent_model_map(
      iree_module_generation_configs)

  root_path = pathlib.PurePath("${%s}" % ROOT_ARTIFACTS_DIR_CMAKE_VARIABLE)
  model_rule_map = model_rule_generator.generate_model_rule_map(
      root_path=root_path, models=dependent_model_map.values())

  output_dir = pathlib.Path(args.output_dir)
  fetch_models_cmake_file = output_dir / GENERATED_E2E_TEST_FETCH_MODELS_CMAKE_FILENAMAE
  model_cmake_rules = itertools.chain.from_iterable(
      rule.cmake_rules for rule in model_rule_map.values())
  fetch_models_cmake_file.write_text("\n".join(model_cmake_rules))

  package_name = "${%s}" % PACKAGE_NAME_CMAKE_VARIABLE
  iree_cmake_rules = iree_rule_generator.generate_rules(
      package_name=package_name,
      root_path=root_path,
      module_generation_configs=iree_module_generation_configs,
      e2e_model_run_configs=iree_e2e_model_run_configs,
      model_rule_map=model_rule_map)

  (output_dir / GENERATED_E2E_TEST_IREE_ARTIFACTS_CMAKE_FILENAME).write_text(
      "\n".join(iree_cmake_rules))


if __name__ == "__main__":
  main(parse_arguments())
