## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generator that generates CMake rules to build artifacts.

The rules will build required artifacts to run e2e tests and benchmarks.
"""

from typing import List
import pathlib

from e2e_test_artifacts.cmake_rule_generators import common_generators, iree_generators
import e2e_test_artifacts.artifacts
import e2e_test_artifacts.cmake_rule_generators.utils as cmake_rule_generators_utils


def generate_rules(
    root_path: pathlib.PurePath,
    root_directory: e2e_test_artifacts.artifacts.RootDirectory) -> List[str]:
  """Generates cmake rules to build benchmarks.
  
  Args:
    root_path: root directory to store all artifacts.
  Returns:
    List of CMake rules.
  """

  model_rule_map = common_generators.generate_model_rule_map(
      root_path=root_path, model_artifact_map=root_directory.model_artifact_map)

  iree_rules = iree_generators.generate_rules(
      root_path=root_path,
      iree_model_dir_map=root_directory.iree_model_dir_map,
      model_rule_map=model_rule_map)

  # Currently the rules are simple so the common rules can be always put at the
  # top. Need a topological sort once the dependency gets complicated.
  all_model_rules: List[cmake_rule_generators_utils.CMakeRule] = list(
      model_rule_map.values())
  return [rule.get_rule() for rule in all_model_rules + iree_rules]
