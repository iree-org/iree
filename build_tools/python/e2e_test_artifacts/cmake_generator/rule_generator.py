## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generator that generates CMake rules to build artifacts.

The rules will build required artifacts to run e2e tests and benchmarks.
"""

from typing import List
import itertools
import pathlib

from e2e_test_artifacts.cmake_generator import model_rule_generator, iree_rule_generator
import e2e_test_artifacts.artifacts


def generate_rules(
    package_name: str, root_path: pathlib.PurePath,
    artifacts_root: e2e_test_artifacts.artifacts.ArtifactsRoot) -> List[str]:
  """Generates cmake rules to build artifacts.
  
  Args:
    package_name: root cmake package name.
    root_path: root directory to store all artifacts.
    artifacts_root: artifact root to be generated.
  Returns:
    List of cmake rules.
  """

  model_rule_map = model_rule_generator.generate_model_rule_map(
      root_path=root_path, artifacts_root=artifacts_root.model_artifacts_root)
  model_cmake_rules = list(
      itertools.chain.from_iterable(
          rule.cmake_rules for rule in model_rule_map.values()))

  iree_cmake_rules = iree_rule_generator.generate_rules(
      package_name=package_name,
      root_path=root_path,
      artifacts_root=artifacts_root.iree_artifacts_root,
      model_rule_map=model_rule_map)

  # Currently the rules are simple so the model rules can be always put at the
  # top. Need a topological sort once the dependency gets complicated.
  return model_cmake_rules + iree_cmake_rules
