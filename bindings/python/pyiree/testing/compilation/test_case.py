# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import textwrap
from typing import Any, Dict, List, Sequence, Tuple, Type, Union
import unittest

import pyiree as iree
from .stages import Stage
from .module import CompilationDefModule

__all__ = [
    "CompilationTestCase",
    "CompilationTestCaseSpec",
    "CompilationUnitTestSpec",
    "get_test_case_specs",
]


class CompilationUnitTestSpec:
  """Stores the information needed to (re)create a compilation unit test."""

  def __init__(self, stage: Stage, target_name: str, source_path: str,
               target_path: str, exported_names: Sequence[str], skip: bool,
               expected_failure: bool):
    self.stage = stage
    self.unit_test_name = "".join([
        f"test_compile_{stage.lower_name}",
        "_failing" if expected_failure else "",
        "__",
        target_name,
    ])
    self.source_path = source_path
    self.target_path = target_path
    self.exported_names = exported_names
    self.skip = skip
    self.expected_failure = expected_failure

  def __str__(self):
    return "\n".join([
        f"{self.unit_test_name}:",
        f"  exported_names: {self.exported_names}",
        f"  source: {self.source_path}",
        f"  target: {self.target_path}",
    ])

  def __repr__(self):
    return str(self)


class CompilationTestCaseSpec:
  """Stores the information needed to create tests of a compilation stage."""

  def __init__(self, name: str,
               unit_test_specs: Sequence[CompilationUnitTestSpec]):
    self.name = name
    self.unit_test_specs = unit_test_specs

  @classmethod
  def create(cls, stage: Stage, all_exported_names: Tuple[str],
             exported_names_to_source_path: Dict[Union[str, Tuple[str]], str],
             passing_all_stages: Tuple[str], passing_this_stage: Tuple[str],
             skipped_this_stage: Tuple[str], failing_this_stage: Tuple[str]):
    test_case_name = f"{stage.upper_name}CompilationTestCase"
    unit_tests = []
    exported_names_to_target_path = {}

    # Create a test for all expected-passing exported names.
    source_path = exported_names_to_source_path[passing_all_stages]
    target_path = stage.get_target_path(source_path)
    exported_names_to_target_path[passing_all_stages] = target_path
    unit_tests.append(
        CompilationUnitTestSpec(stage=stage,
                                target_name=stage.get_target_name(source_path),
                                source_path=source_path,
                                target_path=target_path,
                                exported_names=passing_all_stages,
                                skip=False,
                                expected_failure=False))

    # Create individual unit tests for each exported name if there are multiple.:
    for exported_name in all_exported_names:
      source_path = exported_names_to_source_path[exported_name]
      target_dir = stage.get_target_dir(source_path,
                                        nest_under_partial_lowerings=True)
      # Don't nest single functions under .methods directories.
      target_path = os.path.join(
          target_dir, f"{exported_name}.{stage.target.file_extension}")
      exported_names_to_target_path[exported_name] = target_path

      unit_tests.append(
          CompilationUnitTestSpec(stage=stage,
                                  target_name=exported_name,
                                  source_path=source_path,
                                  target_path=target_path,
                                  exported_names=[exported_name],
                                  skip=(exported_name in skipped_this_stage),
                                  expected_failure=(exported_name
                                                    in failing_this_stage)))

    return cls(test_case_name, unit_tests), exported_names_to_target_path

  def __str__(self):
    result = [f"{self.name}:"]
    for spec in self.unit_test_specs:
      result.append(textwrap.indent(str(spec), "  "))
    return "\n\n".join(result)

  def __repr__(self):
    return str(self)


class CompilationTestCase(unittest.TestCase):
  """Test the translation of one IR to another with on-disk persistance."""

  @classmethod
  def _create_unit_test(cls, spec: CompilationUnitTestSpec):

    def unit_test(self):
      spec.stage.pipeline(spec.source_path, spec.target_path,
                          spec.exported_names)

    if spec.expected_failure:
      unit_test = unittest.expectedFailure(unit_test)
    if spec.skip:
      unit_test = unittest.skip(unit_test)
    setattr(cls, spec.unit_test_name, unit_test)

  @classmethod
  def create_subclass(
      cls,
      test_case_spec: CompilationTestCaseSpec) -> Type["CompilationTestCase"]:
    TestCaseSubclass = type(test_case_spec.name, (cls,), {})
    for spec in test_case_spec.unit_test_specs:
      TestCaseSubclass._create_unit_test(spec)
    return TestCaseSubclass


def _get_coverage_info(
    module_class: Type[CompilationDefModule], stages: Sequence[Stage]
) -> Tuple[Tuple[str], Tuple[str], Dict[Stage, Tuple[str]]]:
  """Run through each stage and annotate passing, failing and skipped names."""
  all_exported_names = copy.deepcopy(module_class.exported_names)
  passing_all_stages = copy.deepcopy(all_exported_names)
  stage_to_coverage = {}

  for stage in stages:
    skipped_this_stage = set.difference(all_exported_names, passing_all_stages)
    failing_this_stage = set()
    for name in passing_all_stages:
      if stage in module_class.expected_compilation_failures.get(name, []):
        failing_this_stage.add(name)

    passing_this_stage = set.difference(passing_all_stages, failing_this_stage)
    passing_all_stages = copy.deepcopy(passing_this_stage)

    stage_to_coverage[stage] = dict(
        passing_this_stage=tuple(passing_this_stage),
        skipped_this_stage=tuple(skipped_this_stage),
        failing_this_stage=tuple(failing_this_stage))

  return tuple(all_exported_names), tuple(passing_all_stages), stage_to_coverage


def get_test_case_specs(
    module_class: Type[CompilationDefModule], stages: Sequence[Stage],
    test_dir: str) -> Tuple[List[CompilationTestCaseSpec], str]:
  """Create a sequence of test case specifications for lowering module_class."""
  # Get expected coverage info.
  exported_names, passing_all_stages, stage_to_coverage = _get_coverage_info(
      module_class, stages)

  # Set up the initial map from exported names to source files.
  initial_source_path = module_class.get_path(test_dir)
  exported_names_to_source_path = {
      name: initial_source_path for name in exported_names
  }
  exported_names_to_source_path[passing_all_stages] = initial_source_path

  # Create test cases for each stage in the lowering.
  test_case_specs = []
  for stage, stage_coverage in stage_to_coverage.items():
    test_case, exported_names_to_source_path = CompilationTestCaseSpec.create(
        stage=stage,
        all_exported_names=exported_names,
        exported_names_to_source_path=exported_names_to_source_path,
        passing_all_stages=passing_all_stages,
        **stage_coverage)
    test_case_specs.append(test_case)

  return test_case_specs, exported_names_to_source_path[passing_all_stages]
