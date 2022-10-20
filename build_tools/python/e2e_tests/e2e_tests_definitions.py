## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Classes to define e2e tests."""

from dataclasses import dataclass
from typing import List
import dataclasses

from e2e_test_framework.definitions import common_definitions, iree_definitions


@dataclass(frozen=True)
class E2EModuleTestConfig(object):
  """Define an e2e module test."""
  # Test name shown in the test rule.
  name: str
  module_generation_config: iree_definitions.ModuleGenerationConfig
  module_execution_config: iree_definitions.ModuleExecutionConfig
  input_data: common_definitions.ModelInputData
  # Can be either a string literal or "@{file path}".
  expected_output: str
  # Extra flags for `iree-run-module`.
  extra_test_flags: List[str] = dataclasses.field(default_factory=list)
