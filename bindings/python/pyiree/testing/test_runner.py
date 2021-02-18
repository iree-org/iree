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

import sys
from typing import Optional, Sequence, Tuple, Type

from absl import flags

from .unittest import TextTestResult, TextTestRunner
from .compilation.module import CompilationDefModule
from .compilation.test_case import CompilationTestCase, get_test_case_specs
from .lowerings import Lowering

flags.DEFINE_string("lowering", "reference",
                    "The compilation lowering to test.")
FLAGS = flags.FLAGS

__all__ = [
    "run_compilation_tests",
]


def _print_test_results(test_results: Sequence[TextTestResult]):
  print("Compilation Test Results:", file=sys.stderr)
  for result in test_results:
    result.printSummary()
  sys.stderr.flush()


def run_compilation_tests(
    module_class: Type[CompilationDefModule],
    lowering: Lowering,
    test_dir: str = None,
    fail_on_unexpected_success: bool = True,
    log_expected_failures: bool = False
) -> Tuple[Sequence[TextTestResult], str]:
  """Runs a sequence of compilation tests to lower the given module's methods.

  Args:
    module_class:
      the CompilationDefModule to lower via lowering. This specifies which
      methods are expected to fail for each stage, and which of the methods to
      consider for compilation.
    lowering:
      the the sequence of compilation stages to lower 'module_class' through.
    test_dir:
      the path to the directory to store the generated test artifacts in.
    fail_on_unexpected_success:
      controls whether unexpected successful compilation stages should be
      treated by the test runner as failures (and halt further testing).
    log_expected_failures:
      controls whether the errors from expected failures are suppressed.

  Returns:
    A sequence of TextTestResults containing the outcomes of each unit test, and
    a string contianing the path to the final representation of 'module_class'
    specified in 'lowering'.
  """
  test_case_specs, lowered_path = get_test_case_specs(module_class,
                                                      lowering.stages, test_dir)
  for spec in test_case_specs:
    print(spec)
    print()
  sys.stdout.flush()

  # Save the module to create the initial source file.
  module_class.save(test_dir)

  test_runner = TextTestRunner(
      fail_on_unexpected_success=fail_on_unexpected_success,
      log_expected_failures=log_expected_failures)
  test_results = []
  for spec in test_case_specs:
    TestCase = CompilationTestCase.create_subclass(spec)
    test_results.append(test_runner.runTestCase(TestCase))
    if not test_results[-1].wasSuccessful():
      _print_test_results(test_results)
      sys.exit(1)

  _print_test_results(test_results)
  return test_results, lowered_path
