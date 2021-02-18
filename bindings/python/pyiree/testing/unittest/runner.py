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

from typing import Type
import unittest

__all__ = [
    "TextTestResult",
    "TextTestRunner",
]


# Formatting based off of absl.testing._pretty_print_reporter.
class TextTestResult(unittest.TextTestResult):
  """TestResult that provides easier to parse text formatting."""
  separator1 = "—" * 80
  separator2 = separator1

  def __init__(self,
               stream,
               descriptions,
               verbosity,
               fail_on_unexpected_success=False,
               log_expected_failures=False):
    # Disable the superclass per-test outputs by setting verbosity=0.
    super().__init__(stream, descriptions, verbosity=0)
    self._per_test_output = verbosity > 0
    self.summary = []
    self.fail_on_unexpected_success = fail_on_unexpected_success
    self.log_expected_failures = log_expected_failures

  def _print(self, *args, **kwargs):
    print(*args, **kwargs, file=self.stream)

  def _get_unit_test_name(self, test):
    return test.id().split(".")[-1]

  def _print_status(self, tag, test, include_in_summary=True):
    status = f"[{tag}] {self._get_unit_test_name(test)}"
    if include_in_summary:
      self.summary.append(status)
    if self._per_test_output:
      self._print(status)
      self.stream.flush()

  def startTest(self, test):
    super().startTest(test)
    self._print_status("             Run ", test, include_in_summary=False)

  def addSuccess(self, test):
    super().addSuccess(test)
    self._print_status("            Pass ", test)

  def addError(self, test, err):
    super().addError(test, err)
    self._print_status(" Fail            ", test)

  def addFailure(self, test, err):
    super().addFailure(test, err)
    self._print_status(" Fail            ", test)

  def addSkip(self, test, reason):
    super().addSkip(test, reason)
    self._print_status("            Skip ", test)

  def addExpectedFailure(self, test, err):
    super().addExpectedFailure(test, err)
    self._print_status("   Expected Fail ", test)

  def addUnexpectedSuccess(self, test):
    super().addUnexpectedSuccess(test)
    self._print_status(" Unexpected Pass ", test)

  def printErrorList(self, flavour, errors):
    for test, err in errors:
      self._print(self.separator1)
      self._print(f"{flavour}: {self._get_unit_test_name(test)}")
      self._print(self.separator1)
      self._print(err)

  def printErrors(self):
    if self.dots or self.showAll:
      self.stream.writeln()
    if self.log_expected_failures:
      self.printErrorList("EXPECTED FAILURE", self.expectedFailures)
    self.printErrorList("ERROR", self.errors)
    self.printErrorList("FAILURE", self.failures)
    self.stream.flush()

  def printSummary(self):
    self._print("\n".join(self.summary))
    self.stream.flush()

  def wasSuccessful(self) -> bool:
    if len(self.unexpectedSuccesses) and self.fail_on_unexpected_success:
      return False
    else:
      return len(self.failures) == len(self.errors) == 0


class TextTestRunner(unittest.TextTestRunner):
  """Subclass TextTestRunner to use the subclassed TextTestResult's formating"""
  resultclass = TextTestResult

  def __init__(self,
               *args,
               fail_on_unexpected_success: bool = True,
               log_expected_failures: bool = False,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.fail_on_unexpected_success = fail_on_unexpected_success
    self.log_expected_failures = log_expected_failures

  def _makeResult(self) -> TextTestResult:
    return self.resultclass(self.stream, self.descriptions, self.verbosity,
                            self.fail_on_unexpected_success,
                            self.log_expected_failures)

  def runTestCase(self, test_case: Type[unittest.TestCase]) -> TextTestResult:
    print(f"Running {test_case.__name__}", file=self.stream)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
    result = self.run(suite)
    # Append newlines to make running multiple tests cases more readable.
    print("\n\n", file=self.stream)
    return result
