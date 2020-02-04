# Copyright 2019 Google LLC
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

"""Bazel macros for running lit tests."""

def iree_lit_test(
        name,
        test_file,
        data,
        size = "small",
        driver = "//iree/tools:run_lit.sh",
        **kwargs):
    """Creates a lit test from the specified source file.

    Args:
      name: name of the generated test suite.
      test_file: the test file with the lit test
      data: binaries used in the lit tests.
      size: size of the tests.
      driver: the shell runner for the lit tests.
      **kwargs: Any additional arguments that will be passed to the underlying sh_test.
    """
    native.sh_test(
        name = name,
        srcs = [driver],
        size = size,
        data = data + [test_file],
        args = ["$(location %s)" % (test_file,)],
        **kwargs
    )

def iree_lit_test_suite(
        name,
        data,
        srcs,
        size = "small",
        driver = "//iree/tools:run_lit.sh",
        tags = [],
        **kwargs):
    """Creates one lit test per source file and a test suite that bundles them.

    Args:
      name: name of the generated test suite.
      data: binaries used in the lit tests.
      srcs: test file sources.
      size: size of the tests.
      driver: the shell runner for the lit tests.
      tags: tags to apply to the test. Note that as in standard test suites, manual
            is treated specially and will also apply to the test suite itself.
      **kwargs: Any additional arguments that will be passed to the underlying tests.
    """
    tests = []
    for test_file in srcs:
        # It's generally good practice to prefix any generated names with the
        # macro name, but we're trying to match the style of the names that are
        # used for LLVM internally.
        test_name = "%s.test" % (test_file)
        iree_lit_test(
            name = test_name,
            test_file = test_file,
            size = size,
            data = data,
            driver = driver,
            tags = tags,
            **kwargs
        )
        tests.append(test_name)

    native.test_suite(
        name = name,
        tests = tests,
        # Note that only the manual tag really has any effect here. Others are
        # used for test suite filtering, but all tests are passed the same tags.
        tags = tags,
    )
