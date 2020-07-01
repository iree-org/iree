# Copyright 2020 Google LLC
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

"""Macro for building e2e keras vision model tests."""

load("//bindings/python:build_defs.oss.bzl", "iree_py_test")

def iree_train_test_suite(
        name,
        configurations,
        tags = None,
        deps = None,
        size = None,
        python_version = "PY3",
        **kwargs):
    """Creates one iree_py_test per configuration tuple and a test suite that bundles them.

    Args:
      name:
        name of the generated test suite.
      configurations:
        a list of tuples of (optimizer, backends).
      tags:
        tags to apply to the test. Note that as in standard test suites, manual
        is treated specially and will also apply to the test suite itself.
      deps:
        test dependencies.
      size:
        size of the tests.
      python_version:
        the python version to run the tests with. Uses python3 by default.
      **kwargs:
        Any additional arguments that will be passed to the underlying tests
        and test_suite.
    """
    tests = []
    for optimizer, backends in configurations:
        test_name = "{}_{}_{}_test".format(name, optimizer, backends)
        tests.append(test_name)

        args = [
            "--optimizer_name={}".format(optimizer),
            "--target_backends={}".format(backends),
        ]

        iree_py_test(
            name = test_name,
            main = "model_train_test.py",
            srcs = ["model_train_test.py"],
            args = args,
            tags = tags,
            deps = deps,
            size = size,
            python_version = python_version,
            **kwargs
        )

    native.test_suite(
        name = name,
        tests = tests,
        # Note that only the manual tag really has any effect here. Others are
        # used for test suite filtering, but all tests are passed the same tags.
        tags = tags,
        # If there are kwargs that need to be passed here which only apply to
        # the generated tests and not to test_suite, they should be extracted
        # into separate named arguments.
        **kwargs
    )
