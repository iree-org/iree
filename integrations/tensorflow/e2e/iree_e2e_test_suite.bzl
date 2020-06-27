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

"""Macro for building a test suite to run the e2e tests on all backends."""

load("//bindings/python:build_defs.oss.bzl", "iree_py_test")

def iree_e2e_test_suite(
        name,
        backends_to_srcs,
        reference_backend,
        deps = None,
        tags = None,
        python_version = "PY3",
        **kwargs):
    """Creates a iree_py_test for all of the given backends and srcs, and a test suite that bundles them.

    Args:
      name:
        name of the generated test suite.
      backends_to_srcs:
        a dictionary mapping backends to a list of test files to run on them.
      reference_backend:
        the backend to use as a source of truth for the expected output results.
      deps:
        test dependencies.
      tags:
        tags to apply to the test. Note that as in standard test suites, manual
        is treated specially and will also apply to the test suite itself.
      python_version:
        the python version to run the tests with. Uses python3 by default.
      **kwargs:
        Any additional arguments that will be passed to the underlying tests and
        test_suite.
    """
    tests = []

    for backend, srcs in backends_to_srcs.items():
        for src in srcs:
            test_name = "{}_{}__{}__{}".format(
                name,
                src[:-3],
                reference_backend,
                backend,
            )
            args = [
                "--target_backends={},{}".format(reference_backend, backend),
            ]

            # TODO(GH-2175): Simplify this after backend names are standardized.
            driver = backend.replace("iree_", "")  # "iree_<driver>" --> "<driver>"
            if driver == "llvmjit":
                driver = "llvm"
            py_test_tags = ["driver={}".format(driver)]
            if tags != None:  # `is` is not supported.
                py_test_tags += tags

            iree_py_test(
                name = test_name,
                main = src,
                srcs = [src],
                deps = deps,
                args = args,
                tags = py_test_tags,
                python_version = python_version,
                **kwargs
            )
            tests.append(test_name)

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
