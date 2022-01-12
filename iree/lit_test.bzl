# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    data = data if data else []

    # Always include llvm-sybmolizer so we get useful stack traces. Maybe it
    # would be better to force everyone to do this explicitly, but since
    # forgetting wouldn't cause the test to fail, only make debugging harder
    # when it does, I think better to hardcode it here.
    llvm_symbolizer = "@llvm-project//llvm:llvm-symbolizer"
    if llvm_symbolizer not in data:
        data.append(llvm_symbolizer)

    # First argument is the test. The rest are tools to add to the path.
    data = [test_file] + data

    native.sh_test(
        name = name,
        srcs = [driver],
        size = size,
        data = data,
        env = {"FILECHECK_OPTS": "--enable-var-scope"},
        args = ["$(location {})".format(file) for file in data],
        **kwargs
    )

def iree_lit_test_suite(
        name,
        srcs,
        data,
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
      **kwargs: Any additional arguments that will be passed to the underlying tests and test_suite.
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
        # If there are kwargs that need to be passed here which only apply to
        # the generated tests and not to test_suite, they should be extracted
        # into separate named arguments.
        **kwargs
    )
