# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for Python tests that are run by pytest."""

load("@rules_python//python:defs.bzl", "py_test")

def iree_py_test(name, srcs, main = None, args = [], deps = [], **kwargs):
    """A py_test whose tests are collected and run by pytest.

    Args:
      name: Name of the test.
      srcs: Python sources: the file with the tests and what it needs from its
        own package.
      main: The file with the tests. Defaults to `name + ".py"`.
      args: Additional command line arguments for pytest.
      deps: Python libraries that the tests need. pytest is added.
      **kwargs: Passed on to py_test.
    """
    if main == None:
        main = name + ".py"
    runner = Label("//build_tools/bazel:pytest_main.py")

    # `__init__.py` files created in the runfiles would turn the directories of
    # the test into packages: pytest would import the test under a package
    # name, and a created `iree/__init__.py` would hide the other packages of
    # the `iree` namespace.
    legacy_create_init = kwargs.pop("legacy_create_init", False)

    py_test(
        name = name,
        srcs = srcs + [runner],
        args = ["$(rootpath {})".format(main)] + args,
        legacy_create_init = legacy_create_init,
        main = runner,
        deps = deps + [Label("@pip//pytest")],
        **kwargs
    )
