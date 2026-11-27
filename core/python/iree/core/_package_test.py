# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs on a pip installed core package and verifies it is set up properly."""

import importlib
import importlib.metadata
import pkgutil

import iree
import iree.core

# Verify version.
version = importlib.metadata.version("iree-base-core")
assert version
print("iree-base-core version:", version)

# The package shares the `iree` namespace with the compiler and runtime
# packages, which are wheels of their own. That only works as long as none of
# them installs an `iree/__init__.py`.
assert getattr(iree, "__file__", None) is None, "`iree` is not a namespace package"

# Import every module of the package. With only the wheel and what it declares
# as dependencies installed, this fails when the wheel misses a module or a
# dependency.
for module in pkgutil.walk_packages(iree.core.__path__, "iree.core."):
    if module.name == "iree.core._package_test":
        continue
    print("Importing", module.name)
    importlib.import_module(module.name)
