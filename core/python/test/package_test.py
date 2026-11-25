# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks that the iree.core package is set up properly."""

import importlib
import unittest


class PackageTest(unittest.TestCase):
    def test_import(self):
        importlib.import_module("iree.core")

    def test_iree_is_a_namespace_package(self):
        # iree.core, iree.compiler and iree.runtime share the `iree` namespace,
        # which only works while none of them provides an `iree/__init__.py`.
        iree = importlib.import_module("iree")
        self.assertIsNone(getattr(iree, "__file__", None))


if __name__ == "__main__":
    unittest.main()
