#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for gen_plugin_abi_hash."""

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import gen_plugin_abi_hash


class ComputeHashTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)

    def _write(self, name: str, text: str) -> str:
        path = self.root / name
        path.write_text(text)
        return str(path)

    def test_content_change_changes_the_hash(self):
        before = self._write("Client.h", "class A { virtual void f(); };\n")
        first = gen_plugin_abi_hash.compute_hash([before])
        self._write("Client.h", "class A { virtual void f(); virtual void g(); };\n")
        self.assertNotEqual(first, gen_plugin_abi_hash.compute_hash([before]))

    def test_argument_order_does_not_matter(self):
        a = self._write("Client.h", "a\n")
        b = self._write("PluginEntryPoint.h", "b\n")
        self.assertEqual(
            gen_plugin_abi_hash.compute_hash([a, b]),
            gen_plugin_abi_hash.compute_hash([b, a]),
        )

    def test_moving_content_between_headers_changes_the_hash(self):
        # The name is hashed with the content, so a declaration that migrates
        # to another header does not slip through.
        a = self._write("Client.h", "struct S { int x; };\n")
        b = self._write("PluginEntryPoint.h", "")
        first = gen_plugin_abi_hash.compute_hash([a, b])
        self._write("Client.h", "")
        self._write("PluginEntryPoint.h", "struct S { int x; };\n")
        self.assertNotEqual(first, gen_plugin_abi_hash.compute_hash([a, b]))

    def test_hash_is_a_short_hex_string(self):
        a = self._write("Client.h", "a\n")
        digest = gen_plugin_abi_hash.compute_hash([a])
        self.assertEqual(len(digest), 16)
        self.assertRegex(digest, r"^[0-9a-f]{16}$")


if __name__ == "__main__":
    unittest.main()
