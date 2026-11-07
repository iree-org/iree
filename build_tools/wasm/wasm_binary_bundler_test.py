# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import unittest

from wasm_binary_bundler import parse_module_manifest


class ModuleManifestTest(unittest.TestCase):
    def test_json_preserves_paths_and_order(self):
        entries = [
            {"module": "first", "path": r"C:\source with spaces\first.mjs"},
            {"module": "second", "path": "/source/second.mjs"},
        ]
        self.assertEqual(parse_module_manifest(json.dumps(entries)), entries)

    def test_lines_preserve_paths_and_order(self):
        self.assertEqual(
            parse_module_manifest(
                "first:C:/source with spaces/first.mjs\n\nsecond:/source/second.mjs\n"
            ),
            [
                {"module": "first", "path": "C:/source with spaces/first.mjs"},
                {"module": "second", "path": "/source/second.mjs"},
            ],
        )

    def test_empty_manifests(self):
        for contents in ("", "\n", "  \n", "[]", "  []\n"):
            with self.subTest(contents=contents):
                self.assertEqual(parse_module_manifest(contents), [])

    def test_malformed_entries_fail(self):
        for contents in ("module", "module:", ":path", '[{"module":'):
            with self.subTest(contents=contents):
                with self.assertRaises(ValueError):
                    parse_module_manifest(contents)


if __name__ == "__main__":
    unittest.main()
