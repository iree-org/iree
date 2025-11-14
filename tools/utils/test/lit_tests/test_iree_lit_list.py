# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree_lit_list tool."""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2]))

from lit_tools import iree_lit_list

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestJSONOutput(unittest.TestCase):
    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_json_full_metadata(self, mock_stdout):
        with patch("sys.argv", ["iree-lit-list", str(self.split_test), "--json"]):
            args = iree_lit_list.parse_arguments()

        rc = iree_lit_list.main(args)
        self.assertEqual(rc, 0)
        data = json.loads(mock_stdout.getvalue())
        self.assertEqual(data["file"], str(self.split_test))
        self.assertEqual(data["count"], 3)
        self.assertEqual(len(data["cases"]), 3)
        self.assertEqual(data["cases"][0]["number"], 1)
        self.assertIn("start_line", data["cases"][0])
        self.assertIn("end_line", data["cases"][0])


class TestJSONExclusivity(unittest.TestCase):
    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_json_with_count_rejected(self, mock_stderr):
        with patch(
            "sys.argv", ["iree-lit-list", str(self.split_test), "--count", "--json"]
        ):
            args = iree_lit_list.parse_arguments()

        rc = iree_lit_list.main(args)
        self.assertEqual(rc, 2)
        self.assertIn("--json cannot be combined", mock_stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
