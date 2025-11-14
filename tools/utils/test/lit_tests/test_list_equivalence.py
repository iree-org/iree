# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Equivalence tests for shared listing output.

Ensures that `iree-lit-list` and `iree-lit-extract --list` produce identical
output for both text and JSON, preventing drift between tools.
"""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2]))

from lit_tools import iree_lit_extract, iree_lit_list

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestEquivalence(unittest.TestCase):
    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_text_equivalence(self, out1):
        # list tool output
        with patch("sys.argv", ["iree-lit-list", str(self.split_test)]):
            args_list = iree_lit_list.parse_arguments()

        rc = iree_lit_list.main(args_list)
        self.assertEqual(rc, 0)
        text_list = out1.getvalue()

        # extract --list output
        out2 = io.StringIO()
        with patch("sys.stdout", out2):
            with patch(
                "sys.argv", ["iree-lit-extract", str(self.split_test), "--list"]
            ):
                args_extract = iree_lit_extract.parse_arguments()

            rc2 = iree_lit_extract.main(args_extract)
            self.assertEqual(rc2, 0)
        text_extract = out2.getvalue()

        self.assertEqual(text_list.strip(), text_extract.strip())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_json_equivalence(self, out1):
        # list tool JSON
        with patch("sys.argv", ["iree-lit-list", str(self.split_test), "--json"]):
            args_list = iree_lit_list.parse_arguments()

        rc = iree_lit_list.main(args_list)
        self.assertEqual(rc, 0)
        payload_list = json.loads(out1.getvalue())

        # extract --list JSON
        out2 = io.StringIO()
        with patch("sys.stdout", out2):
            with patch(
                "sys.argv",
                ["iree-lit-extract", str(self.split_test), "--list", "--json"],
            ):
                args_extract = iree_lit_extract.parse_arguments()

            rc2 = iree_lit_extract.main(args_extract)
            self.assertEqual(rc2, 0)
        payload_extract = json.loads(out2.getvalue())

        self.assertEqual(payload_list, payload_extract)


if __name__ == "__main__":
    unittest.main()
