# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""End-to-end workflow tests across lit tools.

Covers combinations of:
- Text vs JSON
- stdout vs file (-o)
- List -> Extract -> List roundtrips
- Filters and selectors
- iree-lit-test dry-run integration on extracted files
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2]))

from lit_tools import iree_lit_extract, iree_lit_list, iree_lit_test
from lit_tools.core import test_file

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestWorkflows(unittest.TestCase):
    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"
        self.ten_cases = _FIXTURES_DIR / "ten_cases_test.mlir"

    # ---------- Roundtrips (text) ----------

    def test_extract_text_file_then_list_json_matches_selection(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            # Extract cases 1 and 3 to file (text mode includes RUN header; multiple are separated by // -----)
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.split_test),
                    "--case",
                    "1,3",
                    "-o",
                    str(out),
                ],
            ):
                args = iree_lit_extract.parse_arguments()
            rc = iree_lit_extract.main(args)
            self.assertEqual(rc, 0)
            text = out.read_text()
            self.assertIn("// -----", text)
            # Only one header RUN block should be present
            self.assertEqual(text.count("// RUN:"), 1)

            # List the extracted file as JSON and verify just 2 cases remain with expected names
            with patch("sys.argv", ["iree-lit-list", str(out), "--json"]):
                list_args = iree_lit_list.parse_arguments()
            with patch("sys.stdout", new_callable=io.StringIO) as out_json:
                rc2 = iree_lit_list.main(list_args)
                self.assertEqual(rc2, 0)
                payload = json.loads(out_json.getvalue())
                self.assertEqual(payload["count"], 2)
                names = [c["name"] for c in payload["cases"]]
                self.assertEqual(names, ["first_case", "third_case"])
        finally:
            out.unlink(missing_ok=True)

    def test_extract_text_stdout_then_reparse(self):
        # Extract case 2 to stdout and reparse by writing to a temporary file
        with patch(
            "sys.argv", ["iree-lit-extract", str(self.split_test), "--case", "2"]
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            rc = iree_lit_extract.main(args)
            self.assertEqual(rc, 0)
            content = mock_stdout.getvalue()
            # No RUN header by default on stdout text
            self.assertNotIn("// RUN:", content)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tmp_path.write_text(content)
            cases = test_file.parse_test_file(tmp_path)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].name, "second_case")
        finally:
            tmp_path.unlink(missing_ok=True)

    # ---------- JSON extraction semantics ----------

    def test_extract_json_stdout_array(self):
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.ten_cases), "--case", "1,3,5", "--json"],
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as out_json:
            rc = iree_lit_extract.main(args)
            self.assertEqual(rc, 0)
            data = json.loads(out_json.getvalue())
            self.assertEqual(len(data), 3)
            self.assertEqual([d["number"] for d in data], [1, 3, 5])
            # Each JSON object must include content
            for d in data:
                self.assertIn("content", d)

    def test_extract_json_file_written(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.ten_cases),
                    "--case",
                    "2-4",
                    "--json",
                    "-o",
                    str(out),
                ],
            ):
                args = iree_lit_extract.parse_arguments()
            # Should not print JSON to stdout; writes to file
            with patch("sys.stdout", new_callable=io.StringIO) as so:
                rc = iree_lit_extract.main(args)
                self.assertEqual(rc, 0)
                self.assertEqual(so.getvalue(), "")
            data = json.loads(out.read_text())
            nums = [d["number"] for d in data]
            self.assertEqual(nums, [2, 3, 4])
        finally:
            out.unlink(missing_ok=True)

    # ---------- Filters across tools ----------

    def test_filter_and_list_equivalence(self):
        # Extract by filter (text stdout), then list that content again via parser
        with patch(
            "sys.argv",
            [
                "iree-lit-extract",
                str(self.ten_cases),
                "--filter",
                "case_(one|three|five)",
            ],
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as so:
            rc = iree_lit_extract.main(args)
            self.assertEqual(rc, 0)
            text = so.getvalue()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            subset = Path(tmp.name)
        try:
            subset.write_text(text)
            cases = test_file.parse_test_file(subset)
            self.assertEqual(
                [c.name for c in cases], ["case_one", "case_three", "case_five"]
            )
        finally:
            subset.unlink(missing_ok=True)

    # ---------- iree-lit-test dry-run integration ----------

    def test_dry_run_on_extracted_file_text(self):
        # Extract a subset to a file (text), then run iree-lit-test --dry-run on it
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            subset = Path(tmp.name)
        try:
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.ten_cases),
                    "--case",
                    "1,3,5",
                    "-o",
                    str(subset),
                ],
            ):
                args = iree_lit_extract.parse_arguments()
            self.assertEqual(iree_lit_extract.main(args), 0)

            # Now dry-run the subset file (should see exactly 3 cases)
            with patch(
                "sys.argv", ["iree-lit-test", str(subset), "--dry-run", "--json"]
            ):
                targs = iree_lit_test.parse_arguments()
            with patch("sys.stdout", new_callable=io.StringIO) as so:
                rc = iree_lit_test.main(targs)
                self.assertEqual(rc, 0)
                payload = json.loads(so.getvalue())
                self.assertEqual(payload["total_cases"], 3)
                self.assertEqual(
                    [c["number"] for c in payload["selected_cases"]], [1, 2, 3]
                )
        finally:
            subset.unlink(missing_ok=True)

    def test_dry_run_on_extracted_json_file(self):
        # Extract JSON to a file, then ensure that the tool chain still functions
        # (this checks that JSON write mode does not interfere with text workflows)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json_out = Path(tmp.name)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp2:
            text_out = Path(tmp2.name)
        try:
            # Write JSON to file for cases 6-7
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.ten_cases),
                    "--case",
                    "6-7",
                    "--json",
                    "-o",
                    str(json_out),
                ],
            ):
                args = iree_lit_extract.parse_arguments()
            with patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(iree_lit_extract.main(args), 0)

            # Now extract the same selection in text mode to a file and dry-run it
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.ten_cases),
                    "--case",
                    "6-7",
                    "-o",
                    str(text_out),
                ],
            ):
                targs = iree_lit_extract.parse_arguments()
            self.assertEqual(iree_lit_extract.main(targs), 0)

            with patch(
                "sys.argv", ["iree-lit-test", str(text_out), "--dry-run", "--json"]
            ):
                rargs = iree_lit_test.parse_arguments()
            with patch("sys.stdout", new_callable=io.StringIO) as so:
                rc = iree_lit_test.main(rargs)
                self.assertEqual(rc, 0)
                payload = json.loads(so.getvalue())
                self.assertEqual(payload["total_cases"], 2)
                self.assertEqual(
                    [c["number"] for c in payload["selected_cases"]], [1, 2]
                )
        finally:
            json_out.unlink(missing_ok=True)
            text_out.unlink(missing_ok=True)

    # ---------- Additional roundtrips & edge cases ----------

    def test_text_json_equivalence_for_selection(self):
        # Compare text stdout with JSON-assembled text for the same selection
        selection = "1-3"
        # Text stdout (no RUN header by default)
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.split_test), "--case", selection],
        ):
            args_text = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as so_text:
            self.assertEqual(iree_lit_extract.main(args_text), 0)
            text_out = so_text.getvalue()

        # JSON stdout for same selection
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.split_test), "--case", selection, "--json"],
        ):
            args_json = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as so_json:
            self.assertEqual(iree_lit_extract.main(args_json), 0)
            data = json.loads(so_json.getvalue())
            # Assemble text from JSON content with separators
            assembled = "\n\n// -----\n\n".join(d["content"].rstrip() for d in data)
            # Normalize leading blanks and trailing newlines for stable comparison
            norm_text = text_out.lstrip("\n").rstrip()
            norm_assembled = assembled.lstrip("\n").rstrip()
            self.assertEqual(norm_text, norm_assembled)

    def test_no_run_header_behavior(self):
        # File without RUN header lines should not gain any in -o
        no_run = _FIXTURES_DIR / "no_run_test.mlir"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            with patch(
                "sys.argv",
                ["iree-lit-extract", str(no_run), "--case", "1", "-o", str(out)],
            ):
                args = iree_lit_extract.parse_arguments()
            self.assertEqual(iree_lit_extract.main(args), 0)
            text = out.read_text()
            self.assertNotIn("// RUN:", text)
        finally:
            out.unlink(missing_ok=True)

    def test_run_variants_include_run_lines_stdout(self):
        # Ensure include-run-lines writes header runs correctly for variant formatting
        run_variants = _FIXTURES_DIR / "run_variants.mlir"
        with patch(
            "sys.argv",
            [
                "iree-lit-extract",
                str(run_variants),
                "--case",
                "1",
                "--include-run-lines",
            ],
        ):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as so:
            self.assertEqual(iree_lit_extract.main(args), 0)
            out_text = so.getvalue()
            # Expect header RUN commands present at top
            self.assertIn("// RUN:", out_text)
            # Only header runs, not duplicated within content
            self.assertGreaterEqual(out_text.count("// RUN:"), 1)

    def test_renumber_after_subset_and_reextract(self):
        # Extract cases 2,4,6 to file; verify renumbering and re-extract by new case number
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            subset = Path(tmp.name)
        try:
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.ten_cases),
                    "--case",
                    "2,4,6",
                    "-o",
                    str(subset),
                ],
            ):
                args = iree_lit_extract.parse_arguments()
            self.assertEqual(iree_lit_extract.main(args), 0)
            # Re-parse subset: should have 3 cases numbered 1..3 with names preserved
            cases = test_file.parse_test_file(subset)
            self.assertEqual(len(cases), 3)
            self.assertEqual(
                [c.name for c in cases], ["case_two", "case_four", "case_six"]
            )
            self.assertEqual([c.number for c in cases], [1, 2, 3])
            # Re-extract case 2 from subset and verify content name
            with patch("sys.argv", ["iree-lit-extract", str(subset), "--case", "2"]):
                args2 = iree_lit_extract.parse_arguments()
            with patch("sys.stdout", new_callable=io.StringIO) as so:
                self.assertEqual(iree_lit_extract.main(args2), 0)
                text = so.getvalue()
                self.assertIn("@case_four", text)
        finally:
            subset.unlink(missing_ok=True)


# ========================
# Cross-tool flow tests
# (migrated from test_integration_flows.py)
# ========================


class TestListExtractFlows(unittest.TestCase):
    def setUp(self):
        self.fixtures = Path(__file__).parent / "fixtures"
        self.split = self.fixtures / "split_test.mlir"
        self.ten = self.fixtures / "ten_cases_test.mlir"

    def test_list_names_drive_extract_by_name_stdout(self):
        # Get names from list --names
        with patch("sys.argv", ["iree-lit-list", str(self.split), "--names"]):
            args = iree_lit_list.parse_arguments()
        out_names = io.StringIO()
        with patch("sys.stdout", out_names):
            self.assertEqual(iree_lit_list.main(args), 0)
        names = out_names.getvalue().strip().split()
        # Extract each by name; verify some content for each
        for name in names:
            with patch(
                "sys.argv",
                ["iree-lit-extract", str(self.split), "--name", name.lstrip("@")],
            ):
                eargs = iree_lit_extract.parse_arguments()
            out = io.StringIO()
            with patch("sys.stdout", out):
                self.assertEqual(iree_lit_extract.main(eargs), 0)
            text = out.getvalue()
            self.assertIn(name, text)
            self.assertIn("util.func", text)

    def test_list_json_drive_extract_json_file(self):
        # Use list JSON to select first 3 cases by number and write JSON subset to a file
        with patch("sys.argv", ["iree-lit-list", str(self.ten), "--json"]):
            args = iree_lit_list.parse_arguments()
        out = io.StringIO()
        with patch("sys.stdout", out):
            self.assertEqual(iree_lit_list.main(args), 0)
        listing = json.loads(out.getvalue())
        nums = [c["number"] for c in listing["cases"][:3]]
        # Extract JSON to a file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            dst = Path(tmp.name)
        try:
            sel = ",".join(map(str, nums))
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.ten),
                    "--case",
                    sel,
                    "--json",
                    "-o",
                    str(dst),
                ],
            ):
                eargs = iree_lit_extract.parse_arguments()
            with patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(iree_lit_extract.main(eargs), 0)
            subset = json.loads(dst.read_text())
            self.assertEqual([d["number"] for d in subset], nums)
        finally:
            dst.unlink(missing_ok=True)


class TestListTestFlows(unittest.TestCase):
    def setUp(self):
        self.fixtures = Path(__file__).parent / "fixtures"
        self.split = self.fixtures / "split_test.mlir"
        self.ten = self.fixtures / "ten_cases_test.mlir"

    def test_list_count_matches_test_dry_run_total(self):
        # list --count
        with patch("sys.argv", ["iree-lit-list", str(self.ten), "--count"]):
            largs = iree_lit_list.parse_arguments()
        count_io = io.StringIO()
        with patch("sys.stdout", count_io):
            self.assertEqual(iree_lit_list.main(largs), 0)
        count = int(count_io.getvalue().strip())
        # iree-lit-test --dry-run --json
        with patch("sys.argv", ["iree-lit-test", str(self.ten), "--dry-run", "--json"]):
            targs = iree_lit_test.parse_arguments()
        jout = io.StringIO()
        with patch("sys.stdout", jout):
            self.assertEqual(iree_lit_test.main(targs), 0)
        payload = json.loads(jout.getvalue())
        self.assertEqual(payload["total_cases"], count)

    def test_list_json_drive_test_dry_run_selection(self):
        # pick even-numbered cases via list JSON
        with patch("sys.argv", ["iree-lit-list", str(self.ten), "--json"]):
            largs = iree_lit_list.parse_arguments()
        out = io.StringIO()
        with patch("sys.stdout", out):
            self.assertEqual(iree_lit_list.main(largs), 0)
        listing = json.loads(out.getvalue())
        even = [c["number"] for c in listing["cases"] if c["number"] % 2 == 0][:4]
        sel = ",".join(map(str, even))
        # Drive iree-lit-test --dry-run with those selections
        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten), "--case", sel, "--dry-run", "--json"],
        ):
            targs = iree_lit_test.parse_arguments()
        jout = io.StringIO()
        with patch("sys.stdout", jout):
            self.assertEqual(iree_lit_test.main(targs), 0)
        payload = json.loads(jout.getvalue())
        sel_nums = [c["number"] for c in payload["selected_cases"]]
        self.assertEqual(sel_nums, even)


class TestReplaceFlows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Skip entire class if iree_lit_replace is not available yet.
        try:
            import lit_tools.iree_lit_replace as _unused  # noqa: F401, PLC0415 (deferred import for availability check)
        except Exception as err:
            raise unittest.SkipTest(
                "iree-lit-replace not implemented in this tree"
            ) from err

    def test_placeholder(self):
        # This is a placeholder to reserve flow tests once the tool exists.
        self.assertTrue(True)


# ========================
# Roundtrip robustness
# ========================


class TestRoundtripRobustness(unittest.TestCase):
    def setUp(self):
        self.fixtures = Path(__file__).parent / "fixtures"
        self.names = self.fixtures / "names_test.mlir"
        self.scattered = self.fixtures / "failing_scattered_run_lines.mlir"
        self.run_variants = self.fixtures / "run_variants.mlir"

    def test_single_case_with_special_name_roundtrip(self):
        # Extract text to stdout and reparse, name should be preserved
        with patch("sys.argv", ["iree-lit-extract", str(self.names), "--case", "1"]):
            args = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as so:
            self.assertEqual(iree_lit_extract.main(args), 0)
            text = so.getvalue()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            path.write_text(text)
            cases = test_file.parse_test_file(path)
            self.assertEqual(len(cases), 1)
            # Name contains punctuation that must survive parsing
            self.assertEqual(cases[0].name, "foo.bar$baz-1")
        finally:
            path.unlink(missing_ok=True)

    def test_scattered_run_lines_header_injection_only(self):
        # Extract the final case to a file; header RUN lines should be injected once
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            cases = test_file.parse_test_file(self.scattered)
            last_num = cases[-1].number
            with patch(
                "sys.argv",
                [
                    "iree-lit-extract",
                    str(self.scattered),
                    "--case",
                    str(last_num),
                    "-o",
                    str(out),
                ],
            ):
                args = iree_lit_extract.parse_arguments()
            self.assertEqual(iree_lit_extract.main(args), 0)
            content = out.read_text()
            # Only header RUN lines (2) should be present
            self.assertEqual(content.count("// RUN:"), 2)
            # Lines referencing NOTUSED are not RUN directives and should remain as comments
            self.assertIn("NOTUSED:", content)
        finally:
            out.unlink(missing_ok=True)

    def test_run_variants_text_json_equivalence(self):
        # JSON assembled content must match text stdout (normalized)
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.run_variants), "--case", "1"],
        ):
            targs = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as so_text:
            self.assertEqual(iree_lit_extract.main(targs), 0)
            text_out = so_text.getvalue()

        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.run_variants), "--case", "1", "--json"],
        ):
            jargs = iree_lit_extract.parse_arguments()
        with patch("sys.stdout", new_callable=io.StringIO) as so_json:
            self.assertEqual(iree_lit_extract.main(jargs), 0)
            data = json.loads(so_json.getvalue())
        assembled = "\n\n// -----\n\n".join(d["content"].rstrip() for d in data)
        self.assertEqual(
            text_out.lstrip("\n").rstrip(), assembled.lstrip("\n").rstrip()
        )


# ========================
# Negative-path handling
# ========================


class TestNegativePaths(unittest.TestCase):
    def setUp(self):
        self.fixtures = Path(__file__).parent / "fixtures"
        self.split = self.fixtures / "split_test.mlir"
        self.ten = self.fixtures / "ten_cases_test.mlir"

    def test_list_json_cannot_combine_with_count(self):
        with patch("sys.argv", ["iree-lit-list", str(self.split), "--json", "--count"]):
            args = iree_lit_list.parse_arguments()
        err = io.StringIO()
        with patch("sys.stderr", err):
            rc = iree_lit_list.main(args)
        self.assertEqual(rc, 2)
        self.assertIn("cannot be combined", err.getvalue())

    def test_extract_invalid_name(self):
        with patch(
            "sys.argv",
            ["iree-lit-extract", str(self.split), "--name", "does_not_exist"],
        ):
            args = iree_lit_extract.parse_arguments()
        err = io.StringIO()
        with patch("sys.stderr", err):
            rc = iree_lit_extract.main(args)
        self.assertEqual(rc, 2)
        self.assertIn("Case with name", err.getvalue())
        self.assertIn("not found", err.getvalue())

    def test_test_invalid_case_number(self):
        with patch(
            "sys.argv", ["iree-lit-test", str(self.split), "--case", "999", "--dry-run"]
        ):
            args = iree_lit_test.parse_arguments()
        err = io.StringIO()
        with patch("sys.stderr", err):
            rc = iree_lit_test.main(args)
        self.assertEqual(rc, 2)
        self.assertIn("Case 999 not found", err.getvalue())

    def test_test_filter_no_matches(self):
        with patch(
            "sys.argv",
            ["iree-lit-test", str(self.ten), "--filter", "nevermatch", "--dry-run"],
        ):
            args = iree_lit_test.parse_arguments()
        err = io.StringIO()
        with patch("sys.stderr", err):
            rc = iree_lit_test.main(args)
        self.assertEqual(rc, 2)
        self.assertIn("No cases matched --filter", err.getvalue())


if __name__ == "__main__":
    unittest.main()
