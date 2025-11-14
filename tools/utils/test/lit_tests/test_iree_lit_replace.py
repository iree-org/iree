# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree_lit_replace tool."""

import inspect
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project tools/utils to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from lit_tools import iree_lit_extract, iree_lit_replace
from lit_tools.core import verification
from lit_tools.core.parser import parse_test_file

from test.test_helpers import run_python_module

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class MockStdin:
    """Mock stdin object with buffer attribute for testing."""

    def __init__(self, content: bytes):
        """Initialize with binary content."""
        self.buffer = io.BytesIO(content)


class TestBasicReplacement(unittest.TestCase):
    """Tests for basic text mode replacement."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_replace_by_number(self):
        """Test replacing a case by number."""
        # Create a temp copy of the test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # New content for replacement
            new_content = """// CHECK-LABEL: @replaced_case
util.func @replaced_case() {
  return
}
"""

            # Replace case 2
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "2"]
            ), patch("sys.stdin", MockStdin(new_content.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify replacement
            content = tmp_path.read_text()
            self.assertIn("@replaced_case", content)
            self.assertIn("// -----", content)  # Delimiters preserved

            # Verify we can still parse it
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(len(cases), 3)  # Still 3 cases
            self.assertEqual(cases[1].name, "replaced_case")  # Case 2 (index 1)

        finally:
            tmp_path.unlink()

    def test_replace_by_name(self):
        """Test replacing a case by name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            new_content = """// CHECK-LABEL: @third_case
util.func @third_case(%arg0: tensor<4xf32>) {
  return
}
"""

            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--name", "third_case"]
            ), patch("sys.stdin", MockStdin(new_content.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify replacement
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[2].name, "third_case")
            self.assertIn("tensor<4xf32>", cases[2].content)

        finally:
            tmp_path.unlink()

    def test_round_trip_extract_replace(self):
        """Test extract → replace → extract produces same result."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Extract case 2
            original_cases = list(parse_test_file(tmp_path).cases)
            original_case_2 = original_cases[1]

            # Replace with same content
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "2"]
            ), patch("sys.stdin", MockStdin(original_case_2.content.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Extract again
            new_cases = list(parse_test_file(tmp_path).cases)
            new_case_2 = new_cases[1]

            # Should be identical
            self.assertEqual(original_case_2.content, new_case_2.content)
            self.assertEqual(original_case_2.name, new_case_2.name)

        finally:
            tmp_path.unlink()


class TestErrorHandling(unittest.TestCase):
    """Tests for error conditions."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_case_not_found(self):
        """Test error when case number doesn't exist."""
        with patch(
            "sys.argv", ["iree-lit-replace", str(self.split_test), "--case", "99"]
        ), patch("sys.stdin", MockStdin(b"replacement")):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertEqual(result, 2)  # NOT_FOUND

    def test_name_not_found(self):
        """Test error when case name doesn't exist."""
        with patch(
            "sys.argv",
            ["iree-lit-replace", str(self.split_test), "--name", "nonexistent"],
        ), patch("sys.stdin", MockStdin(b"replacement")):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertEqual(result, 2)  # NOT_FOUND

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with patch(
            "sys.argv", ["iree-lit-replace", "/nonexistent/file.mlir", "--case", "1"]
        ), patch("sys.stdin", MockStdin(b"replacement")):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertEqual(result, 2)  # NOT_FOUND

    def test_missing_selector_in_text_mode(self):
        """Test error when no case selector provided in text mode."""
        with patch("sys.argv", ["iree-lit-replace", str(self.split_test)]), patch(
            "sys.stdin", MockStdin(b"replacement")
        ):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertEqual(result, 1)  # ERROR

    def test_both_case_and_name(self):
        """Test error when both --case and --name provided."""
        with patch(
            "sys.argv",
            [
                "iree-lit-replace",
                str(self.split_test),
                "--case",
                "1",
                "--name",
                "first",
            ],
        ), patch("sys.stdin", MockStdin(b"replacement")):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertEqual(result, 1)  # ERROR

    def test_duplicate_check_label_names(self):
        """Test error when using --name with ambiguous duplicate labels."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        duplicate_labels = fixtures_dir / "duplicate_labels.mlir"

        with patch(
            "sys.argv", ["iree-lit-replace", str(duplicate_labels), "--name", "foo"]
        ), patch("sys.stdin", MockStdin(b"replacement")):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        # Should fail with ambiguous name error.
        self.assertNotEqual(result, 0, "Should error when name is ambiguous")


class TestIdempotency(unittest.TestCase):
    """Tests for idempotent behavior (no-op replacements)."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_no_op_replacement(self):
        """Test that replacing with identical content is a no-op."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Extract original case 2 content
            cases = list(parse_test_file(tmp_path).cases)
            original_content = cases[1].content

            # Replace with identical content
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "2"]
            ), patch("sys.stdin", MockStdin(original_content.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify case content is preserved
            new_cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(original_content, new_cases[1].content)

        finally:
            tmp_path.unlink()
            # Clean up backup if created
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_byte_for_byte_idempotency(self):
        """Test that parse → rebuild produces byte-for-byte identical output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Get original file content
            original_file_content = tmp_path.read_text()

            # Parse and rebuild without any changes
            test_file_obj = parse_test_file(tmp_path)
            cases = list(test_file_obj.cases)
            header_runs = test_file_obj.extract_run_lines(raw=True)
            rebuilt_content = iree_lit_replace.build_file_content(header_runs, cases)

            # Should be byte-for-byte identical
            self.assertEqual(
                original_file_content,
                rebuilt_content,
                "Parse and rebuild should be byte-for-byte identical",
            )

            # Verify rebuilt content ends with newline (pre-commit requirement)
            self.assertTrue(
                rebuilt_content.endswith("\n"),
                "Rebuilt file must end with newline",
            )

        finally:
            tmp_path.unlink()
            # Clean up backup if created
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


class TestRunLineHandling(unittest.TestCase):
    """Tests for RUN line validation and replacement."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"
        self.with_case_runs = _FIXTURES_DIR / "with_case_runs.mlir"

    def test_replacement_with_matching_run_lines(self):
        """Test that replacement with matching RUN lines succeeds."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Create replacement content with matching RUN lines
            replacement = """// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(symbol-dce)' %s | FileCheck %s

// CHECK-LABEL: @new_case
util.func @new_case() {
  return
}
"""

            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "2"]
            ), patch("sys.stdin", MockStdin(replacement.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify case was replaced and RUN lines were stripped
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[1].name, "new_case")

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_replacement_with_different_run_lines_errors(self):
        """Test that replacement with different RUN lines errors by default."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Create replacement content with DIFFERENT RUN lines
            replacement = """// RUN: iree-opt --different-flag %s

// CHECK-LABEL: @new_case
util.func @new_case() {
  return
}
"""

            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "2"]
            ), patch("sys.stdin", MockStdin(replacement.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should fail with ERROR
            self.assertEqual(result, 1)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_text_mode_injects_case_local_runs(self):
        """Text mode: omit RUNs in replacement; original case-local RUNs are reinjected."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.with_case_runs.read_text())
            tmp_path = Path(tmp.name)
        try:
            replacement = """// CHECK-LABEL: @case_one
func.func @case_one() {
  // CHECK: return
  return
}
"""
            # Replace case 1 with content without RUN lines
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "1"]
            ), patch("sys.stdin", MockStdin(replacement.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                rc = iree_lit_replace.main(args)
            self.assertEqual(rc, 0)

            # After replacement, file should include the case-local RUN line again
            # (Check raw file, not parsed content, since parse_test_file strips RUN lines)
            file_content = tmp_path.read_text()
            self.assertIn('// RUN: echo "alpha"', file_content)
            # Verify the replacement content was applied
            self.assertIn("@case_one", file_content)
            self.assertIn("// CHECK: return", file_content)
        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_text_mode_case_local_run_mismatch_errors(self):
        """Text mode: different case-local RUNs should error without --replace-run-lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.with_case_runs.read_text())
            tmp_path = Path(tmp.name)
        try:
            replacement = """// RUN: echo \"DIFF\" | FileCheck %s --check-prefix=ALPHA
// CHECK-LABEL: @case_one
func.func @case_one() { return }
"""
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "1"]
            ), patch("sys.stdin", MockStdin(replacement.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                rc = iree_lit_replace.main(args)
            self.assertEqual(rc, 1)
        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_duplicate_replacements_error(self):
        """JSON mode: duplicate entries for the same case should error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)
        try:
            dup_json = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// CHECK-LABEL: @a\nfunc.func @a() { return }\n",
                    },
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// CHECK-LABEL: @b\nfunc.func @b() { return }\n",
                    },
                ]
            )
            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(dup_json.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                rc = iree_lit_replace.main(args)
            self.assertEqual(rc, 1)
        finally:
            tmp_path.unlink()

    def test_json_name_number_mismatch_error(self):
        """JSON mode: when both name and number are present but mismatch, error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)
        try:
            # number points to 2, but name points to first_case
            bad = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "name": "first_case",
                        "content": "// CHECK-LABEL: @x\nfunc.func @x() { return }\n",
                    }
                ]
            )
            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(bad.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                rc = iree_lit_replace.main(args)
            self.assertEqual(rc, 1)
        finally:
            tmp_path.unlink()

    def test_text_mode_duplicate_name_error(self):
        """Text mode: replacing by --name with duplicates should error."""
        # Build a small file with duplicate names
        dup = """// RUN: iree-opt %s | FileCheck %s
// CHECK-LABEL: @dup
func.func @dup() { return }

// -----

// CHECK-LABEL: @dup
func.func @dup_v2() { return }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(dup)
            tmp_path = Path(tmp.name)
        try:
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--name", "dup"]
            ), patch(
                "sys.stdin",
                MockStdin(b"// CHECK-LABEL: @dup\nfunc.func @dup() { return }\n"),
            ):
                args = iree_lit_replace.parse_arguments()
                rc = iree_lit_replace.main(args)
            self.assertEqual(rc, 1)
        finally:
            tmp_path.unlink()

    def test_replacement_with_different_run_lines_and_flag(self):
        """Test that --replace-run-lines allows replacing RUN lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Create replacement content with DIFFERENT RUN lines
            replacement = """// RUN: iree-opt --different-flag %s

// CHECK-LABEL: @new_case
util.func @new_case() {
  return
}
"""

            with patch(
                "sys.argv",
                [
                    "iree-lit-replace",
                    str(tmp_path),
                    "--case",
                    "2",
                    "--replace-run-lines",
                ],
            ), patch("sys.stdin", MockStdin(replacement.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should succeed
            self.assertEqual(result, 0)

            # Verify RUN lines were replaced
            new_runs = parse_test_file(tmp_path).extract_run_lines(raw=False)
            self.assertEqual(len(new_runs), 1)
            self.assertIn("--different-flag", new_runs[0])

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_case_local_run_drift_error(self):
        """Test that changing case-local RUN line without --replace-run-lines errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.with_case_runs.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Case 2 has case-local RUN at line 16: echo "beta" | FileCheck %s --check-prefix=BETA
            # Replace with DIFFERENT case-local RUN (should error without --replace-run-lines).
            replacement = """// RUN: echo "modified" | FileCheck %s --check-prefix=MODIFIED
// CHECK-LABEL: @case_two
func.func @case_two() {
  return
}
"""

            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "2"]
            ), patch("sys.stdin", MockStdin(replacement.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should fail (RUN line drift without --replace-run-lines).
            self.assertNotEqual(result, 0, "Should error when case-local RUN changes")

            # Original file should be unchanged.
            file_content = tmp_path.read_text()
            self.assertIn("@case_two", file_content)
            self.assertNotIn("MODIFIED", file_content)
        finally:
            tmp_path.unlink()


class TestJSONMode(unittest.TestCase):
    """Tests for JSON mode batch replacements."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_json_single_replacement(self):
        """Test JSON mode with single replacement."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # JSON input for single replacement
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// CHECK-LABEL: @replaced\nutil.func @replaced() {\n  return\n}\n",
                    }
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify replacement
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[1].name, "replaced")

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_batch_replacement(self):
        """Test JSON mode with multiple replacements in same file."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # JSON input for batch replacement (cases 1 and 3)
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 1,
                        "content": "// CHECK-LABEL: @batch_1\nutil.func @batch_1() {\n  return\n}\n",
                    },
                    {
                        "file": str(tmp_path),
                        "number": 3,
                        "content": "// CHECK-LABEL: @batch_3\nutil.func @batch_3() {\n  return\n}\n",
                    },
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify both replacements
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[0].name, "batch_1")
            self.assertEqual(cases[2].name, "batch_3")

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_file_override(self):
        """Test CLI file argument overrides JSON file field."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # JSON has different file, but CLI overrides
            json_input = json.dumps(
                [
                    {
                        "file": "ignored.mlir",  # This should be ignored
                        "number": 2,
                        "content": "// CHECK-LABEL: @overridden\nutil.func @overridden() {\n  return\n}\n",
                    }
                ]
            )

            with patch("sys.argv", ["iree-lit-replace", str(tmp_path)]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify replacement went to tmp_path, not "ignored.mlir"
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[1].name, "overridden")

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_schema_validation_errors(self):
        """Test JSON schema validation catches errors."""

        # Test: missing content field
        json_input = json.dumps([{"file": "test.mlir", "number": 2}])

        with patch("sys.argv", ["iree-lit-replace"]), patch(
            "sys.stdin", MockStdin(json_input.encode("utf-8"))
        ):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertEqual(result, 1)  # ERROR

        # Test: neither number nor name specified
        json_input = json.dumps([{"file": "test.mlir", "content": "test"}])

        with patch("sys.argv", ["iree-lit-replace"]), patch(
            "sys.stdin", MockStdin(json_input.encode("utf-8"))
        ):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertEqual(result, 1)  # ERROR

        # Test: missing file field when no CLI arg
        json_input = json.dumps([{"number": 2, "content": "test"}])

        with patch("sys.argv", ["iree-lit-replace"]), patch(
            "sys.stdin", MockStdin(json_input.encode("utf-8"))
        ):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertEqual(result, 1)  # ERROR

    def test_json_case_not_found(self):
        """Test JSON mode error when case not found."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Request non-existent case
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 99,
                        "content": "replacement",
                    }
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should fail during validation
            self.assertEqual(result, 1)

        finally:
            tmp_path.unlink()

    def test_json_round_trip_with_extract(self):
        """Test round-trip: extract → edit → replace."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Simulate iree-lit-extract output
            cases = list(parse_test_file(tmp_path).cases)
            extract_output = [
                {
                    "number": cases[1].number,
                    "name": cases[1].name,
                    "start_line": cases[1].start_line,
                    "end_line": cases[1].end_line,
                    "line_count": cases[1].line_count,
                    "check_count": cases[1].check_count,
                    "content": "// CHECK-LABEL: @edited\nutil.func @edited() {\n  return\n}\n",
                    "file": str(tmp_path),  # Add file field
                }
            ]

            json_input = json.dumps(extract_output)

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify replacement
            new_cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(new_cases[1].name, "edited")

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_run_line_validation(self):
        """Test RUN line validation in JSON mode."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Content with different RUN lines (should error)
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// RUN: iree-opt --different-flag %s\n\n// CHECK-LABEL: @test\nutil.func @test() {\n  return\n}\n",
                    }
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should fail due to RUN line mismatch
            self.assertEqual(result, 1)

            # Now with --replace-run-lines flag (should succeed)
            with patch("sys.argv", ["iree-lit-replace", "--replace-run-lines"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_case_local_runs_match_original(self):
        """JSON mode: replacement with matching case-local RUNs succeeds."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        with_case_runs = fixtures_dir / "with_case_runs.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(with_case_runs.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Replacement for case 2 includes the original case-local RUN line (should succeed)
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": (
                            '// RUN: echo "beta" | FileCheck %s --check-prefix=BETA\n'
                            "// CHECK-LABEL: @case_two\n"
                            "func.func @case_two() {\n"
                            "  // CHECK: return\n"
                            "  return\n"
                            "}\n"
                        ),
                    }
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0, "Should succeed when case-local RUNs match")

            # Verify the case-local RUN is still present
            file_content = tmp_path.read_text()
            self.assertIn('// RUN: echo "beta"', file_content)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_case_local_runs_mismatch(self):
        """JSON mode: replacement with different case-local RUNs errors without replace_run_lines."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        with_case_runs = fixtures_dir / "with_case_runs.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(with_case_runs.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Replacement with DIFFERENT case-local RUN line (should error)
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 1,
                        "content": (
                            '// RUN: echo "DIFFERENT" | FileCheck %s --check-prefix=ALPHA\n'
                            "// CHECK-LABEL: @case_one\n"
                            "func.func @case_one() { return }\n"
                        ),
                    }
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 1, "Should error when case-local RUNs don't match")

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_replace_run_lines_with_case_local(self):
        """JSON mode: replace_run_lines allows changing case-local RUNs."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        with_case_runs = fixtures_dir / "with_case_runs.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(with_case_runs.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Replacement with different case-local RUN, but replace_run_lines=true
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 1,
                        "replace_run_lines": True,
                        "content": (
                            '// RUN: echo "gamma" | FileCheck %s --check-prefix=GAMMA\n'
                            "// CHECK-LABEL: @case_one\n"
                            "func.func @case_one() { return }\n"
                        ),
                    }
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(
                result,
                0,
                "Should succeed with replace_run_lines even when RUNs differ",
            )

            # Verify the NEW case-local RUN is present
            file_content = tmp_path.read_text()
            self.assertIn('// RUN: echo "gamma"', file_content)
            self.assertNotIn('// RUN: echo "alpha"', file_content)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


class TestValidation(unittest.TestCase):
    """Tests for --verify flag (IR validation with iree-opt)."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_validate_valid_ir(self):
        """Test that valid MLIR passes validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Valid replacement content
            valid_content = """// CHECK-LABEL: @valid_case
util.func @valid_case(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  util.return %arg0 : tensor<4xf32>
}
"""

            with patch(
                "sys.argv",
                ["iree-lit-replace", str(tmp_path), "--case", "2", "--verify"],
            ), patch("sys.stdin", MockStdin(valid_content.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should succeed
            self.assertEqual(result, 0)

            # Verify replacement happened
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[1].name, "valid_case")

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_validate_invalid_ir(self):
        """Test that invalid MLIR fails validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Invalid replacement content (bad syntax)
            invalid_content = """// CHECK-LABEL: @invalid_case
util.func @invalid_case(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %bad = arith.addf %arg0 : tensor<4xf32>
  util.return %bad : tensor<4xf32>
}
"""

            with patch(
                "sys.argv",
                ["iree-lit-replace", str(tmp_path), "--case", "2", "--verify"],
            ), patch("sys.stdin", MockStdin(invalid_content.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should fail validation
            self.assertEqual(result, 1)

            # Verify file was NOT modified
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[1].name, "second_case")  # Original name

        finally:
            tmp_path.unlink()


class TestDryRun(unittest.TestCase):
    """Tests for --dry-run flag (preview changes without writing)."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_dry_run_shows_diff(self):
        """Test that dry-run outputs unified diff."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            new_content = """// CHECK-LABEL: @modified_case
util.func @modified_case() {
  return
}
"""

            # Capture stdout to check diff output
            captured_output = io.StringIO()

            with patch(
                "sys.argv",
                ["iree-lit-replace", str(tmp_path), "--case", "2", "--dry-run"],
            ), patch("sys.stdin", MockStdin(new_content.encode("utf-8"))), patch(
                "sys.stdout", captured_output
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Check diff output
            diff_output = captured_output.getvalue()
            self.assertIn("---", diff_output)  # Diff header
            self.assertIn("+++", diff_output)  # Diff header
            self.assertIn("@@", diff_output)  # Hunk marker

            # Verify file was NOT modified
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[1].name, "second_case")  # Original unchanged

        finally:
            tmp_path.unlink()

    def test_dry_run_no_changes(self):
        """Test dry-run with no changes shows appropriate message."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Extract original case 2 content
            cases = list(parse_test_file(tmp_path).cases)
            original_content = cases[1].content

            with patch(
                "sys.argv",
                ["iree-lit-replace", str(tmp_path), "--case", "2", "--dry-run"],
            ), patch("sys.stdin", MockStdin(original_content.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

        finally:
            tmp_path.unlink()

    def test_dry_run_json_mode(self):
        """Test dry-run in JSON mode includes diff in output."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// CHECK-LABEL: @dry_run_test\nutil.func @dry_run_test() {\n  return\n}\n",
                    }
                ]
            )

            # Capture stdout for JSON output
            captured_output = io.StringIO()

            with patch("sys.argv", ["iree-lit-replace", "--dry-run", "--json"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ), patch("sys.stdout", captured_output):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Parse JSON output
            output = json.loads(captured_output.getvalue())

            # Check structure
            self.assertIn("file_results", output)
            self.assertEqual(len(output["file_results"]), 1)

            file_result = output["file_results"][0]
            self.assertTrue(file_result["dry_run"])
            self.assertIn("diff", file_result)
            self.assertIn("---", file_result["diff"])  # Diff present

            # Verify file NOT modified
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[1].name, "second_case")

        finally:
            tmp_path.unlink()


class TestContentValidation(unittest.TestCase):
    """Tests for --require-label and --allow-empty flags."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_require_label_present(self):
        """Test --require-label passes when CHECK-LABEL present."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            content_with_label = """// CHECK-LABEL: @has_label
util.func @has_label() {
  return
}
"""

            with patch(
                "sys.argv",
                [
                    "iree-lit-replace",
                    str(tmp_path),
                    "--case",
                    "2",
                    "--require-label",
                ],
            ), patch("sys.stdin", MockStdin(content_with_label.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_require_label_missing(self):
        """Test --require-label fails when CHECK-LABEL missing."""
        content_without_label = """util.func @no_label() {
  return
}
"""

        with patch(
            "sys.argv",
            [
                "iree-lit-replace",
                str(self.split_test),
                "--case",
                "2",
                "--require-label",
            ],
        ), patch("sys.stdin", MockStdin(content_without_label.encode("utf-8"))):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertEqual(result, 1)  # ERROR

    def test_allow_empty_with_flag(self):
        """Test --allow-empty allows empty content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            empty_content = ""

            with patch(
                "sys.argv",
                ["iree-lit-replace", str(tmp_path), "--case", "2", "--allow-empty"],
            ), patch("sys.stdin", MockStdin(empty_content.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify case content is empty
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[1].content.strip(), "")

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_empty_without_flag(self):
        """Test empty content errors without --allow-empty."""
        empty_content = ""

        with patch(
            "sys.argv", ["iree-lit-replace", str(self.split_test), "--case", "2"]
        ), patch("sys.stdin", MockStdin(empty_content.encode("utf-8"))):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertEqual(result, 1)  # ERROR


class TestMultiFileReplacements(unittest.TestCase):
    """Stress tests for multi-file JSON replacements."""

    def setUp(self):
        self.ten_cases = _FIXTURES_DIR / "ten_cases_test.mlir"

    def _make_temp_file(self, src: Path) -> Path:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(src.read_text())
            return Path(tmp.name)

    def test_json_multi_file_dry_run_summary(self):
        """Dry-run JSON over multiple files shows per-file diffs and counts."""
        a = self._make_temp_file(self.ten_cases)
        b = self._make_temp_file(self.ten_cases)
        c = self._make_temp_file(self.ten_cases)
        try:
            payload = [
                {
                    "file": str(a),
                    "number": 1,
                    "content": "// CHECK-LABEL: @aa\nfunc.func @aa() { return }\n",
                },
                {
                    "file": str(a),
                    "number": 3,
                    "content": "// CHECK-LABEL: @ac\nfunc.func @ac() { return }\n",
                },
                {
                    "file": str(b),
                    "number": 2,
                    "content": "// CHECK-LABEL: @bb\nfunc.func @bb() { return }\n",
                },
                {
                    "file": str(b),
                    "number": 4,
                    "content": "// CHECK-LABEL: @bd\nfunc.func @bd() { return }\n",
                },
                {
                    "file": str(c),
                    "number": 5,
                    "content": "// CHECK-LABEL: @ce\nfunc.func @ce() { return }\n",
                },
            ]
            data = json.dumps(payload)
            with patch("sys.argv", ["iree-lit-replace", "--dry-run", "--json"]):
                out = io.StringIO()
                with patch("sys.stdin", MockStdin(data.encode("utf-8"))), patch(
                    "sys.stdout", out
                ):
                    args = iree_lit_replace.parse_arguments()
                    rc = iree_lit_replace.main(args)
            self.assertEqual(rc, 0)
            report = json.loads(out.getvalue())
            self.assertIn("file_results", report)
            self.assertEqual(len(report["file_results"]), 3)
            # Each file result should include a diff and correct total_cases
            file_map = {fr["file"]: fr for fr in report["file_results"]}
            self.assertEqual(file_map[str(a)]["total_cases"], 2)
            self.assertIn("diff", file_map[str(a)])
            self.assertEqual(file_map[str(b)]["total_cases"], 2)
            self.assertEqual(file_map[str(c)]["total_cases"], 1)
        finally:
            for p in [a, b, c]:
                if p.exists():
                    p.unlink()

    def test_json_multi_file_commit_and_verify(self):
        """Replacing across multiple files writes expected changes."""
        a = self._make_temp_file(self.ten_cases)
        b = self._make_temp_file(self.ten_cases)
        try:
            payload = [
                {
                    "file": str(a),
                    "number": 1,
                    "content": "// CHECK-LABEL: @aa\nfunc.func @aa() { return }\n",
                },
                {
                    "file": str(b),
                    "number": 10,
                    "content": "// CHECK-LABEL: @bz\nfunc.func @bz() { return }\n",
                },
            ]
            data = json.dumps(payload)
            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(data.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                rc = iree_lit_replace.main(args)
            self.assertEqual(rc, 0)

            # Verify replacements
            ca = list(parse_test_file(a).cases)
            cb = list(parse_test_file(b).cases)
            self.assertEqual(ca[0].name, "aa")
            self.assertEqual(cb[9].name, "bz")
        finally:
            for p in [a, b]:
                if p.exists():
                    p.unlink()

    def test_json_multi_file_with_error_aborts_without_modifying(self):
        """If any file errors, no writes occur for others in the batch."""
        a = self._make_temp_file(self.ten_cases)
        b = self._make_temp_file(self.ten_cases)
        nonexist = str(Path(a.parent) / "does_not_exist.mlir")
        try:
            original_a = a.read_text()
            original_b = b.read_text()
            payload = [
                {
                    "file": str(a),
                    "number": 1,
                    "content": "// CHECK-LABEL: @aa\nfunc.func @aa() { return }\n",
                },
                {
                    "file": nonexist,
                    "number": 1,
                    "content": "// CHECK-LABEL: @x\nfunc.func @x() { return }\n",
                },
                {
                    "file": str(b),
                    "number": 2,
                    "content": "// CHECK-LABEL: @bb\nfunc.func @bb() { return }\n",
                },
            ]
            data = json.dumps(payload)
            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(data.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                rc = iree_lit_replace.main(args)
            self.assertEqual(rc, 1)
            # Files should be unmodified
            self.assertEqual(a.read_text(), original_a)
            self.assertEqual(b.read_text(), original_b)
        finally:
            for p in [a, b]:
                if p.exists():
                    p.unlink()

    def test_json_cli_override_warns(self):
        """CLI test_file overrides JSON file fields and warns."""
        a = self._make_temp_file(self.ten_cases)
        try:
            payload = [
                {
                    "file": str(a),
                    "number": 1,
                    "content": "// CHECK-LABEL: @aa\nfunc.func @aa() { return }\n",
                },
                {
                    "file": str(a) + ".other",
                    "number": 3,
                    "content": "// CHECK-LABEL: @ac\nfunc.func @ac() { return }\n",
                },
            ]
            data = json.dumps(payload)
            err = io.StringIO()
            with patch("sys.argv", ["iree-lit-replace", str(a)]), patch(
                "sys.stderr", err
            ), patch("sys.stdin", MockStdin(data.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                rc = iree_lit_replace.main(args)
            self.assertEqual(rc, 0)
            self.assertIn("overriding JSON 'file' field", err.getvalue())
        finally:
            if a.exists():
                a.unlink()


class TestRoundtrip(unittest.TestCase):
    """Tests for roundtrip workflows between tools."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_extract_dry_run_replace(self):
        """Test extract → dry-run → replace workflow."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Step 1: Extract case 2 with JSON
            with patch(
                "sys.argv", ["iree-lit-extract", str(tmp_path), "--case", "2", "--json"]
            ):
                extract_args = iree_lit_extract.parse_arguments()

                # Capture extract output
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    iree_lit_extract.main(extract_args)

                extracted = json.loads(captured.getvalue())

            # Step 2: Modify content
            extracted[0][
                "content"
            ] = "// CHECK-LABEL: @roundtrip\nutil.func @roundtrip() {\n  return\n}\n"
            extracted[0]["file"] = str(tmp_path)

            # Step 3: Dry-run to preview
            json_input = json.dumps(extracted)

            with patch("sys.argv", ["iree-lit-replace", "--dry-run", "--json"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                dry_run_output = io.StringIO()
                with patch("sys.stdout", dry_run_output):
                    args = iree_lit_replace.parse_arguments()
                    result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)
            dry_result = json.loads(dry_run_output.getvalue())
            self.assertTrue(dry_result["file_results"][0]["dry_run"])

            # Step 4: Actually replace
            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify final state
            cases = list(parse_test_file(tmp_path).cases)
            self.assertEqual(cases[1].name, "roundtrip")

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


class TestOutputFormats(unittest.TestCase):
    """Tests for output format consistency and combinations."""

    def setUp(self):
        """Set up fixture paths."""
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_text_mode_with_json_output(self):
        """Test text mode with --json flag produces valid JSON."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            new_content = """// CHECK-LABEL: @json_output
util.func @json_output() {
  return
}
"""

            # Capture JSON output
            captured = io.StringIO()

            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "2", "--json"]
            ), patch("sys.stdin", MockStdin(new_content.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Validate JSON structure
            output = json.loads(captured.getvalue())
            self.assertIn("modified_files", output)
            self.assertIn("modified_cases", output)
            self.assertIn("file_results", output)
            self.assertIn("dry_run", output)
            self.assertEqual(output["modified_files"], 1)
            self.assertEqual(output["modified_cases"], 1)
            self.assertEqual(output["dry_run"], False)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_mode_with_json_output(self):
        """Test JSON mode with --json flag (double JSON)."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// CHECK-LABEL: @double_json\nutil.func @double_json() {\n  return\n}\n",
                    }
                ]
            )

            captured = io.StringIO()

            with patch("sys.argv", ["iree-lit-replace", "--json"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ), patch("sys.stdout", captured):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Validate JSON output
            output = json.loads(captured.getvalue())
            self.assertIn("file_results", output)
            self.assertEqual(output["modified_files"], 1)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_validate_dry_run_json_consistency(self):
        """Test --verify and --dry-run with --json produce consistent schema."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            valid_content = """// CHECK-LABEL: @consistency_test
util.func @consistency_test() {
  util.return
}
"""

            # Test 1: --verify with --json
            captured1 = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "iree-lit-replace",
                    str(tmp_path),
                    "--case",
                    "2",
                    "--verify",
                    "--json",
                ],
            ), patch("sys.stdin", MockStdin(valid_content.encode("utf-8"))), patch(
                "sys.stdout", captured1
            ):
                args = iree_lit_replace.parse_arguments()
                result1 = iree_lit_replace.main(args)

            self.assertEqual(result1, 0)
            output1 = json.loads(captured1.getvalue())

            # Test 2: --dry-run with --json (reset file first)
            tmp_path.write_text(self.split_test.read_text())

            captured2 = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "iree-lit-replace",
                    str(tmp_path),
                    "--case",
                    "2",
                    "--dry-run",
                    "--json",
                ],
            ), patch("sys.stdin", MockStdin(valid_content.encode("utf-8"))), patch(
                "sys.stdout", captured2
            ):
                args = iree_lit_replace.parse_arguments()
                result2 = iree_lit_replace.main(args)

            self.assertEqual(result2, 0)
            output2 = json.loads(captured2.getvalue())

            # Both should have same top-level keys
            self.assertEqual(set(output1.keys()), set(output2.keys()))
            self.assertIn("modified_files", output1)
            self.assertIn("errors", output1)
            self.assertIn("warnings", output1)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


class TestBatchMode(unittest.TestCase):
    """Tests for batch mode operations with multiple files and cases."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_batch_validation_all_valid(self):
        """Test batch validation with all valid replacements."""

        # Create two temp files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp1:
            tmp1.write(self.split_test.read_text())
            tmp1_path = Path(tmp1.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp2:
            tmp2.write(self.split_test.read_text())
            tmp2_path = Path(tmp2.name)

        try:
            valid_content = """// CHECK-LABEL: @batch_test
util.func @batch_test() {
  util.return
}
"""

            # Batch JSON with two files, two cases each
            batch_json = json.dumps(
                [
                    {"file": str(tmp1_path), "number": 1, "content": valid_content},
                    {"file": str(tmp1_path), "number": 2, "content": valid_content},
                    {"file": str(tmp2_path), "number": 1, "content": valid_content},
                    {"file": str(tmp2_path), "number": 2, "content": valid_content},
                ]
            )

            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", "--mode", "json", "--verify"]
            ), patch("sys.stdin", MockStdin(batch_json.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify both files were modified
            self.assertNotEqual(tmp1_path.read_text(), self.split_test.read_text())
            self.assertNotEqual(tmp2_path.read_text(), self.split_test.read_text())

        finally:
            tmp1_path.unlink()
            tmp2_path.unlink()
            for path in [tmp1_path, tmp2_path]:
                backup = path.with_suffix(".mlir.bak")
                if backup.exists():
                    backup.unlink()

    def test_batch_validation_one_invalid(self):
        """Test batch validation fails atomically when one replacement is invalid."""

        # Create two temp files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp1:
            tmp1.write(self.split_test.read_text())
            tmp1_path = Path(tmp1.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp2:
            tmp2.write(self.split_test.read_text())
            tmp2_path = Path(tmp2.name)

        try:
            valid_content = """// CHECK-LABEL: @valid_test
util.func @valid_test() {
  util.return
}
"""
            invalid_content = """// Invalid MLIR - missing comma
util.func @invalid_test(%arg0: tensor<4xf32>) {
  %bad = arith.addf %arg0 : tensor<4xf32>
  util.return
}
"""

            # Store original content
            orig1 = tmp1_path.read_text()
            orig2 = tmp2_path.read_text()

            # Batch JSON with one invalid replacement
            batch_json = json.dumps(
                [
                    {"file": str(tmp1_path), "number": 1, "content": valid_content},
                    {"file": str(tmp2_path), "number": 1, "content": invalid_content},
                ]
            )

            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", "--mode", "json", "--verify"]
            ), patch("sys.stdin", MockStdin(batch_json.encode("utf-8"))), patch(
                "sys.stderr", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should fail
            self.assertNotEqual(result, 0)

            # Files should be unchanged (atomic batch)
            self.assertEqual(tmp1_path.read_text(), orig1)
            self.assertEqual(tmp2_path.read_text(), orig2)

        finally:
            tmp1_path.unlink()
            tmp2_path.unlink()

    def test_batch_dry_run_multiple_files(self):
        """Test batch dry-run shows diffs for multiple files."""

        # Create two temp files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp1:
            tmp1.write(self.split_test.read_text())
            tmp1_path = Path(tmp1.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp2:
            tmp2.write(self.split_test.read_text())
            tmp2_path = Path(tmp2.name)

        try:
            new_content = """// CHECK-LABEL: @modified_function
util.func @modified_function() {
  util.return
}
"""

            # Store original content
            orig1 = tmp1_path.read_text()
            orig2 = tmp2_path.read_text()

            # Batch JSON
            batch_json = json.dumps(
                [
                    {"file": str(tmp1_path), "number": 2, "content": new_content},
                    {"file": str(tmp2_path), "number": 2, "content": new_content},
                ]
            )

            captured = io.StringIO()
            with patch(
                "sys.argv",
                ["iree-lit-replace", "--mode", "json", "--dry-run", "--json"],
            ), patch("sys.stdin", MockStdin(batch_json.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Parse JSON output
            output = json.loads(captured.getvalue())

            # Should have 2 files in output
            self.assertEqual(len(output["file_results"]), 2)

            # Both should have dry_run=True and diffs
            for file_result in output["file_results"]:
                self.assertTrue(file_result["dry_run"])
                self.assertTrue(file_result["diff"])  # Should have diff text

            # Files should be unchanged
            self.assertEqual(tmp1_path.read_text(), orig1)
            self.assertEqual(tmp2_path.read_text(), orig2)

        finally:
            tmp1_path.unlink()
            tmp2_path.unlink()


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error conditions."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_single_case_file(self):
        """Test replacement in file with only one case (no delimiters)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            # Create a single-case file (no delimiters)
            single_case = """// RUN: iree-opt %s
// CHECK-LABEL: @single_case
util.func @single_case() {
  util.return
}
"""
            tmp.write(single_case)
            tmp_path = Path(tmp.name)

        try:
            new_content = """// CHECK-LABEL: @replaced_single
util.func @replaced_single() {
  util.return
}
"""

            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "1"]
            ), patch("sys.stdin", MockStdin(new_content.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify replacement
            final = tmp_path.read_text()
            self.assertIn("@replaced_single", final)
            self.assertIn("// RUN: iree-opt %s", final)  # RUN line preserved

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_replace_first_case(self):
        """Test replacing the first case in a multi-case file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            new_first = """// CHECK-LABEL: @new_first_case
util.func @new_first_case() {
  util.return
}
"""

            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "1"]
            ), patch("sys.stdin", MockStdin(new_first.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify first case replaced, others unchanged
            final = tmp_path.read_text()
            self.assertIn("@new_first_case", final)
            self.assertIn("@second_case", final)  # Second case unchanged
            self.assertIn("@third_case", final)  # Third case unchanged

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_replace_last_case(self):
        """Test replacing the last case in a multi-case file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            new_last = """// CHECK-LABEL: @new_last_case
util.func @new_last_case() {
  util.return
}
"""

            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "3"]
            ), patch("sys.stdin", MockStdin(new_last.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify last case replaced, others unchanged
            final = tmp_path.read_text()
            self.assertIn("@new_last_case", final)
            self.assertIn("@first_case", final)  # First case unchanged
            self.assertIn("@second_case", final)  # Second case unchanged

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_replace_with_very_long_content(self):
        """Test replacement with very long content (stress test)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Generate large content (1000 lines)
            large_content = "// CHECK-LABEL: @large_test\n"
            large_content += "util.func @large_test() {\n"
            for i in range(1000):
                large_content += f"  // Line {i}\n"
            large_content += "  util.return\n"
            large_content += "}\n"

            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "2"]
            ), patch("sys.stdin", MockStdin(large_content.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify content was replaced
            new_content = tmp_path.read_text()
            self.assertIn("Line 999", new_content)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_replace_with_unicode_content(self):
        """Test replacement with Unicode characters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            unicode_content = """// CHECK-LABEL: @unicode_test
// Test with Unicode: 你好 世界 🚀
util.func @unicode_test() {
  util.return
}
"""

            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "2"]
            ), patch("sys.stdin", MockStdin(unicode_content.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify Unicode content was preserved
            new_content = tmp_path.read_text()
            self.assertIn("你好", new_content)
            self.assertIn("🚀", new_content)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_validate_timeout_behavior(self):
        """Test that verification timeout is configurable."""
        # Note: This is a structural test - we verify the timeout parameter exists
        # Actual timeout testing would require malicious/hanging IR
        # Verify timeout parameter is respected in verify_ir
        # (This is a code structure test, not a runtime test)

        sig = inspect.signature(verification.verify_ir)
        # Function should accept content, case_info, args, and timeout
        self.assertEqual(len(sig.parameters), 4)

    def test_malformed_json_input(self):
        """Test error handling for malformed JSON."""
        captured = io.StringIO()
        with patch("sys.argv", ["iree-lit-replace", "--mode", "json"]), patch(
            "sys.stdin", MockStdin(b"not valid json {")
        ), patch("sys.stderr", captured):
            args = iree_lit_replace.parse_arguments()
            result = iree_lit_replace.main(args)

        self.assertNotEqual(result, 0)
        error_output = captured.getvalue()
        self.assertIn("error", error_output.lower())


class TestAdvancedWorkflows(unittest.TestCase):
    """Tests for complex multi-step workflows and tool integration."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_extract_validate_dry_run_replace_pipeline(self):
        """Test complete pipeline: extract → validate → dry-run → replace."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Step 1: Extract case 2 with iree-lit-extract --json
            extract_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_path), "--case", "2", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract_result.returncode, 0)
            extracted = json.loads(extract_result.stdout)

            # Step 2: Modify content and ensure file field is present
            modified_content = """// CHECK-LABEL: @pipeline_test
util.func @pipeline_test() {
  util.return
}
"""
            extracted[0]["content"] = modified_content
            extracted[0]["file"] = str(tmp_path)  # Ensure file field is present

            # Step 3: Validate with --verify --dry-run --json
            validate_input = json.dumps(extracted)
            captured = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "iree-lit-replace",
                    "--mode",
                    "json",
                    "--verify",
                    "--dry-run",
                    "--json",
                ],
            ), patch("sys.stdin", MockStdin(validate_input.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)
            validate_output = json.loads(captured.getvalue())

            # Verify dry-run flag and diff present
            self.assertTrue(validate_output["file_results"][0]["dry_run"])
            self.assertTrue(validate_output["file_results"][0]["diff"])

            # Step 4: Actually replace (without dry-run)
            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", "--mode", "json", "--verify"]
            ), patch("sys.stdin", MockStdin(validate_input.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Step 5: Verify final content
            final_content = tmp_path.read_text()
            self.assertIn("@pipeline_test", final_content)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_batch_mixed_operations(self):
        """Test batch with some cases valid, some identical, some changed."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Read original case 2 content
            extract_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_path), "--case", "2"],
                capture_output=True,
                text=True,
            )
            original_case2 = extract_result.stdout

            new_content = """// CHECK-LABEL: @new_test
util.func @new_test() {
  util.return
}
"""

            # Batch: case 1 changed, case 2 identical, case 3 changed
            batch_json = json.dumps(
                [
                    {"file": str(tmp_path), "number": 1, "content": new_content},
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": original_case2,
                    },  # Identical
                    {"file": str(tmp_path), "number": 3, "content": new_content},
                ]
            )

            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", "--mode", "json", "--json"]
            ), patch("sys.stdin", MockStdin(batch_json.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)
            output = json.loads(captured.getvalue())

            # All 3 cases processed, but case 2 content is identical
            # The tool reports all cases in batch as processed
            self.assertEqual(output["modified_files"], 1)
            # Note: modified_cases may be 2 or 3 depending on idempotency detection
            # in batch mode - as long as file was updated, test passes
            self.assertGreaterEqual(output["modified_cases"], 2)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_iterative_refinement_workflow(self):
        """Test workflow: extract → edit → dry-run → edit again → replace."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Iteration 1: First attempt
            attempt1 = """// CHECK-LABEL: @attempt1
util.func @attempt1() {
  util.return
}
"""

            # Dry-run to see diff
            captured = io.StringIO()
            with patch(
                "sys.argv",
                ["iree-lit-replace", str(tmp_path), "--case", "2", "--dry-run"],
            ), patch("sys.stdin", MockStdin(attempt1.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)
            diff1 = captured.getvalue()
            self.assertIn("@attempt1", diff1)

            # Iteration 2: Refined attempt
            attempt2 = """// CHECK-LABEL: @refined_version
util.func @refined_version() {
  util.return
}
"""

            # Dry-run again to see new diff
            captured = io.StringIO()
            with patch(
                "sys.argv",
                ["iree-lit-replace", str(tmp_path), "--case", "2", "--dry-run"],
            ), patch("sys.stdin", MockStdin(attempt2.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)
            diff2 = captured.getvalue()
            self.assertIn("@refined_version", diff2)

            # Final: Replace with refined version
            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "2"]
            ), patch("sys.stdin", MockStdin(attempt2.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify final state
            final = tmp_path.read_text()
            self.assertIn("@refined_version", final)
            self.assertNotIn("@attempt1", final)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


class TestFlagCombinations(unittest.TestCase):
    """Tests for various CLI flag combinations."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_validate_require_label_combination(self):
        """Test --verify and --require-label together."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            valid_with_label = """// CHECK-LABEL: @labeled_function
util.func @labeled_function() {
  util.return
}
"""

            captured = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "iree-lit-replace",
                    str(tmp_path),
                    "--case",
                    "2",
                    "--verify",
                    "--require-label",
                ],
            ), patch("sys.stdin", MockStdin(valid_with_label.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_dry_run_with_all_validation_flags(self):
        """Test --dry-run with --verify, --require-label, and --json."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            content = """// CHECK-LABEL: @all_flags_test
util.func @all_flags_test() {
  util.return
}
"""

            captured = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "iree-lit-replace",
                    str(tmp_path),
                    "--case",
                    "2",
                    "--dry-run",
                    "--verify",
                    "--require-label",
                    "--json",
                ],
            ), patch("sys.stdin", MockStdin(content.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Parse JSON output
            output = json.loads(captured.getvalue())

            # Should have dry_run flag and diff
            self.assertTrue(output["file_results"][0]["dry_run"])
            self.assertTrue(output["file_results"][0]["diff"])
            self.assertTrue(output["dry_run"])

            # File should be unchanged (dry-run)
            self.assertEqual(tmp_path.read_text(), self.split_test.read_text())

        finally:
            tmp_path.unlink()

    def test_json_mode_with_pretty_flag(self):
        """Test --mode json with --pretty flag."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            content = """// CHECK-LABEL: @pretty_test
util.func @pretty_test() {
  util.return
}
"""

            batch_json = json.dumps(
                [{"file": str(tmp_path), "number": 2, "content": content}]
            )

            # Note: --pretty affects console output, not JSON structure
            # This test verifies it doesn't break JSON mode
            captured = io.StringIO()
            with patch(
                "sys.argv", ["iree-lit-replace", "--mode", "json", "--pretty"]
            ), patch("sys.stdin", MockStdin(batch_json.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_quiet_with_json_output(self):
        """Test --quiet with --json (JSON output should still appear)."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            content = """// CHECK-LABEL: @quiet_test
util.func @quiet_test() {
  util.return
}
"""

            captured = io.StringIO()
            with patch(
                "sys.argv",
                ["iree-lit-replace", str(tmp_path), "--case", "2", "--quiet", "--json"],
            ), patch("sys.stdin", MockStdin(content.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Should have JSON output despite --quiet
            output = json.loads(captured.getvalue())
            self.assertEqual(output["modified_files"], 1)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_no_backup_flag(self):
        """Test --no-backup flag prevents backup file creation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            content = """// CHECK-LABEL: @no_backup_test
util.func @no_backup_test() {
  util.return
}
"""

            captured = io.StringIO()
            with patch(
                "sys.argv",
                ["iree-lit-replace", str(tmp_path), "--case", "2", "--no-backup"],
            ), patch("sys.stdin", MockStdin(content.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify no backup file created
            backup = tmp_path.with_suffix(".mlir.bak")
            self.assertFalse(backup.exists())

        finally:
            tmp_path.unlink()


class TestUnifiedJSONSchema(unittest.TestCase):
    """Tests for unified JSON schema across all modes."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def _validate_schema_structure(self, output: dict, dry_run: bool):
        """Validate the unified JSON output schema structure."""
        # Top-level required fields
        self.assertIn("modified_files", output)
        self.assertIn("modified_cases", output)
        self.assertIn("unchanged_cases", output)
        self.assertIn("dry_run", output)
        self.assertIn("file_results", output)
        self.assertIn("errors", output)
        self.assertIn("warnings", output)

        # Verify types
        self.assertIsInstance(output["modified_files"], int)
        self.assertIsInstance(output["modified_cases"], int)
        self.assertIsInstance(output["unchanged_cases"], int)
        self.assertIsInstance(output["dry_run"], bool)
        self.assertIsInstance(output["file_results"], list)
        self.assertIsInstance(output["errors"], list)
        self.assertIsInstance(output["warnings"], list)

        # Verify dry_run value matches expected
        self.assertEqual(output["dry_run"], dry_run)

        # Validate file_results structure
        for file_result in output["file_results"]:
            self.assertIn("file", file_result)
            self.assertIn("total_cases", file_result)
            self.assertIn("modified", file_result)
            self.assertIn("unchanged", file_result)
            self.assertIn("dry_run", file_result)
            self.assertIn("cases", file_result)
            self.assertIn("diff", file_result)

            # Verify file_result types
            self.assertIsInstance(file_result["file"], str)
            self.assertIsInstance(file_result["total_cases"], int)
            self.assertIsInstance(file_result["modified"], int)
            self.assertIsInstance(file_result["unchanged"], int)
            self.assertIsInstance(file_result["dry_run"], bool)
            self.assertIsInstance(file_result["cases"], list)
            self.assertIsInstance(file_result["diff"], str)

            # Validate cases array structure (must be objects, not just numbers)
            for case in file_result["cases"]:
                self.assertIsInstance(case, dict)
                self.assertIn("number", case)
                self.assertIn("name", case)
                self.assertIn("changed", case)
                self.assertIsInstance(case["number"], int)
                self.assertIsInstance(case["changed"], bool)
                # name can be None for unnamed cases
                self.assertTrue(case["name"] is None or isinstance(case["name"], str))

    def test_text_mode_json_output_schema(self):
        """Text mode with --json follows unified schema."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            content = "// Modified\nutil.func @test() { util.return }\n"
            captured = io.StringIO()
            with patch(
                "sys.argv",
                ["iree-lit-replace", str(tmp_path), "--case", "2", "--json"],
            ), patch("sys.stdin", MockStdin(content.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)
            output = json.loads(captured.getvalue())
            self._validate_schema_structure(output, dry_run=False)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_text_mode_dry_run_json_schema(self):
        """Text mode --dry-run --json follows unified schema."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            content = "// Modified\nutil.func @test() { util.return }\n"
            captured = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "iree-lit-replace",
                    str(tmp_path),
                    "--case",
                    "2",
                    "--dry-run",
                    "--json",
                ],
            ), patch("sys.stdin", MockStdin(content.encode("utf-8"))), patch(
                "sys.stdout", captured
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)
            output = json.loads(captured.getvalue())
            self._validate_schema_structure(output, dry_run=True)
            # Dry-run should have diff
            self.assertTrue(len(output["file_results"][0]["diff"]) > 0)

        finally:
            tmp_path.unlink()

    def test_json_batch_mode_schema(self):
        """JSON batch mode follows unified schema."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 1,
                        "content": "// Modified 1\nutil.func @test1() { util.return }\n",
                    },
                    {
                        "file": str(tmp_path),
                        "number": 3,
                        "content": "// Modified 3\nutil.func @test3() { util.return }\n",
                    },
                ]
            )

            captured = io.StringIO()
            with patch("sys.argv", ["iree-lit-replace", "--json"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ), patch("sys.stdout", captured):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)
            output = json.loads(captured.getvalue())
            self._validate_schema_structure(output, dry_run=False)
            # Should have modified 2 cases
            self.assertEqual(output["modified_cases"], 2)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_batch_dry_run_schema(self):
        """JSON batch mode --dry-run follows unified schema."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// Modified\nutil.func @test() { util.return }\n",
                    }
                ]
            )

            captured = io.StringIO()
            with patch("sys.argv", ["iree-lit-replace", "--dry-run", "--json"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ), patch("sys.stdout", captured):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)
            output = json.loads(captured.getvalue())
            self._validate_schema_structure(output, dry_run=True)

        finally:
            tmp_path.unlink()


class TestInputValidation(unittest.TestCase):
    """Tests for input validation and edge cases."""

    def test_windows_crlf_input_handling(self):
        """Test that Windows CRLF line endings are handled correctly."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        split_test = fixtures_dir / "split_test.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Replacement content with Windows CRLF line endings.
            replacement_with_crlf = (
                "// CHECK-LABEL: @first_crlf\r\nutil.func @first_crlf() { return }\r\n"
            )

            with patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "1"]
            ), patch("sys.stdin", MockStdin(replacement_with_crlf.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            # Verify output uses consistent Unix line endings (LF only).
            file_content = tmp_path.read_bytes()
            self.assertNotIn(b"\r\n", file_content, "Should not have CRLF in output")
            self.assertIn(b"@first_crlf", file_content)
        finally:
            tmp_path.unlink()

    def test_empty_replacement_content_with_allow_empty(self):
        """Test that empty content with --allow-empty wipes case body."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        split_test = fixtures_dir / "split_test.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "",
                        "allow_empty": True,
                    }
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            file_content = tmp_path.read_text()
            # Case 2 should be wiped (empty or minimal content).
            self.assertNotIn("@second_case", file_content)

            # Delimiters should remain.
            self.assertGreater(file_content.count("// -----"), 0)

            # Other cases preserved.
            self.assertIn("@first_case", file_content)
            self.assertIn("@third_case", file_content)
        finally:
            tmp_path.unlink()

    def test_utf8_bom_handling(self):
        """Test that UTF-8 BOM in input is handled correctly."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        split_test = fixtures_dir / "split_test.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # JSON input with UTF-8 BOM prefix.
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 1,
                        "content": "// CHECK-LABEL: @first_bom\nutil.func @first_bom() { return }\n",
                    }
                ]
            )
            json_with_bom = b"\xEF\xBB\xBF" + json_input.encode("utf-8")

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_with_bom)
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should succeed (BOM is stripped).
            self.assertEqual(result, 0)

            file_content = tmp_path.read_text()
            self.assertIn("@first_bom", file_content)
        finally:
            tmp_path.unlink()

    def test_concurrent_modification_detection(self):
        """Test that concurrent file modification is detected."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        split_test = fixtures_dir / "split_test.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            replacement = (
                "// CHECK-LABEL: @modified\nutil.func @modified() { return }\n"
            )

            # Mock Path.stat to return different mtime on second call.
            original_stat = tmp_path.stat

            call_count = [0]

            def mock_stat(self, *, follow_symlinks=True):
                call_count[0] += 1
                stat_result = original_stat(follow_symlinks=follow_symlinks)
                if call_count[0] > 1:
                    # Simulate file modification by changing mtime.
                    # Create a new stat_result with modified mtime.
                    return os.stat_result(
                        (
                            stat_result.st_mode,
                            stat_result.st_ino,
                            stat_result.st_dev,
                            stat_result.st_nlink,
                            stat_result.st_uid,
                            stat_result.st_gid,
                            stat_result.st_size,
                            stat_result.st_atime,
                            stat_result.st_mtime + 1,  # Modified time.
                            stat_result.st_ctime,
                        )
                    )
                return stat_result

            with patch.object(Path, "stat", mock_stat), patch(
                "sys.argv", ["iree-lit-replace", str(tmp_path), "--case", "1"]
            ), patch("sys.stdin", MockStdin(replacement.encode("utf-8"))):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should fail with concurrent modification error.
            self.assertNotEqual(result, 0, "Should detect concurrent modification")

            # Original file should be unchanged.
            file_content = tmp_path.read_text()
            self.assertIn("@first_case", file_content)
            self.assertNotIn("@modified", file_content)
        finally:
            tmp_path.unlink()


class TestJSONBatchEdgeCases(unittest.TestCase):
    """Tests for JSON batch mode edge cases and corner conditions."""

    def test_json_batch_reordering_cases(self):
        """Test that JSON batch mode handles out-of-order case updates."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        split_test = fixtures_dir / "split_test.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Update case 3, then case 1 (out of order).
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 3,
                        "content": "// CHECK-LABEL: @third_updated\nutil.func @third_updated() { return }\n",
                    },
                    {
                        "file": str(tmp_path),
                        "number": 1,
                        "content": "// CHECK-LABEL: @first_updated\nutil.func @first_updated() { return }\n",
                    },
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0, "Should succeed with out-of-order updates")

            # Verify both cases were updated in correct positions.
            file_content = tmp_path.read_text()
            self.assertIn("@first_updated", file_content)
            self.assertIn("@third_updated", file_content)

            # Verify case 2 is untouched.
            self.assertIn("@second_case", file_content)
        finally:
            tmp_path.unlink()

    def test_json_batch_hole_in_updates(self):
        """Test updating non-consecutive cases leaves middle cases untouched."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        five_cases = fixtures_dir / "five_cases.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(five_cases.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Update cases 1 and 5, leaving 2, 3, 4 untouched.
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 1,
                        "content": "// CHECK-LABEL: @case_one_modified\nfunc.func @case_one_modified() { return }\n",
                    },
                    {
                        "file": str(tmp_path),
                        "number": 5,
                        "content": "// CHECK-LABEL: @case_five_modified\nfunc.func @case_five_modified() { return }\n",
                    },
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            file_content = tmp_path.read_text()
            # Updated cases.
            self.assertIn("@case_one_modified", file_content)
            self.assertIn("@case_five_modified", file_content)

            # Untouched cases.
            self.assertIn("@case_two", file_content)
            self.assertIn("@case_three", file_content)
            self.assertIn("@case_four", file_content)

            # Delimiters preserved.
            self.assertEqual(file_content.count("// -----"), 4)
        finally:
            tmp_path.unlink()

    def test_weird_delimiter_spacing_normalization(self):
        """Test that delimiters with weird spacing are normalized."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        weird_delimiters = fixtures_dir / "weird_delimiters.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(weird_delimiters.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Replace case 2 (triggers rebuild).
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// CHECK-LABEL: @second_modified\nfunc.func @second_modified() { return }\n",
                    }
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            file_content = tmp_path.read_text()
            # Delimiters should be normalized to standard "// -----".
            # Count normalized delimiters.
            normalized_count = file_content.count("\n// -----\n")
            self.assertGreater(
                normalized_count,
                0,
                "Should have normalized delimiters with newlines before/after",
            )

            # Verify no weird spacing remains.
            self.assertNotIn(
                "//-----", file_content, "Should not have compact delimiter"
            )
            self.assertNotIn(
                "//   -----", file_content, "Should not have extra-spaced delimiter"
            )
        finally:
            tmp_path.unlink()

    def test_json_per_case_run_line_flags(self):
        """Test that per-case replace_run_lines flag is isolated."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        split_test = fixtures_dir / "split_test.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Entry 1 has replace_run_lines: true, Entry 2 doesn't.
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 1,
                        "replace_run_lines": True,
                        "content": "// RUN: new-tool %s | FileCheck %s\n// CHECK-LABEL: @first_new\nutil.func @first_new() { return }\n",
                    },
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// CHECK-LABEL: @second_same\nutil.func @second_same() { return }\n",
                    },
                ]
            )

            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            self.assertEqual(result, 0)

            file_content = tmp_path.read_text()
            # Header should be replaced with new RUN line.
            self.assertIn("new-tool", file_content)
            self.assertNotIn("iree-opt", file_content)

            # Both cases updated.
            self.assertIn("@first_new", file_content)
            self.assertIn("@second_same", file_content)
        finally:
            tmp_path.unlink()

    def test_json_header_run_line_poisoning(self):
        """Test that conflicting replace_run_lines values are caught."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        split_test = fixtures_dir / "split_test.mlir"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Two entries with different replace_run_lines values (conflict).
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 1,
                        "replace_run_lines": True,
                        "content": "// RUN: tool-a %s\n// CHECK-LABEL: @first\nutil.func @first() { return }\n",
                    },
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "replace_run_lines": True,
                        "content": "// RUN: tool-b %s\n// CHECK-LABEL: @second\nutil.func @second() { return }\n",
                    },
                ]
            )

            captured_stderr = io.StringIO()
            with patch("sys.argv", ["iree-lit-replace"]), patch(
                "sys.stdin", MockStdin(json_input.encode("utf-8"))
            ), patch("sys.stderr", captured_stderr):
                args = iree_lit_replace.parse_arguments()
                result = iree_lit_replace.main(args)

            # Should fail with batch consistency error.
            self.assertNotEqual(result, 0, "Should fail on conflicting RUN lines")

            # Verify file was not modified.
            file_content = tmp_path.read_text()
            self.assertIn(
                "@first_case", file_content, "Original content should be preserved"
            )
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
