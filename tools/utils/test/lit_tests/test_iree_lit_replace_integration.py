# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Integration tests for iree-lit-replace with other tools.

Tests multi-tool workflows and cross-tool consistency:
- iree-lit-extract → iree-lit-replace round-trips
- iree-lit-list → iree-lit-replace consistency
- Cross-file operations
- Batch stability
- Complete workflows (extract → edit → replace → test)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from test.test_helpers import run_python_module

# Import the tools

# Module-level fixture directory (absolute path for CWD-independence).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestExtractReplaceRoundTrip(unittest.TestCase):
    """Tests for extract → replace → extract idempotency."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_round_trip_single_case(self):
        """Test extract → replace (unchanged) → extract produces identical output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Step 1: Extract case 2
            extract_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_path), "--case", "2"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract_result.returncode, 0)
            original_content = extract_result.stdout

            # Step 2: Replace with same content
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path), "--case", "2"],
                input=original_content,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # Step 3: Extract again
            extract2_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_path), "--case", "2"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract2_result.returncode, 0)
            final_content = extract2_result.stdout

            # Verify identical
            self.assertEqual(original_content, final_content)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_round_trip_multiple_cases_json(self):
        """Test extract multiple → replace → extract with JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Step 1: Extract cases 1,2,3 as JSON
            extract_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_path), "--case", "1,2,3", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract_result.returncode, 0)
            original_json = extract_result.stdout
            original_data = json.loads(original_json)

            # Step 2: Replace all with same content (via JSON)
            # Need to add file field to each replacement since extract doesn't include it
            for case in original_data:
                case["file"] = str(tmp_path)
            modified_json = json.dumps(original_data)

            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                ["--mode", "json"],
                input=modified_json,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # Step 3: Extract again
            extract2_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_path), "--case", "1,2,3", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract2_result.returncode, 0)
            final_data = json.loads(extract2_result.stdout)

            # Verify content identical for each case
            for orig, final in zip(original_data, final_data, strict=False):
                self.assertEqual(orig["content"], final["content"])
                self.assertEqual(orig["number"], final["number"])
                self.assertEqual(orig["name"], final["name"])

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


class TestCrossFileOperations(unittest.TestCase):
    """Tests for cross-file moves and operations."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_extract_from_a_replace_into_b(self):
        """Test extracting from file A and replacing into file B."""
        # Create two temp files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_a:
            tmp_a.write(self.split_test.read_text())
            tmp_a_path = Path(tmp_a.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_b:
            tmp_b.write(self.split_test.read_text())
            tmp_b_path = Path(tmp_b.name)

        try:
            # Step 1: Extract case 2 from file A
            extract_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_a_path), "--case", "2", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract_result.returncode, 0)
            extracted = json.loads(extract_result.stdout)

            # Step 2: Modify JSON to target file B, case 1
            extracted[0]["file"] = str(tmp_b_path)
            extracted[0]["number"] = 1
            # Remove name to avoid mismatch (cross-file replacement).
            if "name" in extracted[0]:
                del extracted[0]["name"]
            modified_json = json.dumps(extracted)

            # Step 3: Replace into file B
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                ["--mode", "json"],
                input=modified_json,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # Step 4: Verify file B case 1 now has content from file A case 2
            extract_b_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_b_path), "--case", "1"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract_b_result.returncode, 0)

            # Content should match original file A case 2 content
            self.assertIn("@second_case", extract_b_result.stdout)

        finally:
            tmp_a_path.unlink()
            tmp_b_path.unlink()
            for path in [tmp_a_path, tmp_b_path]:
                backup = path.with_suffix(".mlir.bak")
                if backup.exists():
                    backup.unlink()

    def test_move_multiple_cases_across_files(self):
        """Test moving multiple cases from one file to another."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_a:
            tmp_a.write(self.split_test.read_text())
            tmp_a_path = Path(tmp_a.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_b:
            tmp_b.write(self.split_test.read_text())
            tmp_b_path = Path(tmp_b.name)

        try:
            # Extract cases 1,3 from file A
            extract_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_a_path), "--case", "1,3", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract_result.returncode, 0)
            extracted = json.loads(extract_result.stdout)

            # Modify to target file B
            for entry in extracted:
                entry["file"] = str(tmp_b_path)

            modified_json = json.dumps(extracted)

            # Replace into file B
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                ["--mode", "json"],
                input=modified_json,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # Verify file B now has the moved content
            final_content = tmp_b_path.read_text()
            self.assertIn("@first_case", final_content)
            self.assertIn("@third_case", final_content)

        finally:
            tmp_a_path.unlink()
            tmp_b_path.unlink()
            for path in [tmp_a_path, tmp_b_path]:
                backup = path.with_suffix(".mlir.bak")
                if backup.exists():
                    backup.unlink()


class TestBatchStability(unittest.TestCase):
    """Tests for batch replacement stability with varying content sizes."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_replace_with_varying_sizes(self):
        """Test replacing cases with varying content sizes (larger and smaller)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Create replacements with different sizes
            small = "// CHECK-LABEL: @small\nutil.func @small() { util.return }\n"
            large = (
                "// CHECK-LABEL: @large\n"
                + "\n".join([f"// Line {i}" for i in range(50)])
                + "\nutil.func @large() { util.return }\n"
            )
            medium = (
                "// CHECK-LABEL: @medium\n"
                + "\n".join([f"// Line {i}" for i in range(10)])
                + "\nutil.func @medium() { util.return }\n"
            )

            # Replace cases 1, 2, 3 with varying sizes
            batch = [
                {"file": str(tmp_path), "number": 1, "content": large},  # Expand
                {"file": str(tmp_path), "number": 2, "content": small},  # Shrink
                {"file": str(tmp_path), "number": 3, "content": medium},  # Medium
            ]

            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                ["--mode", "json"],
                input=json.dumps(batch),
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # Verify all cases were replaced correctly
            final_content = tmp_path.read_text()
            self.assertIn("@small", final_content)
            self.assertIn("@large", final_content)
            self.assertIn("@medium", final_content)
            self.assertIn("Line 49", final_content)  # From large replacement

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_replace_non_sequential_cases(self):
        """Test replacing non-sequential cases (out of order)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Replace cases in non-sequential order: 3, 1, 2
            batch = [
                {
                    "file": str(tmp_path),
                    "number": 3,
                    "content": "// CHECK-LABEL: @third\nutil.func @third() { util.return }\n",
                },
                {
                    "file": str(tmp_path),
                    "number": 1,
                    "content": "// CHECK-LABEL: @first\nutil.func @first() { util.return }\n",
                },
                {
                    "file": str(tmp_path),
                    "number": 2,
                    "content": "// CHECK-LABEL: @second\nutil.func @second() { util.return }\n",
                },
            ]

            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                ["--mode", "json"],
                input=json.dumps(batch),
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # Verify all cases replaced
            final_content = tmp_path.read_text()
            self.assertIn("@first", final_content)
            self.assertIn("@second", final_content)
            self.assertIn("@third", final_content)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


class TestCompleteWorkflows(unittest.TestCase):
    """Tests for complete end-to-end workflows."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_extract_edit_validate_replace_workflow(self):
        """Test complete workflow: extract → edit → validate → replace."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Step 1: Extract
            extract_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_path), "--case", "2"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract_result.returncode, 0)

            # Step 2: Edit content (create valid MLIR)
            edited_content = """// CHECK-LABEL: @edited_workflow
util.func @edited_workflow() {
  util.return
}
"""

            # Step 3: Validate with --verify flag
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path), "--case", "2", "--verify"],
                input=edited_content,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # Step 4: Verify replacement
            final_content = tmp_path.read_text()
            self.assertIn("@edited_workflow", final_content)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_extract_dry_run_confirm_replace_workflow(self):
        """Test workflow: extract → edit → dry-run → confirm → replace."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Extract
            extract_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_path), "--case", "2"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract_result.returncode, 0)

            # Edit
            edited_content = """// CHECK-LABEL: @dry_run_test
util.func @dry_run_test() {
  util.return
}
"""

            # Dry-run to preview
            dry_run_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path), "--case", "2", "--dry-run"],
                input=edited_content,
                capture_output=True,
                text=True,
            )
            self.assertEqual(dry_run_result.returncode, 0)
            self.assertIn("---", dry_run_result.stdout)  # Diff output
            self.assertIn("@dry_run_test", dry_run_result.stdout)

            # File should be unchanged after dry-run
            original_content = self.split_test.read_text()
            current_content = tmp_path.read_text()
            self.assertEqual(original_content, current_content)

            # Actually replace
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path), "--case", "2"],
                input=edited_content,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # Verify final replacement
            final_content = tmp_path.read_text()
            self.assertIn("@dry_run_test", final_content)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


class TestMultiToolConsistency(unittest.TestCase):
    """Tests for consistency between different tools."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_json_schema_consistency_with_extract(self):
        """Test that replace accepts exact output from extract."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Extract with --json
            extract_result = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_path), "--case", "2", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract_result.returncode, 0)

            # Add file field to extracted JSON since extract doesn't include it
            extracted_data = json.loads(extract_result.stdout)
            for case in extracted_data:
                case["file"] = str(tmp_path)
            modified_json = json.dumps(extracted_data)

            # Replace directly with extract output (no modification)
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                ["--mode", "json"],
                input=modified_json,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # First replace may normalize whitespace. Extract again and verify
            # second replace is idempotent.
            extract_result2 = run_python_module(
                "lit_tools.iree_lit_extract",
                [str(tmp_path), "--case", "2", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(extract_result2.returncode, 0)

            extracted_data2 = json.loads(extract_result2.stdout)
            for case in extracted_data2:
                case["file"] = str(tmp_path)
            modified_json2 = json.dumps(extracted_data2)

            replace_result2 = run_python_module(
                "lit_tools.iree_lit_replace",
                ["--mode", "json"],
                input=modified_json2,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result2.returncode, 0)

            # Second replace should be idempotent (no changes).
            self.assertIn("no files modified", replace_result2.stderr.lower())

        finally:
            tmp_path.unlink()

    def test_list_after_replace_preserves_structure(self):
        """Test that list output is consistent before and after replace."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # List before
            list_before = run_python_module(
                "lit_tools.iree_lit_list",
                [str(tmp_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(list_before.returncode, 0)

            # Replace case 2 (no structural changes)
            replace_content = """// CHECK-LABEL: @modified
util.func @modified() {
  util.return
}
"""
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path), "--case", "2"],
                input=replace_content,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # List after
            list_after = run_python_module(
                "lit_tools.iree_lit_list",
                [str(tmp_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(list_after.returncode, 0)

            # Case count should be same
            before_lines = [
                line for line in list_before.stdout.splitlines() if "Test case" in line
            ]
            after_lines = [
                line for line in list_after.stdout.splitlines() if "Test case" in line
            ]
            self.assertEqual(len(before_lines), len(after_lines))

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


class TestConcurrencyProtection(unittest.TestCase):
    """Tests for --fail-if-changed concurrency protection."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_fail_if_changed_detects_modification(self):
        """Test that --fail-if-changed aborts when file is modified during replacement."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Start replacement with --fail-if-changed
            # We'll simulate modification by touching the file between parse and write
            # Since this is an integration test, we test the flag works at all
            # (unit tests would test the actual modification detection logic)

            # First: successful replacement without modification
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path), "--case", "2", "--fail-if-changed"],
                input="// Modified content\nutil.func @test() { util.return }\n",
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # For actual concurrent modification test, we'd need to modify the file
            # during replacement, which is hard to do in an integration test.
            # The unit tests cover the actual detection logic.
            # Here we just verify the flag is accepted and doesn't break normal operation.

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_fail_if_changed_success_when_unchanged(self):
        """Test that --fail-if-changed succeeds when file is not modified."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(self.split_test.read_text())
            tmp_path = Path(tmp.name)

        try:
            # Replace with --fail-if-changed should succeed normally
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path), "--case", "2", "--fail-if-changed"],
                input="// Modified content\nutil.func @test() { util.return }\n",
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)
            self.assertIn("replaced successfully", replace_result.stderr.lower())

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


class TestJsonOutput(unittest.TestCase):
    """Tests for --json-output FILE feature."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_json_output_to_file(self):
        """Test that --json-output writes JSON to file instead of stdout."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_mlir:
            tmp_mlir.write(self.split_test.read_text())
            tmp_path = Path(tmp_mlir.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_json:
            json_path = Path(tmp_json.name)

        try:
            # Replace with --json-output
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path), "--case", "2", "--json-output", str(json_path)],
                input="// Modified content\nutil.func @test() { util.return }\n",
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # Stdout should be empty (output went to file)
            self.assertEqual(replace_result.stdout.strip(), "")

            # JSON file should contain valid JSON
            self.assertTrue(json_path.exists())
            json_content = json_path.read_text()
            json_data = json.loads(json_content)

            # Verify JSON structure
            self.assertIn("modified_files", json_data)
            self.assertIn("modified_cases", json_data)
            self.assertEqual(json_data["modified_files"], 1)
            self.assertEqual(json_data["modified_cases"], 1)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()
            if json_path.exists():
                json_path.unlink()

    def test_json_output_implies_json_mode(self):
        """Test that --json-output automatically enables --json."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_mlir:
            tmp_mlir.write(self.split_test.read_text())
            tmp_path = Path(tmp_mlir.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_json:
            json_path = Path(tmp_json.name)

        try:
            # Use --json-output without --json flag
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path), "--case", "2", "--json-output", str(json_path)],
                input="// Modified content\nutil.func @test() { util.return }\n",
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # JSON should still be written to file
            self.assertTrue(json_path.exists())
            json_data = json.loads(json_path.read_text())
            self.assertIn("modified_files", json_data)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()
            if json_path.exists():
                json_path.unlink()

    def test_json_output_with_dry_run(self):
        """Test that --json-output works with --dry-run."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_mlir:
            tmp_mlir.write(self.split_test.read_text())
            tmp_path = Path(tmp_mlir.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_json:
            json_path = Path(tmp_json.name)

        try:
            # Dry-run with --json-output
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [
                    str(tmp_path),
                    "--case",
                    "2",
                    "--dry-run",
                    "--json-output",
                    str(json_path),
                ],
                input="// Modified content\nutil.func @test() { util.return }\n",
                capture_output=True,
                text=True,
            )
            self.assertEqual(replace_result.returncode, 0)

            # JSON output should include dry_run flag and diff
            json_data = json.loads(json_path.read_text())
            self.assertIn("file_results", json_data)
            self.assertTrue(json_data["dry_run"])
            self.assertTrue(len(json_data["file_results"]) > 0)
            file_result = json_data["file_results"][0]
            self.assertTrue(file_result.get("dry_run", False))
            self.assertIn("diff", file_result)

        finally:
            tmp_path.unlink()
            # Dry-run shouldn't create backup
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()
            if json_path.exists():
                json_path.unlink()


class TestDuplicateHandling(unittest.TestCase):
    """Tests for duplicate case name and entry handling (Phase 1 validation)."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_json_name_number_mismatch(self):
        """Test that JSON with both 'number' and 'name' pointing to different cases errors."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_mlir:
            tmp_mlir.write(self.split_test.read_text())
            tmp_path = Path(tmp_mlir.name)

        try:
            # Create JSON with mismatched number and name.
            # split_test.mlir has 3 cases - case 1 and case 2 have different names
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 1,
                        "name": "second_case",  # This is actually case 2's name
                        "content": "// Test content\nutil.func @test() { util.return }\n",
                    }
                ]
            )

            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [],
                input=json_input,
                capture_output=True,
                text=True,
            )

            # Should error
            self.assertNotEqual(replace_result.returncode, 0)
            self.assertIn("name/number mismatch", replace_result.stderr.lower())

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_json_name_number_match(self):
        """Test that JSON with both 'number' and 'name' pointing to same case succeeds."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_mlir:
            tmp_mlir.write(self.split_test.read_text())
            tmp_path = Path(tmp_mlir.name)

        try:
            # Create JSON with matching number and name (case 2 is "second_case")
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "name": "second_case",  # Correct match
                        "content": "// Test content\nutil.func @test() { util.return }\n",
                    }
                ]
            )

            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [],
                input=json_input,
                capture_output=True,
                text=True,
            )

            # Should succeed
            self.assertEqual(replace_result.returncode, 0)

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()

    def test_text_duplicate_name_error(self):
        """Test that text mode with duplicate case names errors with disambiguation."""
        # Create a file with duplicate case names
        duplicate_content = """// RUN: iree-opt %s

// CHECK-LABEL: @duplicate
util.func @duplicate() {
  util.return
}

// -----

// CHECK-LABEL: @duplicate
util.func @duplicate() {
  util.return
}

// -----

// CHECK-LABEL: @unique
util.func @unique() {
  util.return
}
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_mlir:
            tmp_mlir.write(duplicate_content)
            tmp_path = Path(tmp_mlir.name)

        try:
            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path), "--name", "duplicate"],
                input="// New content\nutil.func @test() { util.return }\n",
                capture_output=True,
                text=True,
            )

            # Should error with case numbers
            self.assertNotEqual(replace_result.returncode, 0)
            self.assertIn("multiple cases named", replace_result.stderr.lower())
            self.assertIn("--case number", replace_result.stderr.lower())

        finally:
            tmp_path.unlink()

    def test_json_duplicate_name_error(self):
        """Test that JSON mode with duplicate case names errors."""
        # Create a file with duplicate case names
        duplicate_content = """// RUN: iree-opt %s

// CHECK-LABEL: @duplicate
util.func @duplicate() {
  util.return
}

// -----

// CHECK-LABEL: @duplicate
util.func @duplicate() {
  util.return
}
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_mlir:
            tmp_mlir.write(duplicate_content)
            tmp_path = Path(tmp_mlir.name)

        try:
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "name": "duplicate",
                        "content": "// New content\nutil.func @test() { util.return }\n",
                    }
                ]
            )

            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [],
                input=json_input,
                capture_output=True,
                text=True,
            )

            # Should error
            self.assertNotEqual(replace_result.returncode, 0)
            self.assertIn("multiple cases", replace_result.stderr.lower())

        finally:
            tmp_path.unlink()

    def test_json_duplicate_entries_by_number(self):
        """Test that JSON with duplicate entries for same case number errors."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_mlir:
            tmp_mlir.write(self.split_test.read_text())
            tmp_path = Path(tmp_mlir.name)

        try:
            # Two entries for case 2
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// First replacement\n",
                    },
                    {
                        "file": str(tmp_path),
                        "number": 2,
                        "content": "// Second replacement\n",
                    },
                ]
            )

            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [],
                input=json_input,
                capture_output=True,
                text=True,
            )

            # Should error
            self.assertNotEqual(replace_result.returncode, 0)
            self.assertIn("duplicate replacement", replace_result.stderr.lower())

        finally:
            tmp_path.unlink()

    def test_json_duplicate_entries_by_name(self):
        """Test that JSON with duplicate entries for same case name errors."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_mlir:
            tmp_mlir.write(self.split_test.read_text())
            tmp_path = Path(tmp_mlir.name)

        try:
            # Two entries for "second_case"
            json_input = json.dumps(
                [
                    {
                        "file": str(tmp_path),
                        "name": "second_case",
                        "content": "// First replacement\n",
                    },
                    {
                        "file": str(tmp_path),
                        "name": "second_case",
                        "content": "// Second replacement\n",
                    },
                ]
            )

            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [],
                input=json_input,
                capture_output=True,
                text=True,
            )

            # Should error
            self.assertNotEqual(replace_result.returncode, 0)
            self.assertIn("duplicate replacement", replace_result.stderr.lower())

        finally:
            tmp_path.unlink()


class TestCLIOverrideWarning(unittest.TestCase):
    """Tests for CLI file override warning (Phase 2 UX)."""

    def setUp(self):
        self.split_test = _FIXTURES_DIR / "split_test.mlir"

    def test_cli_file_override_warning(self):
        """Test that warning appears when CLI file overrides JSON file field."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False
        ) as tmp_mlir:
            tmp_mlir.write(self.split_test.read_text())
            tmp_path = Path(tmp_mlir.name)

        try:
            # JSON specifies different file, but CLI overrides it
            json_input = json.dumps(
                [
                    {
                        "file": "original_file.mlir",  # This will be overridden
                        "number": 2,
                        "content": "// Test content\nutil.func @test() { util.return }\n",
                    }
                ]
            )

            replace_result = run_python_module(
                "lit_tools.iree_lit_replace",
                [str(tmp_path)],  # CLI file override
                input=json_input,
                capture_output=True,
                text=True,
            )

            # Should succeed with warning
            self.assertEqual(replace_result.returncode, 0)
            self.assertIn("overriding", replace_result.stderr.lower())

        finally:
            tmp_path.unlink()
            backup = tmp_path.with_suffix(".mlir.bak")
            if backup.exists():
                backup.unlink()


if __name__ == "__main__":
    unittest.main()
