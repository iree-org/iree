# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree-lit-lint tool."""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

# Add tools/utils to path.
sys.path.insert(0, str(Path(__file__).parents[2]))

from common import exit_codes
from lit_tools import iree_lit_lint


class TestRawSSAIdentifierChecker(unittest.TestCase):
    """Test detection of raw SSA identifiers in CHECK lines."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_raw_ssa_identifiers(self):
        """Test that raw SSA identifiers are detected as errors."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "1", "--json", "--quiet"]
        )
        result = iree_lit_lint.main(args)

        # Should exit with error code since errors were found.
        self.assertEqual(result, exit_codes.ERROR)

    def test_json_output_structure(self):
        """Test JSON output contains proper issue structure."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "1", "--json", "--quiet"]
        )

        # Capture stdout.

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        self.assertIn("issues", output)
        self.assertGreater(len(output["issues"]), 0)

        # Find raw_ssa_identifier issue (may not be first due to other checks).
        raw_ssa_issues = [
            i for i in output["issues"] if i["code"] == "raw_ssa_identifier"
        ]
        self.assertGreater(
            len(raw_ssa_issues), 0, "Should find at least one raw_ssa_identifier issue"
        )

        issue = raw_ssa_issues[0]
        self.assertEqual(issue["severity"], "error")
        self.assertEqual(issue["code"], "raw_ssa_identifier")
        self.assertIn("message", issue)
        self.assertIn("line", issue)
        self.assertIn("help", issue)

    def test_errors_only_flag(self):
        """Test --errors-only filters to only errors."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "1", "--errors-only", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # All issues should be errors.
        for issue in output["issues"]:
            self.assertEqual(issue["severity"], "error")

    def test_detects_arg_in_operands(self):
        """Test that %arg0 in operands is flagged."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "40", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())
        raw_ssa_issues = [
            i for i in output["issues"] if i["code"] == "raw_ssa_identifier"
        ]
        # Should find %arg0 and %arg1.
        self.assertGreaterEqual(len(raw_ssa_issues), 2)
        self.assertEqual(result, exit_codes.ERROR)

    def test_detects_semantic_names(self):
        """Test that %buffer, %offset are flagged."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "41", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())
        raw_ssa_issues = [
            i for i in output["issues"] if i["code"] == "raw_ssa_identifier"
        ]
        # Should find %buffer and %offset.
        self.assertGreaterEqual(len(raw_ssa_issues), 2)
        self.assertEqual(result, exit_codes.ERROR)

    def test_detects_in_signatures(self):
        """Test that %arg0 in signatures is flagged."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "42", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())
        raw_ssa_issues = [
            i for i in output["issues"] if i["code"] == "raw_ssa_identifier"
        ]
        # Should find %arg0 in signature.
        self.assertGreaterEqual(len(raw_ssa_issues), 1)
        self.assertEqual(result, exit_codes.ERROR)

    def test_multiple_raw_ssa_per_line(self):
        """Test that multiple raw SSA on one line are all flagged."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "43", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())
        raw_ssa_issues = [
            i for i in output["issues"] if i["code"] == "raw_ssa_identifier"
        ]
        # Should find %a, %b, %c, %d (4 raw SSA values).
        self.assertGreaterEqual(len(raw_ssa_issues), 4)
        self.assertEqual(result, exit_codes.ERROR)

    def test_nolint_suppresses(self):
        """Test that NOLINT on preceding line suppresses rest of case."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "44", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())
        raw_ssa_issues = [
            i for i in output["issues"] if i["code"] == "raw_ssa_identifier"
        ]
        # NOLINT should suppress all raw_ssa_identifier errors.
        self.assertEqual(len(raw_ssa_issues), 0)

    def test_captures_not_flagged(self):
        """Test that %[[NAME]] captures are not flagged."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "45", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())
        raw_ssa_issues = [
            i for i in output["issues"] if i["code"] == "raw_ssa_identifier"
        ]
        # Captures should NOT trigger raw_ssa_identifier.
        self.assertEqual(len(raw_ssa_issues), 0)

    def test_constants_exempted(self):
        """Test that MLIR constants (%c0, %c123, %cst) are exempted."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "46", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())
        raw_ssa_issues = [
            i for i in output["issues"] if i["code"] == "raw_ssa_identifier"
        ]
        # Constants (%c0, %c123_i32, %cst, %cst_0) are exempted.
        self.assertEqual(len(raw_ssa_issues), 0)


class TestExcessiveWildcardChecker(unittest.TestCase):
    """Test detection of excessive wildcards in CHECK lines."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_excessive_wildcards(self):
        """Test that excessive wildcards trigger warnings."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "10", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warnings but not errors.
        self.assertEqual(result, exit_codes.SUCCESS)
        self.assertGreater(output["warnings"], 0)

        # Check for wildcard warning.
        wildcard_issues = [
            i for i in output["issues"] if i["code"] == "excessive_wildcards"
        ]
        self.assertGreater(len(wildcard_issues), 0)


class TestNonSemanticCaptureChecker(unittest.TestCase):
    """Test detection of non-semantic capture names."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_constant_based_names(self):
        """Test detection of %[[C0]] style names."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "11", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warnings for non-semantic names.
        semantic_issues = [
            i for i in output["issues"] if i["code"] == "non_semantic_capture"
        ]
        self.assertGreater(len(semantic_issues), 0)

    def test_provides_suggestions(self):
        """Test that suggestions are provided for better names."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "11", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Check that some issues have suggestions.
        issues_with_suggestions = [i for i in output["issues"] if "suggestions" in i]
        self.assertGreater(len(issues_with_suggestions), 0)


class TestZeroCheckLinesChecker(unittest.TestCase):
    """Test detection of test cases with no CHECK lines."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_zero_checks(self):
        """Test that cases with no CHECKs are flagged."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "2", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should be an error.
        self.assertEqual(result, exit_codes.ERROR)

        # Check for zero_check_lines error.
        zero_check_issues = [
            i for i in output["issues"] if i["code"] == "zero_check_lines"
        ]
        self.assertGreater(len(zero_check_issues), 0)


class TestTodoWithoutExplanationChecker(unittest.TestCase):
    """Test detection of TODO/FIXME without explanation."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_bare_todo(self):
        """Test that bare TODO comments are flagged as errors."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "3", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have errors for bare TODOs.
        self.assertEqual(result, exit_codes.ERROR)

        # Check for TODO errors.
        todo_issues = [
            i for i in output["issues"] if i["code"] == "todo_without_explanation"
        ]
        self.assertGreater(len(todo_issues), 0)

    def test_detects_weak_explanation(self):
        """Test that weak TODO explanations are flagged."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "4", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have errors for weak TODOs.
        self.assertEqual(result, exit_codes.ERROR)

        # Check for TODO errors.
        todo_issues = [
            i for i in output["issues"] if i["code"] == "todo_without_explanation"
        ]
        self.assertGreater(len(todo_issues), 0)

    def test_accepts_good_explanation(self):
        """Test that detailed TODO explanations pass."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "5", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no TODO errors.
        self.assertEqual(result, exit_codes.SUCCESS)
        todo_issues = [
            i for i in output["issues"] if i["code"] == "todo_without_explanation"
        ]
        self.assertEqual(len(todo_issues), 0)


class TestCHECKNOTWithoutAnchorChecker(unittest.TestCase):
    """Test detection of CHECK-NOT without anchors."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_unanchored_check_not(self):
        """Test that CHECK-NOT without any anchors is flagged."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "6", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have errors.
        self.assertEqual(result, exit_codes.ERROR)

        # Check for anchor errors.
        anchor_issues = [
            i for i in output["issues"] if i["code"] == "check_not_without_anchor"
        ]
        self.assertGreater(len(anchor_issues), 0)

    def test_accepts_properly_anchored(self):
        """Test that CHECK-NOT with both anchors passes."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "9", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no errors.
        self.assertEqual(result, exit_codes.SUCCESS)
        anchor_issues = [
            i for i in output["issues"] if i["code"] == "check_not_without_anchor"
        ]
        self.assertEqual(len(anchor_issues), 0)


class TestMatchedCaptureNameChecker(unittest.TestCase):
    """Test detection of mismatched capture names."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_mismatched_names(self):
        """Test that mismatched IR/CHECK names trigger warnings."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "12", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warnings (not errors).
        self.assertEqual(result, exit_codes.SUCCESS)

        # Check for mismatch warnings.
        mismatch_issues = [
            i for i in output["issues"] if i["code"] == "mismatched_capture_name"
        ]
        self.assertGreater(len(mismatch_issues), 0)
        self.assertEqual(mismatch_issues[0]["severity"], "warning")

    def test_accepts_matched_names(self):
        """Test that matched names don't trigger warnings."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "13", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no mismatch warnings.
        mismatch_issues = [
            i for i in output["issues"] if i["code"] == "mismatched_capture_name"
        ]
        self.assertEqual(len(mismatch_issues), 0)


class TestWildcardInTerminatorChecker(unittest.TestCase):
    """Test detection of wildcards in terminators."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_wildcard_terminators(self):
        """Test that wildcards in terminators trigger warnings."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "15", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warnings.
        self.assertEqual(result, exit_codes.SUCCESS)

        # Check for wildcard warnings.
        wildcard_issues = [
            i for i in output["issues"] if i["code"] == "wildcard_in_terminator"
        ]
        self.assertGreater(len(wildcard_issues), 0)
        self.assertEqual(wildcard_issues[0]["severity"], "warning")

    def test_accepts_explicit_terminators(self):
        """Test that explicit terminator operands pass."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "16", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no wildcard warnings.
        wildcard_issues = [
            i for i in output["issues"] if i["code"] == "wildcard_in_terminator"
        ]
        self.assertEqual(len(wildcard_issues), 0)


class TestCHECKWithoutLabelContextChecker(unittest.TestCase):
    """Test detection of CHECK without LABEL context."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_check_before_label(self):
        """Test that CHECK before LABEL triggers warning."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "18", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warnings.
        self.assertEqual(result, exit_codes.SUCCESS)

        # Check for label context warnings.
        label_issues = [
            i for i in output["issues"] if i["code"] == "check_without_label_context"
        ]
        self.assertGreater(len(label_issues), 0)
        self.assertEqual(label_issues[0]["severity"], "warning")

    def test_accepts_proper_label_context(self):
        """Test that CHECK after LABEL passes."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "19", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no label context warnings.
        label_issues = [
            i for i in output["issues"] if i["code"] == "check_without_label_context"
        ]
        self.assertEqual(len(label_issues), 0)


class TestUnusedCaptureChecker(unittest.TestCase):
    """Test detection of unused captures."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_unused_captures(self):
        """Test that unused captures trigger warnings."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "21", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warnings.
        self.assertEqual(result, exit_codes.SUCCESS)

        # Check for unused capture warnings.
        unused_issues = [i for i in output["issues"] if i["code"] == "unused_capture"]
        self.assertGreater(len(unused_issues), 0)
        self.assertEqual(unused_issues[0]["severity"], "warning")

    def test_accepts_used_captures(self):
        """Test that used captures don't trigger warnings."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "22", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no unused capture warnings.
        unused_issues = [i for i in output["issues"] if i["code"] == "unused_capture"]
        self.assertEqual(len(unused_issues), 0)

    def test_accepts_check_same_usage(self):
        """Test that captures used in CHECK-SAME pass."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "23", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no unused capture warnings.
        unused_issues = [i for i in output["issues"] if i["code"] == "unused_capture"]
        self.assertEqual(len(unused_issues), 0)


class TestGoodPractices(unittest.TestCase):
    """Test that good practices don't trigger false positives."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_good_test.mlir"

    def test_no_issues_for_good_test(self):
        """Test that well-written tests pass linting."""
        args = iree_lit_lint.parse_arguments([str(self.fixture), "--json", "--quiet"])

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no errors.
        self.assertEqual(result, exit_codes.SUCCESS)
        self.assertEqual(output["errors"], 0)
        self.assertEqual(len(output["issues"]), 0)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI argument parsing and integration."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_good_test.mlir"

    def test_list_mode(self):
        """Test --list mode shows test cases."""
        args = iree_lit_lint.parse_arguments([str(self.fixture), "--list"])

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = f.getvalue()

        self.assertEqual(result, exit_codes.SUCCESS)
        self.assertIn("good_example", output)

    def test_case_selection(self):
        """Test --case selection works."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "1", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        self.assertEqual(result, exit_codes.SUCCESS)
        self.assertEqual(output["cases_linted"], 1)

    def test_min_severity_filtering(self):
        """Test --min-severity filters correctly."""
        fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

        # Get all issues.
        args_all = iree_lit_lint.parse_arguments(
            [str(fixture), "--case", "10", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args_all)

        output_all = json.loads(f.getvalue())
        total_issues = len(output_all["issues"])

        # Get only errors.
        args_errors = iree_lit_lint.parse_arguments(
            [
                str(fixture),
                "--case",
                "10",
                "--min-severity",
                "error",
                "--json",
                "--quiet",
            ]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            iree_lit_lint.main(args_errors)

        output_errors = json.loads(f.getvalue())
        error_issues = len(output_errors["issues"])

        # Should have fewer issues when filtering to errors only.
        # (This fixture has warnings, not errors)
        self.assertLessEqual(error_issues, total_issues)


class TestCheckLabelFormatChecker(unittest.TestCase):
    """Test detection of CHECK-LABEL format issues."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_accepts_unambiguous_bare_label(self):
        """Test that unambiguous bare labels don't trigger warnings."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "33", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no warnings for unambiguous labels.
        self.assertEqual(result, exit_codes.SUCCESS)
        label_issues = [
            i
            for i in output["issues"]
            if i["code"] in ["bare_function_name_in_label", "ambiguous_label"]
        ]
        self.assertEqual(len(label_issues), 0)

    def test_detects_ambiguous_label(self):
        """Test that ambiguous bare labels trigger warnings with matches shown."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "34", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warning for ambiguous label.
        self.assertEqual(result, exit_codes.SUCCESS)
        ambiguous_issues = [
            i for i in output["issues"] if i["code"] == "ambiguous_label"
        ]
        self.assertGreater(len(ambiguous_issues), 0)

        # Check that issue shows matches.
        issue = ambiguous_issues[0]
        self.assertIn("@dispatch", issue["message"])
        # Check help text shows the matches
        self.assertIn("appears", issue["help"])
        self.assertIn("times", issue["help"])

    def test_detects_label_not_found(self):
        """Test that labels not in IR trigger warnings."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "35", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warning for label not found.
        self.assertEqual(result, exit_codes.SUCCESS)
        not_found_issues = [
            i for i in output["issues"] if i["code"] == "label_not_found_in_ir"
        ]
        self.assertGreater(len(not_found_issues), 0)

        # Check message mentions label not found.
        issue = not_found_issues[0]
        self.assertIn("not found", issue["message"])

    def test_accepts_label_with_operation_prefix(self):
        """Test that labels with operation prefix don't trigger checks."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "36", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no label format warnings.
        self.assertEqual(result, exit_codes.SUCCESS)
        label_issues = [
            i
            for i in output["issues"]
            if i["code"]
            in [
                "bare_function_name_in_label",
                "ambiguous_label",
                "label_not_found_in_ir",
            ]
        ]
        self.assertEqual(len(label_issues), 0)

    def test_detects_ambiguous_global(self):
        """Test that ambiguous global labels trigger warnings."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "37", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warning for ambiguous global.
        self.assertEqual(result, exit_codes.SUCCESS)
        ambiguous_issues = [
            i for i in output["issues"] if i["code"] == "ambiguous_label"
        ]
        self.assertGreater(len(ambiguous_issues), 0)


class TestInvalidWildcardPatterns(unittest.TestCase):
    """Test detection of invalid wildcard patterns in SSA and symbol references."""

    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_detects_invalid_ssa_wildcard(self):
        """Test that %{{.*}} triggers warning."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "38", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warnings (not errors).
        self.assertEqual(result, exit_codes.SUCCESS)

        # Check for invalid SSA zero-or-more pattern warnings.
        ssa_issues = [
            i for i in output["issues"] if i["code"] == "invalid_ssa_zero_or_more"
        ]
        self.assertGreater(len(ssa_issues), 0)
        self.assertEqual(ssa_issues[0]["severity"], "warning")

        # Verify suggestion.
        self.assertIn("suggestions", ssa_issues[0])
        self.assertIn("%{{.+}}", ssa_issues[0]["suggestions"])

    def test_detects_invalid_symbol_wildcard(self):
        """Test that @{{.*}} triggers warning."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--case", "39", "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warnings (not errors).
        self.assertEqual(result, exit_codes.SUCCESS)

        # Check for invalid symbol zero-or-more pattern warnings.
        symbol_issues = [
            i for i in output["issues"] if i["code"] == "invalid_symbol_zero_or_more"
        ]
        self.assertGreater(len(symbol_issues), 0)
        self.assertEqual(symbol_issues[0]["severity"], "warning")

        # Verify suggestion.
        self.assertIn("suggestions", symbol_issues[0])
        self.assertIn("@{{.+}}", symbol_issues[0]["suggestions"])


class TestParseArguments(unittest.TestCase):
    """Test argument parsing."""

    def test_parse_minimal_args(self):
        """Test parsing minimal required arguments."""
        args = iree_lit_lint.parse_arguments(["test.mlir"])
        self.assertEqual(args.test_file, Path("test.mlir"))
        self.assertFalse(args.json)
        self.assertFalse(args.errors_only)
        self.assertEqual(args.min_severity, "info")

    def test_parse_all_options(self):
        """Test parsing all options."""
        args = iree_lit_lint.parse_arguments(
            [
                "test.mlir",
                "--case",
                "1",
                "--errors-only",
                "--json",
                "--quiet",
            ]
        )
        self.assertEqual(args.test_file, Path("test.mlir"))
        self.assertEqual(args.case, ["1"])
        self.assertTrue(args.errors_only)
        self.assertTrue(args.json)
        self.assertTrue(args.quiet)


class TestGroupedOutput(unittest.TestCase):
    """Test grouped output format."""

    def setUp(self):
        """Set up test fixtures."""
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"
        self.good_fixture = Path(__file__).parent / "fixtures" / "lint_good_test.mlir"

    def test_grouped_output_format(self):
        """Test that default output groups errors by code."""
        args = iree_lit_lint.parse_arguments([str(self.good_fixture), "--quiet"])

        result = iree_lit_lint.main(args)

        # Should succeed (no errors in good test file).
        self.assertEqual(result, exit_codes.SUCCESS)

    def test_grouped_output_shows_occurrences(self):
        """Test that grouped output shows occurrence count."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--errors-only", "--case", "1"]
        )

        # Capture stdout to check format.
        output = io.StringIO()
        with redirect_stdout(output):
            iree_lit_lint.main(args)

        output_text = output.getvalue()

        # Should have [ERROR] headers with occurrence counts.
        self.assertIn("[ERROR]", output_text)
        self.assertIn("occurrence", output_text)

        # Should have file:line:severity:code:message format for each occurrence.
        self.assertRegex(
            output_text,
            r"lint_test\.mlir:\d+:?\d*: error: \w+: .+",
            "Expected IDE-parseable format in output",
        )

    def test_grouped_output_deduplicates_help(self):
        """Test that help text appears once per error type."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--errors-only", "--case", "1"]
        )

        output = io.StringIO()
        with redirect_stdout(output):
            iree_lit_lint.main(args)

        output_text = output.getvalue()

        # raw_ssa_identifier appears twice in case 1 (lines 6 and 8).
        # Help text should appear once, not twice.
        help_text = "Use %[[NAME:.+]] instead of %0"
        occurrences = output_text.count(help_text)
        self.assertEqual(
            occurrences,
            1,
            f"Expected help text to appear once, found {occurrences} times",
        )

        # But should have two file:line occurrences for raw_ssa_identifier.
        raw_ssa_lines = [
            line
            for line in output_text.split("\n")
            if "raw_ssa_identifier" in line and "lint_test.mlir:" in line
        ]
        self.assertEqual(len(raw_ssa_lines), 2)


class TestIndividualErrorsFlag(unittest.TestCase):
    """Test --individual-errors flag for ungrouped output."""

    def setUp(self):
        """Set up test fixtures."""
        self.fixture = Path(__file__).parent / "fixtures" / "lint_test.mlir"

    def test_individual_errors_flag(self):
        """Test that --individual-errors shows old format."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--errors-only", "--case", "1", "--individual-errors"]
        )

        output = io.StringIO()
        with redirect_stdout(output):
            iree_lit_lint.main(args)

        output_text = output.getvalue()

        # Should NOT have grouped format headers.
        self.assertNotIn("[ERROR]", output_text)
        self.assertNotIn("occurrences", output_text)

        # Should have snippet lines (old format).
        self.assertIn("  help:", output_text)

        # Should still have file:line:severity format.
        self.assertRegex(output_text, r"lint_test\.mlir:\d+:?\d*: error:")

    def test_individual_errors_duplicates_help(self):
        """Test that --individual-errors shows help for each occurrence."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture), "--errors-only", "--case", "1", "--individual-errors"]
        )

        output = io.StringIO()
        with redirect_stdout(output):
            iree_lit_lint.main(args)

        output_text = output.getvalue()

        # raw_ssa_identifier appears twice - should have two separate help blocks.
        # Count occurrences of "  help:" to verify help appears for each error.
        help_lines = output_text.count("  help:")
        self.assertEqual(
            help_lines,
            2,
            f"Expected help to appear twice (once per error), found {help_lines}",
        )


class TestSplitBoundaryChecks(unittest.TestCase):
    """Test detection of split boundary issues."""

    def setUp(self):
        self.fixture_missing = (
            Path(__file__).parent / "fixtures" / "lint_split_missing.mlir"
        )
        self.fixture_boundary = (
            Path(__file__).parent / "fixtures" / "lint_split_boundary.mlir"
        )
        self.fixture_good = Path(__file__).parent / "fixtures" / "lint_split_good.mlir"

    def test_detects_missing_split_input_file(self):
        """Test that delimiters without --split-input-file triggers warning."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture_missing), "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warning (not error).
        self.assertEqual(result, exit_codes.SUCCESS)

        # Check for missing split input file warning.
        missing_issues = [
            i for i in output["issues"] if i["code"] == "missing_split_input_file"
        ]
        self.assertGreater(len(missing_issues), 0)
        self.assertEqual(missing_issues[0]["severity"], "warning")
        self.assertIn("--split-input-file", missing_issues[0]["help"])

    def test_detects_first_check_not_label(self):
        """Test that first CHECK after split not being CHECK-LABEL triggers warning."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture_boundary), "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warnings (not errors).
        self.assertEqual(result, exit_codes.SUCCESS)

        # Check for first check not label warnings.
        not_label_issues = [
            i
            for i in output["issues"]
            if i["code"] == "first_check_not_label_after_split"
        ]
        # Should have 3: case 2 (CHECK-DAG), case 4 (CHECK-SAME), case 5 (CHECK-NEXT).
        self.assertGreaterEqual(len(not_label_issues), 3)
        self.assertEqual(not_label_issues[0]["severity"], "warning")

    def test_detects_unanchored_check_dag(self):
        """Test that CHECK-DAG before CHECK-LABEL after split triggers warning."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture_boundary), "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have warnings (not errors).
        self.assertEqual(result, exit_codes.SUCCESS)

        # Check for unanchored CHECK-DAG warnings.
        dag_issues = [
            i
            for i in output["issues"]
            if i["code"] == "unanchored_check_dag_after_split"
        ]
        # Should have 1: case 2 (CHECK-DAG).
        self.assertGreaterEqual(len(dag_issues), 1)
        self.assertEqual(dag_issues[0]["severity"], "warning")
        self.assertIn("CHECK-LABEL", dag_issues[0]["help"])

    def test_accepts_proper_label_patterns(self):
        """Test that CHECK-LABEL first after split doesn't trigger warnings."""
        args = iree_lit_lint.parse_arguments(
            [str(self.fixture_good), "--json", "--quiet"]
        )

        f = io.StringIO()
        with redirect_stdout(f):
            result = iree_lit_lint.main(args)

        output = json.loads(f.getvalue())

        # Should have no split boundary warnings.
        split_issues = [
            i
            for i in output["issues"]
            if i["code"]
            in [
                "first_check_not_label_after_split",
                "unanchored_check_dag_after_split",
                "missing_split_input_file",
            ]
        ]
        self.assertEqual(len(split_issues), 0)
        self.assertEqual(result, exit_codes.SUCCESS)


if __name__ == "__main__":
    unittest.main()
