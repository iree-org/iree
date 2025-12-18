# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for CI triage output formatters."""

import json
import sys
import unittest
from pathlib import Path

# Add project tools/utils to path for imports.
sys.path.insert(0, str(Path(__file__).parents[2]))

from ci.core import extractors, patterns


class TestMarkdownFormatter(unittest.TestCase):
    """Tests for markdown output formatting."""

    def test_format_single_actionable_issue(self):
        """Test markdown output for single actionable root cause."""
        # Create synthetic triage result.
        rule = patterns.RootCauseRule(
            name="filecheck_failure",
            primary_pattern="filecheck_failed",
            secondary_patterns=[],
            description="FileCheck pattern not found",
            priority=80,
            actionable=True,
            category="test_failure",
        )
        match = patterns.PatternMatch(
            pattern_name="filecheck_failed",
            match_text="CHECK: expected string not found",
            line_number=42,
            context_before=["func.func @test() {"],
            context_after=["  return"],
            extracted_fields={"file_path": ["test.mlir:42:11"]},
        )
        root_cause = patterns.RootCause(
            rule=rule, primary_matches=[match], secondary_matches=[]
        )
        result = extractors.TriageResult(
            run_id="12345",
            job_id="67890",
            job_name="linux_x64_test",
            root_causes=[root_cause],
        )

        formatter = extractors.MarkdownFormatter()
        output = formatter.format(result)

        # Verify key sections present.
        self.assertIn("# CI Failure Triage Report", output)
        self.assertIn("filecheck_failure", output)
        self.assertIn("Line 42", output)
        self.assertIn("test.mlir:42:11", output)
        self.assertIn("1 actionable", output)

    def test_format_infrastructure_issue(self):
        """Test markdown output for non-actionable infrastructure issue."""
        rule = patterns.RootCauseRule(
            name="rocm_cleanup_crash",
            primary_pattern="rocclr_memobj",
            secondary_patterns=["aborted"],
            description="ROCm cleanup crash",
            priority=100,
            actionable=False,
            category="infrastructure",
        )
        match = patterns.PatternMatch(
            pattern_name="rocclr_memobj",
            match_text="rocclr/device/device.cpp:2891",
            line_number=15,
            context_before=[],
            context_after=[],
        )
        root_cause = patterns.RootCause(
            rule=rule, primary_matches=[match], secondary_matches=[]
        )
        result = extractors.TriageResult(
            run_id="12345",
            job_id="67890",
            job_name="amd_hip_test",
            root_causes=[root_cause],
        )

        formatter = extractors.MarkdownFormatter()
        output = formatter.format(result)

        self.assertIn("Infrastructure Issues (Non-Actionable)", output)
        self.assertIn("rocm_cleanup_crash", output)
        self.assertIn("infrastructure", output)

    def test_format_no_context_mode(self):
        """Test markdown output with context disabled."""
        rule = patterns.RootCauseRule(
            name="test_failure",
            primary_pattern="lit_test_failed",
            secondary_patterns=[],
            description="Test failed",
            priority=75,
            actionable=True,
            category="test",
        )
        match = patterns.PatternMatch(
            pattern_name="lit_test_failed",
            match_text="TEST 'test.mlir' FAILED",
            line_number=10,
            context_before=["line before"],
            context_after=["line after"],
        )
        root_cause = patterns.RootCause(
            rule=rule, primary_matches=[match], secondary_matches=[]
        )
        result = extractors.TriageResult(
            run_id="12345",
            job_id="67890",
            job_name="test_job",
            root_causes=[root_cause],
        )

        formatter = extractors.MarkdownFormatter()
        output_with_context = formatter.format(result, include_context=True)
        output_no_context = formatter.format(result, include_context=False)

        # With context should include error context section.
        self.assertIn("Error Context", output_with_context)
        self.assertIn("line before", output_with_context)
        self.assertIn("line after", output_with_context)

        # Without context should not.
        self.assertNotIn("Error Context", output_no_context)
        self.assertNotIn("line before", output_no_context)


class TestJSONFormatter(unittest.TestCase):
    """Tests for JSON output formatting and LLM usability."""

    def test_json_structure_valid(self):
        """Test that JSON output is valid and parseable."""
        rule = patterns.RootCauseRule(
            name="compile_error",
            primary_pattern="compile_error",
            secondary_patterns=[],
            description="Compilation failed",
            priority=80,
            actionable=True,
            category="code",
        )
        match = patterns.PatternMatch(
            pattern_name="compile_error",
            match_text="error: use of undeclared identifier 'foo'",
            line_number=42,
            context_before=["int main() {"],
            context_after=["  return 0;"],
            extracted_fields={
                "file_path": ["test.c:42:10"],
                "error_message": ["use of undeclared identifier 'foo'"],
            },
        )
        root_cause = patterns.RootCause(
            rule=rule, primary_matches=[match], secondary_matches=[]
        )
        result = extractors.TriageResult(
            run_id="12345",
            job_id="67890",
            job_name="build_test",
            root_causes=[root_cause],
        )

        # Convert to JSON.
        json_output = result.to_dict()

        # Verify it's valid JSON by serializing and parsing.
        json_str = json.dumps(json_output)
        parsed = json.loads(json_str)

        self.assertIsInstance(parsed, dict)
        self.assertIn("run_id", parsed)
        self.assertIn("job_id", parsed)
        self.assertIn("root_causes", parsed)

    def test_json_required_fields(self):
        """Test that all required fields are present for LLM consumption."""
        rule = patterns.RootCauseRule(
            name="test_error",
            primary_pattern="test_pattern",
            secondary_patterns=[],
            description="Test description",
            priority=50,
            actionable=True,
            category="test",
        )
        match = patterns.PatternMatch(
            pattern_name="test_pattern",
            match_text="test match",
            line_number=1,
            context_before=[],
            context_after=[],
        )
        root_cause = patterns.RootCause(
            rule=rule, primary_matches=[match], secondary_matches=[]
        )
        result = extractors.TriageResult(
            run_id="111",
            job_id="222",
            job_name="test_job",
            root_causes=[root_cause],
        )

        json_output = result.to_dict()

        # Verify top-level fields.
        self.assertEqual(json_output["run_id"], "111")
        self.assertEqual(json_output["job_id"], "222")
        self.assertEqual(json_output["job_name"], "test_job")
        self.assertIsInstance(json_output["root_causes"], list)
        self.assertEqual(len(json_output["root_causes"]), 1)

        # Verify root cause fields.
        rc = json_output["root_causes"][0]
        self.assertEqual(rc["name"], "test_error")
        self.assertEqual(rc["category"], "test")
        self.assertEqual(rc["priority"], 50)
        self.assertTrue(rc["actionable"])
        self.assertEqual(rc["description"], "Test description")
        self.assertEqual(rc["primary_pattern"], "test_pattern")
        self.assertIsInstance(rc["secondary_patterns"], list)

        # Verify match fields.
        self.assertIsInstance(rc["matches"], list)
        self.assertEqual(len(rc["matches"]), 1)
        m = rc["matches"][0]
        self.assertEqual(m["pattern_name"], "test_pattern")
        self.assertEqual(m["match_text"], "test match")
        self.assertEqual(m["line_number"], 1)
        self.assertIsInstance(m["context_before"], list)
        self.assertIsInstance(m["context_after"], list)
        self.assertIsInstance(m["extracted_fields"], dict)

    def test_json_extracted_fields_format(self):
        """Test that extracted fields are in LLM-friendly format."""
        rule = patterns.RootCauseRule(
            name="error_with_fields",
            primary_pattern="error_pattern",
            secondary_patterns=[],
            description="Error with extracted fields",
            priority=75,
            actionable=True,
            category="test",
        )
        match = patterns.PatternMatch(
            pattern_name="error_pattern",
            match_text="error at line 42",
            line_number=10,
            context_before=[],
            context_after=[],
            extracted_fields={
                "file_path": ["src/test.cpp:42:5"],
                "error_message": ["undefined reference to 'foo'"],
                "symbol": ["foo"],
            },
        )
        root_cause = patterns.RootCause(
            rule=rule, primary_matches=[match], secondary_matches=[]
        )
        result = extractors.TriageResult(
            run_id="123",
            job_id="456",
            job_name="test",
            root_causes=[root_cause],
        )

        json_output = result.to_dict()
        fields = json_output["root_causes"][0]["matches"][0]["extracted_fields"]

        # Verify extracted fields are accessible.
        self.assertIn("file_path", fields)
        self.assertIn("error_message", fields)
        self.assertIn("symbol", fields)

        # Verify they're lists (for multiple captures).
        self.assertIsInstance(fields["file_path"], list)
        self.assertEqual(fields["file_path"][0], "src/test.cpp:42:5")

    def test_json_serialization_complete(self):
        """Test full JSON serialization with multiple root causes."""
        # Create two root causes.
        rule1 = patterns.RootCauseRule(
            name="error1",
            primary_pattern="p1",
            secondary_patterns=[],
            description="First error",
            priority=80,
            actionable=True,
            category="code",
        )
        rule2 = patterns.RootCauseRule(
            name="error2",
            primary_pattern="p2",
            secondary_patterns=["p3"],
            description="Second error",
            priority=70,
            actionable=False,
            category="infra",
        )

        match1 = patterns.PatternMatch(
            pattern_name="p1",
            match_text="match1",
            line_number=1,
            context_before=[],
            context_after=[],
        )
        match2 = patterns.PatternMatch(
            pattern_name="p2",
            match_text="match2",
            line_number=2,
            context_before=[],
            context_after=[],
        )
        match3 = patterns.PatternMatch(
            pattern_name="p3",
            match_text="match3",
            line_number=3,
            context_before=[],
            context_after=[],
        )

        rc1 = patterns.RootCause(
            rule=rule1, primary_matches=[match1], secondary_matches=[]
        )
        rc2 = patterns.RootCause(
            rule=rule2, primary_matches=[match2], secondary_matches=[match3]
        )

        result = extractors.TriageResult(
            run_id="999",
            job_id="888",
            job_name="multi_error_job",
            root_causes=[rc1, rc2],
        )

        # Serialize to JSON string.
        json_str = json.dumps(result.to_dict(), indent=2)

        # Verify it parses back.
        parsed = json.loads(json_str)

        self.assertEqual(len(parsed["root_causes"]), 2)
        self.assertEqual(parsed["root_causes"][0]["name"], "error1")
        self.assertEqual(parsed["root_causes"][1]["name"], "error2")

        # Verify secondary matches included.
        self.assertEqual(len(parsed["root_causes"][1]["matches"]), 2)


class TestChecklistFormatter(unittest.TestCase):
    """Tests for checklist output formatting."""

    def test_checklist_format_actionable_only(self):
        """Test checklist only includes actionable items."""
        # Create one actionable and one non-actionable.
        actionable_rule = patterns.RootCauseRule(
            name="fix_this",
            primary_pattern="error",
            secondary_patterns=[],
            description="Needs fixing",
            priority=80,
            actionable=True,
            category="code",
        )
        infrastructure_rule = patterns.RootCauseRule(
            name="infra_issue",
            primary_pattern="flake",
            secondary_patterns=[],
            description="Infrastructure flake",
            priority=50,
            actionable=False,
            category="infrastructure",
        )

        match1 = patterns.PatternMatch(
            pattern_name="error",
            match_text="error: bad code",
            line_number=10,
            context_before=[],
            context_after=[],
            extracted_fields={"file_path": ["test.c:10:5"]},
        )
        match2 = patterns.PatternMatch(
            pattern_name="flake",
            match_text="random failure",
            line_number=20,
            context_before=[],
            context_after=[],
        )

        rc1 = patterns.RootCause(
            rule=actionable_rule, primary_matches=[match1], secondary_matches=[]
        )
        rc2 = patterns.RootCause(
            rule=infrastructure_rule, primary_matches=[match2], secondary_matches=[]
        )

        result = extractors.TriageResult(
            run_id="123",
            job_id="456",
            job_name="test",
            root_causes=[rc1, rc2],
        )

        formatter = extractors.ChecklistFormatter()
        output = formatter.format(result)

        # Should only include actionable item.
        self.assertIn("RC-1: fix_this", output)
        self.assertNotIn("infra_issue", output)

    def test_checklist_format_with_extracted_fields(self):
        """Test checklist includes fix hints from extracted fields."""
        rule = patterns.RootCauseRule(
            name="compile_failure",
            primary_pattern="compile_error",
            secondary_patterns=[],
            description="Compilation failed",
            priority=80,
            actionable=True,
            category="code",
        )

        match = patterns.PatternMatch(
            pattern_name="compile_error",
            match_text="error: undefined reference",
            line_number=42,
            context_before=[],
            context_after=[],
            extracted_fields={
                "file_path": ["build.c:42:10"],
                "error_message": ["undefined reference to 'foo'"],
            },
        )

        rc = patterns.RootCause(
            rule=rule, primary_matches=[match], secondary_matches=[]
        )

        result = extractors.TriageResult(
            run_id="123",
            job_id="456",
            job_name="build",
            root_causes=[rc],
        )

        formatter = extractors.ChecklistFormatter()
        output = formatter.format(result)

        # Should include file and error in fix hint.
        self.assertIn("File: build.c:42:10", output)
        self.assertIn("Error: undefined reference to 'foo'", output)


if __name__ == "__main__":
    unittest.main()
