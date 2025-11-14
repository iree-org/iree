# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for verification helper functions and auto-detection logic."""

import unittest
from dataclasses import dataclass

from lit_tools.core import verification


# Mock TestCase for testing verification helpers.
@dataclass
class MockTestCase:
    """Mock test case for testing verification logic."""

    number: int
    name: str | None
    content: str


class TestShouldSkipVerificationForCase(unittest.TestCase):
    """Tests for should_skip_verification_for_case() in verification module."""

    def test_detects_expected_error(self):
        """Test detection of expected-error directive."""
        case = MockTestCase(
            number=1,
            name="test_invalid",
            content="""
func.func @test_invalid() {
  %c0 = arith.constant 0.0 : f32
  // expected-error @+1 {{invalid operand type}}
  %bad = arith.addi %c0, %c0 : f32
  return
}
""",
        )
        self.assertTrue(verification.should_skip_verification_for_case(case))

    def test_detects_expected_warning(self):
        """Test detection of expected-warning directive."""
        case = MockTestCase(
            number=1,
            name="test_warning",
            content="""
func.func @test_warning() {
  // expected-warning @+1 {{deprecated API}}
  %result = some.deprecated.op : i32
  return
}
""",
        )
        self.assertTrue(verification.should_skip_verification_for_case(case))

    def test_detects_expected_note(self):
        """Test detection of expected-note directive."""
        case = MockTestCase(
            number=1,
            name="test_note",
            content="""
func.func @test_note() {
  // expected-note @+1 {{see related diagnostic}}
  %x = some.op : i32
  return
}
""",
        )
        self.assertTrue(verification.should_skip_verification_for_case(case))

    def test_valid_ir_no_skip(self):
        """Test that valid IR without expected-error is not skipped."""
        case = MockTestCase(
            number=1,
            name="test_valid",
            content="""
func.func @test_valid() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %sum = arith.addi %c0, %c1 : i32
  return
}
""",
        )
        self.assertFalse(verification.should_skip_verification_for_case(case))

    def test_empty_content_no_skip(self):
        """Test that empty content is not skipped."""
        case = MockTestCase(number=1, name="test_empty", content="")
        self.assertFalse(verification.should_skip_verification_for_case(case))

    def test_comment_with_expected_word_no_skip(self):
        """Test that comments mentioning 'expected-error' in prose are not detected."""
        case = MockTestCase(
            number=1,
            name="test_misleading",
            content="""
func.func @test_misleading() {
  // This function tests behavior when we have an expected-error in documentation
  %x = arith.constant 0 : i32
  return
}
""",
        )
        # The pattern "// expected-error" doesn't match "an expected-error" in prose.
        # This is good - we want to detect directives, not mentions in comments.
        self.assertFalse(verification.should_skip_verification_for_case(case))

    def test_multiple_directives(self):
        """Test detection when multiple diagnostic directives present."""
        case = MockTestCase(
            number=1,
            name="test_multiple",
            content="""
func.func @test_multiple() {
  // expected-warning @+1 {{first issue}}
  %x = some.op : i32
  // expected-error @+1 {{second issue}}
  %y = some.bad.op : f32
  return
}
""",
        )
        self.assertTrue(verification.should_skip_verification_for_case(case))


class TestShouldSkipVerificationForContent(unittest.TestCase):
    """Tests for should_skip_verification_for_content() in verification module."""

    def test_detects_expected_error(self):
        """Test detection of expected-error directive in replacement content."""
        content = """
func.func @test_invalid() {
  %c0 = arith.constant 0.0 : f32
  // expected-error @+1 {{invalid operand type}}
  %bad = arith.addi %c0, %c0 : f32
  return
}
"""
        self.assertTrue(verification.should_skip_verification_for_content(content))

    def test_detects_expected_warning(self):
        """Test detection of expected-warning directive."""
        content = """
func.func @test_warning() {
  // expected-warning @+1 {{deprecated}}
  %x = deprecated.op : i32
  return
}
"""
        self.assertTrue(verification.should_skip_verification_for_content(content))

    def test_detects_expected_note(self):
        """Test detection of expected-note directive."""
        content = """
func.func @test_note() {
  // expected-note @+1 {{see diagnostic}}
  %x = some.op : i32
  return
}
"""
        self.assertTrue(verification.should_skip_verification_for_content(content))

    def test_valid_ir_no_skip(self):
        """Test that valid IR without expected-error is not skipped."""
        content = """
func.func @test_valid() {
  %c0 = arith.constant 0 : i32
  return
}
"""
        self.assertFalse(verification.should_skip_verification_for_content(content))

    def test_empty_content_no_skip(self):
        """Test that empty content is not skipped."""
        self.assertFalse(verification.should_skip_verification_for_content(""))

    def test_whitespace_only_no_skip(self):
        """Test that whitespace-only content is not skipped."""
        self.assertFalse(
            verification.should_skip_verification_for_content("   \n  \n  ")
        )


class TestInvalidIRDetection(unittest.TestCase):
    """Tests that actually invalid IR is detected by verification."""

    def test_type_mismatch_fails_verification(self):
        """Test that type mismatch (float value with int type) fails verification."""
        content = """
func.func @test_invalid() {
  %c0 = arith.constant 0.0 : i32
  return
}
"""
        # This should fail verification (can't use float 0.0 with i32 type).
        valid, error_msg = verification.verify_ir(content)
        self.assertFalse(valid)
        self.assertIn("Verification failed", error_msg)

    def test_undefined_ssa_value_fails_verification(self):
        """Test that undefined SSA value reference fails verification."""
        content = """
func.func @test_undefined() {
  %result = arith.addi %undefined, %undefined : i32
  return
}
"""
        # This should fail verification (undefined SSA values).
        valid, error_msg = verification.verify_ir(content)
        self.assertFalse(valid)
        self.assertIn("Verification failed", error_msg)

    def test_invalid_operation_fails_verification(self):
        """Test that invalid operation structure fails verification."""
        content = """
func.func @test_invalid_op() {
  %c0 = arith.constant 1 : i32
  %c1 = arith.constant 2 : i32
  %bad = arith.addi %c0, %c1 : f32
  return
}
"""
        # This should fail verification (adding i32 values but declaring result as f32).
        valid, error_msg = verification.verify_ir(content)
        self.assertFalse(valid)
        self.assertIn("Verification failed", error_msg)


class TestVerificationHelperConsistency(unittest.TestCase):
    """Test that both helpers behave consistently."""

    def test_same_content_same_result(self):
        """Test that both helpers give same result for same content."""
        content = """
func.func @test() {
  // expected-error @+1 {{error}}
  %x = bad.op : i32
  return
}
"""
        case = MockTestCase(number=1, name="test", content=content)

        extract_result = verification.should_skip_verification_for_case(case)
        replace_result = verification.should_skip_verification_for_content(content)

        self.assertEqual(extract_result, replace_result)

    def test_valid_content_consistency(self):
        """Test both helpers agree on valid content."""
        content = """
func.func @valid() {
  %c0 = arith.constant 0 : i32
  return
}
"""
        case = MockTestCase(number=1, name="valid", content=content)

        extract_result = verification.should_skip_verification_for_case(case)
        replace_result = verification.should_skip_verification_for_content(content)

        self.assertEqual(extract_result, replace_result)
        self.assertFalse(extract_result)


if __name__ == "__main__":
    unittest.main()
