# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for the core document model (Line, Span, FileDoc, CheckLabelExtractor).

These tests verify the foundational data structures for line-oriented
lit test file parsing.
"""

import unittest
from pathlib import Path

from lit_tools.core.document import (
    CheckLabelExtractor,
    FileDoc,
    Line,
    Span,
)


class TestLine(unittest.TestCase):
    """Tests for Line dataclass."""

    def test_line_creation_with_lf(self):
        """Test Line creation with LF newline."""
        line = Line(text="hello world", newline="\n", index=0, source_line=1)
        self.assertEqual(line.text, "hello world")
        self.assertEqual(line.newline, "\n")
        self.assertEqual(line.index, 0)
        self.assertEqual(line.source_line, 1)
        self.assertEqual(line.tags, frozenset())

    def test_line_creation_with_crlf(self):
        """Test Line creation with CRLF newline."""
        line = Line(text="test", newline="\r\n", index=5, source_line=6)
        self.assertEqual(line.newline, "\r\n")
        self.assertEqual(line.get_full_line(), "test\r\n")

    def test_line_creation_without_newline(self):
        """Test Line creation for EOF without trailing newline."""
        line = Line(text="last line", newline="", index=10, source_line=11)
        self.assertEqual(line.newline, "")
        self.assertEqual(line.get_full_line(), "last line")

    def test_line_with_tags(self):
        """Test Line creation with tags."""
        line = Line(
            text="// RUN: test",
            newline="\n",
            index=0,
            source_line=1,
            tags=frozenset(["RUN_HEADER"]),
        )
        self.assertIn("RUN_HEADER", line.tags)

    def test_line_with_tags_immutable_update(self):
        """Test immutable tag update via with_tags()."""
        line1 = Line(text="test", newline="\n", index=0, source_line=1)
        line2 = line1.with_tags(frozenset(["CHECK"]))

        self.assertEqual(line1.tags, frozenset())
        self.assertEqual(line2.tags, frozenset(["CHECK"]))
        self.assertEqual(line1.text, line2.text)  # Other fields unchanged

    def test_line_get_full_line(self):
        """Test get_full_line() returns text + newline."""
        line1 = Line(text="hello", newline="\n", index=0, source_line=1)
        self.assertEqual(line1.get_full_line(), "hello\n")

        line2 = Line(text="world", newline="", index=1, source_line=2)
        self.assertEqual(line2.get_full_line(), "world")

    def test_line_invalid_newline_raises(self):
        """Test that invalid newline character raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Line(text="test", newline="\r", index=0, source_line=1)
        self.assertIn("Invalid newline character", str(cm.exception))

    def test_line_negative_index_raises(self):
        """Test that negative index raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Line(text="test", newline="\n", index=-1, source_line=1)
        self.assertIn("index must be non-negative", str(cm.exception))

    def test_line_invalid_source_line_raises(self):
        """Test that source_line < 1 raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Line(text="test", newline="\n", index=0, source_line=0)
        self.assertIn("Source line must be >= 1", str(cm.exception))


class TestSpan(unittest.TestCase):
    """Tests for Span dataclass."""

    def test_span_creation_valid(self):
        """Test valid Span creation."""
        span = Span(start=5, end=10)
        self.assertEqual(span.start, 5)
        self.assertEqual(span.end, 10)

    def test_span_length(self):
        """Test Span.length property."""
        span = Span(start=5, end=10)
        self.assertEqual(span.length, 5)

    def test_span_zero_length(self):
        """Test zero-length Span (empty case)."""
        span = Span(start=5, end=5)
        self.assertEqual(span.length, 0)
        self.assertTrue(span.is_empty)

    def test_span_is_empty(self):
        """Test Span.is_empty property."""
        self.assertTrue(Span(start=0, end=0).is_empty)
        self.assertFalse(Span(start=0, end=1).is_empty)

    def test_span_contains(self):
        """Test Span.contains() method."""
        span = Span(start=5, end=10)
        self.assertTrue(span.contains(5))
        self.assertTrue(span.contains(7))
        self.assertTrue(span.contains(9))
        self.assertFalse(span.contains(4))
        self.assertFalse(span.contains(10))  # Exclusive end

    def test_span_invalid_negative_start(self):
        """Test that negative start raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Span(start=-1, end=5)
        self.assertIn("start must be non-negative", str(cm.exception))

    def test_span_invalid_end_before_start(self):
        """Test that end < start raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            Span(start=10, end=5)
        self.assertIn("end must be >= start", str(cm.exception))


class TestFileDoc(unittest.TestCase):
    """Tests for FileDoc class."""

    def test_from_text_simple_lf(self):
        """Test parsing simple text with LF newlines."""
        text = "line1\nline2\nline3\n"
        doc = FileDoc.from_text(text)

        self.assertEqual(len(doc.lines), 3)
        self.assertEqual(doc.lines[0].text, "line1")
        self.assertEqual(doc.lines[0].newline, "\n")
        self.assertEqual(doc.lines[0].index, 0)
        self.assertEqual(doc.lines[0].source_line, 1)

        self.assertEqual(doc.lines[2].text, "line3")
        self.assertEqual(doc.lines[2].source_line, 3)

    def test_from_text_no_trailing_newline(self):
        """Test parsing file without trailing newline."""
        text = "line1\nline2"
        doc = FileDoc.from_text(text)

        self.assertEqual(len(doc.lines), 2)
        self.assertEqual(doc.lines[1].text, "line2")
        self.assertEqual(doc.lines[1].newline, "")

    def test_from_text_crlf_newlines(self):
        """Test parsing file with CRLF newlines."""
        text = "line1\r\nline2\r\n"
        doc = FileDoc.from_text(text)

        self.assertEqual(len(doc.lines), 2)
        self.assertEqual(doc.lines[0].newline, "\r\n")
        self.assertEqual(doc.lines[1].newline, "\r\n")

    def test_from_text_mixed_newlines(self):
        """Test parsing file with mixed LF and CRLF."""
        text = "line1\nline2\r\nline3\n"
        doc = FileDoc.from_text(text)

        self.assertEqual(len(doc.lines), 3)
        self.assertEqual(doc.lines[0].newline, "\n")
        self.assertEqual(doc.lines[1].newline, "\r\n")
        self.assertEqual(doc.lines[2].newline, "\n")

    def test_from_text_empty_string(self):
        """Test parsing empty string."""
        doc = FileDoc.from_text("")
        self.assertEqual(len(doc.lines), 0)

    def test_from_text_single_line_no_newline(self):
        """Test parsing single line without newline."""
        doc = FileDoc.from_text("single line")
        self.assertEqual(len(doc.lines), 1)
        self.assertEqual(doc.lines[0].text, "single line")
        self.assertEqual(doc.lines[0].newline, "")

    def test_from_text_with_path(self):
        """Test that path is stored in FileDoc."""
        doc = FileDoc.from_text("test", path=Path("/tmp/test.mlir"))
        self.assertEqual(doc.path, Path("/tmp/test.mlir"))

    def test_to_text_preserves_content(self):
        """Test that to_text() preserves exact content."""
        original = "line1\nline2\r\nline3"
        doc = FileDoc.from_text(original)
        reconstructed = doc.to_text()
        self.assertEqual(original, reconstructed)

    def test_to_text_roundtrip_with_trailing_newline(self):
        """Test roundtrip with trailing newline."""
        original = "// RUN: test\n// CHECK: foo\nfunc @test() { }\n"
        doc = FileDoc.from_text(original)
        self.assertEqual(doc.to_text(), original)

    def test_to_text_roundtrip_without_trailing_newline(self):
        """Test roundtrip without trailing newline."""
        original = "// RUN: test\n// CHECK: foo\nfunc @test() { }"
        doc = FileDoc.from_text(original)
        self.assertEqual(doc.to_text(), original)

    def test_slice_valid_range(self):
        """Test slicing valid range."""
        doc = FileDoc.from_text("line1\nline2\nline3\nline4\n")
        span = Span(start=1, end=3)
        sliced = doc.slice(span)

        self.assertEqual(len(sliced), 2)
        self.assertEqual(sliced[0].text, "line2")
        self.assertEqual(sliced[1].text, "line3")

    def test_slice_entire_document(self):
        """Test slicing entire document."""
        doc = FileDoc.from_text("line1\nline2\n")
        span = Span(start=0, end=2)
        sliced = doc.slice(span)

        self.assertEqual(len(sliced), 2)
        self.assertEqual(sliced[0].text, "line1")
        self.assertEqual(sliced[1].text, "line2")

    def test_slice_empty_span(self):
        """Test slicing with empty span."""
        doc = FileDoc.from_text("line1\nline2\n")
        span = Span(start=1, end=1)
        sliced = doc.slice(span)

        self.assertEqual(len(sliced), 0)

    def test_slice_out_of_bounds_raises(self):
        """Test that slicing beyond document raises IndexError."""
        doc = FileDoc.from_text("line1\nline2\n")
        span = Span(start=0, end=10)

        with self.assertRaises(IndexError) as cm:
            doc.slice(span)
        self.assertIn("extends beyond document", str(cm.exception))

    def test_with_line_tags_updates_single_line(self):
        """Test updating tags for a single line."""
        doc = FileDoc.from_text("line1\nline2\n")
        new_doc = doc.with_line_tags(0, frozenset(["RUN_HEADER"]))

        # Original unchanged
        self.assertEqual(doc.lines[0].tags, frozenset())

        # New doc has updated tags
        self.assertIn("RUN_HEADER", new_doc.lines[0].tags)
        self.assertEqual(new_doc.lines[1].tags, frozenset())

    def test_with_line_tags_invalid_index_raises(self):
        """Test that invalid line index raises IndexError."""
        doc = FileDoc.from_text("line1\n")
        with self.assertRaises(IndexError):
            doc.with_line_tags(10, frozenset(["TEST"]))


class TestCheckLabelExtractor(unittest.TestCase):
    """Tests for CheckLabelExtractor utility."""

    def test_extract_simple_check_label(self):
        """Test extracting simple CHECK-LABEL."""
        lines = [
            Line("// CHECK-LABEL: @foo", "\n", 0, 1),
            Line("func @foo() { }", "\n", 1, 2),
        ]
        names = CheckLabelExtractor.extract_all_names(lines)
        self.assertEqual(names, ["foo"])

    def test_extract_check_label_with_prefix(self):
        """Test extracting CHECK-LABEL with operation prefix."""
        lines = [
            Line("// CHECK-LABEL: util.func @bar", "\n", 0, 1),
            Line("util.func @bar() { }", "\n", 1, 2),
        ]
        names = CheckLabelExtractor.extract_all_names(lines)
        self.assertEqual(names, ["bar"])

    def test_extract_multiple_check_labels(self):
        """Test extracting multiple CHECK-LABELs in one case."""
        lines = [
            Line("// CHECK-LABEL: @first", "\n", 0, 1),
            Line("// FOO-LABEL: @first", "\n", 1, 2),
            Line("// BAR-LABEL: @second", "\n", 2, 3),
        ]
        names = CheckLabelExtractor.extract_all_names(lines)
        self.assertEqual(names, ["first", "second"])

    def test_extract_no_labels(self):
        """Test extraction when no CHECK-LABELs present."""
        lines = [
            Line("// CHECK: constant", "\n", 0, 1),
            Line("func @test() { }", "\n", 1, 2),
        ]
        names = CheckLabelExtractor.extract_all_names(lines)
        self.assertEqual(names, [])

    def test_extract_special_characters_in_name(self):
        """Test extracting names with dots, $, hyphens."""
        lines = [
            Line("// CHECK-LABEL: @foo.bar$baz-test_1", "\n", 0, 1),
        ]
        names = CheckLabelExtractor.extract_all_names(lines)
        self.assertEqual(names, ["foo.bar$baz-test_1"])

    def test_extract_with_signature(self):
        """Test extracting name when followed by function signature."""
        lines = [
            Line("// CHECK-LABEL: @test(%arg0: i32)", "\n", 0, 1),
        ]
        names = CheckLabelExtractor.extract_all_names(lines)
        self.assertEqual(names, ["test"])

    def test_extract_stream_executable(self):
        """Test extracting from stream.executable pattern."""
        lines = [
            Line("// CHECK-LABEL: stream.executable @my_dispatch", "\n", 0, 1),
        ]
        names = CheckLabelExtractor.extract_all_names(lines)
        self.assertEqual(names, ["my_dispatch"])

    def test_extract_primary_name_returns_first(self):
        """Test that extract_primary_name returns first name found."""
        lines = [
            Line("// CHECK-LABEL: @first", "\n", 0, 1),
            Line("// FOO-LABEL: @second", "\n", 1, 2),
        ]
        primary, all_names = CheckLabelExtractor.extract_primary_name(lines)
        self.assertEqual(primary, "first")
        self.assertEqual(all_names, ("first", "second"))

    def test_extract_primary_name_no_labels(self):
        """Test extract_primary_name when no labels present."""
        lines = [
            Line("func @test() { }", "\n", 0, 1),
        ]
        primary, all_names = CheckLabelExtractor.extract_primary_name(lines)
        self.assertIsNone(primary)
        self.assertEqual(all_names, ())

    def test_extract_custom_prefix_label(self):
        """Test extracting with custom prefix (e.g., ORDINAL-LABEL)."""
        lines = [
            Line("// ORDINAL-LABEL: @custom", "\n", 0, 1),
        ]
        names = CheckLabelExtractor.extract_all_names(lines)
        self.assertEqual(names, ["custom"])


if __name__ == "__main__":
    unittest.main()
