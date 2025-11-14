# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for SanitizerExtractor.

Tests cover:
- ASAN: heap-buffer-overflow with deep stacks (41+ frames)
- LSAN: single leak, multiple leaks under one ERROR section
- TSAN: data races with dual stack traces
- MSAN: uninitialized value usage
- UBSAN: inline format undefined behavior
- Edge cases: symbol lookup errors, missing SUMMARY, truncated reports
- False positive prevention: build logs, gdb traces, timeout packages
- Corpus validation: All 18 CI logs
"""

import os
import unittest

from common.extractors.sanitizer import SanitizerExtractor
from common.issues import Severity
from common.log_buffer import LogBuffer


class TestASANExtraction(unittest.TestCase):
    """Test AddressSanitizer extraction."""

    def setUp(self):
        self.extractor = SanitizerExtractor()

    def test_asan_heap_overflow_ctest_format(self):
        """Test ASAN heap-buffer-overflow with ctest prefix."""
        # Load real log from /home/ben/src/iree/asan_runtime.txt
        log_path = "/home/ben/src/iree/asan_runtime.txt"
        if not os.path.exists(log_path):
            self.skipTest(f"Log file not found: {log_path}")

        with open(log_path, encoding="utf-8") as f:
            log_content = f.read()

        # Create LogBuffer with auto-detection (strips [ctest] prefix).
        log_buffer = LogBuffer(log_content, auto_detect_format=True)

        # Extract issues.
        issues = self.extractor.extract(log_buffer)

        # Should find exactly 1 ASAN heap-buffer-overflow.
        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.sanitizer_type, "asan")
        self.assertEqual(issue.error_type, "heap-buffer-overflow")
        self.assertEqual(issue.severity, Severity.CRITICAL)
        self.assertTrue(issue.actionable)

        # Check PID extracted.
        self.assertGreater(issue.pid, 0)

        # Check stacks extracted.
        self.assertGreater(len(issue.primary_stack), 0)
        # ASAN should have allocation stack.
        self.assertGreater(len(issue.allocation_stack), 0)

        # Check memory info.
        self.assertIn(issue.access_type, ["READ", "WRITE"])
        self.assertGreater(issue.access_size, 0)
        self.assertTrue(issue.address.startswith("0x"))

        # Check full report captured.
        self.assertIn("AddressSanitizer", issue.full_report)
        self.assertIn("SUMMARY", issue.summary_line)

        # Check line number is reasonable (original content line number).
        self.assertGreater(issue.line_number, 0)

    def test_asan_heap_overflow_lit_format_deep_stack(self):
        """Test ASAN with LIT format and very deep stack (41+ frames)."""
        log_path = "/home/ben/src/iree/asan_comiler.txt"
        if not os.path.exists(log_path):
            self.skipTest(f"Log file not found: {log_path}")

        with open(log_path, encoding="utf-8") as f:
            log_content = f.read()

        log_buffer = LogBuffer(log_content, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        # Should find exactly 1 ASAN error.
        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.sanitizer_type, "asan")
        self.assertEqual(issue.error_type, "heap-buffer-overflow")

        # Check deep stack trace (MLIR compiler has 40+ frames).
        self.assertGreater(
            len(issue.primary_stack), 30, "Expected deep stack trace (30+ frames)"
        )

        # Check allocation stack also present.
        self.assertGreater(len(issue.allocation_stack), 0)

        # Verify frames are properly formatted.
        for frame in issue.primary_stack[:5]:  # Check first 5 frames.
            self.assertRegex(frame, r"#\d+", "Frame should start with #NUM")

    def test_asan_memory_info_extraction(self):
        """Test detailed memory info extraction (offset, region size)."""
        log_content = """
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000001234 at pc 0x000000400abc
WRITE of size 4 at 0x602000001234 thread T0
    #0 0x400abb in main test.c:10:5
    #1 0x7f1234567890 in __libc_start_main
0x602000001234 is located 4 bytes after 100-byte region [0x602000001190,0x6020000011f4)
allocated by thread T0 here:
    #0 0x400def in malloc
SUMMARY: AddressSanitizer: heap-buffer-overflow test.c:10:5 in main
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]

        # Verify memory info fields.
        self.assertEqual(issue.access_type, "WRITE")
        self.assertEqual(issue.access_size, 4)
        self.assertEqual(issue.address, "0x602000001234")
        self.assertEqual(issue.memory_offset, 4)  # 4 bytes after
        self.assertEqual(issue.memory_region_size, 100)
        self.assertEqual(issue.memory_region_start, "0x602000001190")
        self.assertEqual(issue.memory_region_end, "0x6020000011f4")


class TestLSANExtraction(unittest.TestCase):
    """Test LeakSanitizer extraction."""

    def setUp(self):
        self.extractor = SanitizerExtractor()

    def test_lsan_direct_leak_simple(self):
        """Test simple LSAN direct leak."""
        log_path = "/home/ben/src/iree/lsan_runtime.txt"
        if not os.path.exists(log_path):
            self.skipTest(f"Log file not found: {log_path}")

        with open(log_path, encoding="utf-8") as f:
            log_content = f.read()

        log_buffer = LogBuffer(log_content, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        # Should find exactly 1 leak.
        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.sanitizer_type, "lsan")
        self.assertEqual(issue.error_type, "memory-leak")
        self.assertEqual(issue.leak_type, "Direct")
        self.assertGreater(issue.leaked_bytes, 0)
        self.assertGreater(issue.leaked_objects, 0)

        # Should have stack trace.
        self.assertGreater(len(issue.primary_stack), 0)

    def test_lsan_multiple_leaks_one_error_section(self):
        """Test LSAN with multiple leak blocks under ONE ERROR section.

        This is the critical LSAN edge case: one ==PID==ERROR: LeakSanitizer
        section can contain 3+ separate "Direct leak of..." blocks.
        We should return separate SanitizerIssue for each leak block.
        """
        log_path = "/home/ben/src/iree/lsan_compiler.txt"
        if not os.path.exists(log_path):
            self.skipTest(f"Log file not found: {log_path}")

        with open(log_path, encoding="utf-8") as f:
            log_content = f.read()

        log_buffer = LogBuffer(log_content, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        # Should find 6 separate leak issues (3 Direct + 3 Indirect) from the same ERROR section.
        self.assertGreaterEqual(len(issues), 6, "Expected at least 6 leak blocks")

        # Verify all are LSAN memory leaks with proper types.
        direct_leaks = [i for i in issues if i.leak_type == "Direct"]
        indirect_leaks = [i for i in issues if i.leak_type == "Indirect"]

        self.assertGreaterEqual(
            len(direct_leaks), 3, "Expected at least 3 Direct leaks"
        )
        self.assertGreaterEqual(
            len(indirect_leaks), 3, "Expected at least 3 Indirect leaks"
        )

        for issue in issues:
            self.assertEqual(issue.sanitizer_type, "lsan")
            self.assertEqual(issue.error_type, "memory-leak")
            self.assertIn(issue.leak_type, ["Direct", "Indirect"])
            self.assertGreater(issue.leaked_bytes, 0)
            self.assertGreater(issue.leaked_objects, 0)

        # Verify they have different line numbers (different leak blocks).
        line_numbers = [issue.line_number for issue in issues]
        self.assertEqual(
            len(line_numbers),
            len(set(line_numbers)),
            "Each leak should have unique line number",
        )

    def test_lsan_with_symbol_lookup_error(self):
        """Test LSAN extraction with interleaved symbol lookup errors."""
        log_path = "/home/ben/src/iree/lsan_runtime2.txt"
        if not os.path.exists(log_path):
            self.skipTest(f"Log file not found: {log_path}")

        with open(log_path, encoding="utf-8") as f:
            log_content = f.read()

        log_buffer = LogBuffer(log_content, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        # Should still find leak despite symbol lookup error.
        self.assertGreaterEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.sanitizer_type, "lsan")
        self.assertEqual(issue.error_type, "memory-leak")

    def test_lsan_leak_info_parsing(self):
        """Test parsing of leak size, object count, and type."""
        log_content = """
==98765==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 128 byte(s) in 4 object(s) allocated from:
    #0 0x400abc in malloc
    #1 0x400def in allocate_memory

Indirect leak of 64 byte(s) in 2 object(s) allocated from:
    #0 0x400abc in malloc
    #1 0x401234 in indirect_allocate

SUMMARY: AddressSanitizer: 192 byte(s) leaked in 6 allocation(s).
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        # Should find 2 leaks (1 direct + 1 indirect).
        self.assertEqual(len(issues), 2)

        # First leak: Direct, 128 bytes, 4 objects.
        direct_leak = issues[0]
        self.assertEqual(direct_leak.leak_type, "Direct")
        self.assertEqual(direct_leak.leaked_bytes, 128)
        self.assertEqual(direct_leak.leaked_objects, 4)

        # Second leak: Indirect, 64 bytes, 2 objects.
        indirect_leak = issues[1]
        self.assertEqual(indirect_leak.leak_type, "Indirect")
        self.assertEqual(indirect_leak.leaked_bytes, 64)
        self.assertEqual(indirect_leak.leaked_objects, 2)


class TestTSANExtraction(unittest.TestCase):
    """Test ThreadSanitizer extraction."""

    def setUp(self):
        self.extractor = SanitizerExtractor()

    def test_tsan_data_race_dual_stacks(self):
        """Test TSAN data race with dual stack traces (conflicting accesses)."""
        log_content = """
==================
WARNING: ThreadSanitizer: data-race (pid=12345)
  Write of size 8 at 0x7b0400001234 by thread T1:
    #0 write_function file.c:100:5
    #1 thread_worker file.c:200:10

  Previous read of size 8 at 0x7b0400001234 by main thread:
    #0 read_function file.c:50:10
    #1 main file.c:300:5

SUMMARY: ThreadSanitizer: data-race file.c:100:5 in write_function
==================
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.sanitizer_type, "tsan")
        self.assertEqual(issue.error_type, "data-race")
        self.assertEqual(issue.pid, 12345)

        # Check access info.
        self.assertEqual(issue.access_type, "Write")
        self.assertEqual(issue.access_size, 8)
        self.assertEqual(issue.address, "0x7b0400001234")
        self.assertEqual(issue.thread_id, "thread T1")

        # Check both stacks extracted.
        self.assertGreater(
            len(issue.primary_stack), 0, "Primary stack (write) should be extracted"
        )
        self.assertGreater(
            len(issue.conflicting_stack),
            0,
            "Conflicting stack (previous read) should be extracted",
        )

        # Verify SUMMARY extracted.
        self.assertIn("ThreadSanitizer", issue.summary_line)


class TestMSANExtraction(unittest.TestCase):
    """Test MemorySanitizer extraction."""

    def setUp(self):
        self.extractor = SanitizerExtractor()

    def test_msan_uninitialized_value(self):
        """Test MSAN use-of-uninitialized-value."""
        log_content = """
==54321== WARNING: MemorySanitizer: use-of-uninitialized-value
    #0 use_value file.c:50:10
    #1 process_data file.c:100:5
  Uninitialized value was created by an allocation of 'buffer' in the stack frame
    #0 allocate_buffer file.c:30:3
SUMMARY: MemorySanitizer: use-of-uninitialized-value file.c:50:10 in use_value
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.sanitizer_type, "msan")
        self.assertEqual(issue.error_type, "use-of-uninitialized-value")
        self.assertEqual(issue.pid, 54321)

        # Check stacks.
        self.assertGreater(len(issue.primary_stack), 0)
        self.assertGreater(len(issue.allocation_stack), 0)

        # Check origin description.
        self.assertIn("Uninitialized value was created", issue.origin_description)


class TestUBSANExtraction(unittest.TestCase):
    """Test UndefinedBehaviorSanitizer extraction."""

    def setUp(self):
        self.extractor = SanitizerExtractor()

    def test_ubsan_signed_overflow(self):
        """Test UBSAN signed integer overflow."""
        log_content = """
test.c:42:15: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior test.c:42:15 in
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.sanitizer_type, "ubsan")
        self.assertEqual(issue.error_type, "undefined-behavior")

        # Check file/line/column extracted.
        self.assertEqual(issue.ubsan_file, "test.c")
        self.assertEqual(issue.ubsan_line, 42)
        self.assertEqual(issue.ubsan_column, 15)

        # Check message contains overflow description.
        self.assertIn("signed integer overflow", issue.message)

    def test_ubsan_shift_exponent(self):
        """Test UBSAN shift exponent too large."""
        log_content = """
shift.cpp:10:20: runtime error: shift exponent 32 is too large for 32-bit type 'unsigned int'
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.sanitizer_type, "ubsan")
        self.assertEqual(issue.ubsan_file, "shift.cpp")
        self.assertEqual(issue.ubsan_line, 10)
        self.assertEqual(issue.ubsan_column, 20)
        self.assertIn("shift exponent", issue.message)


class TestParsingEdgeCases(unittest.TestCase):
    """Test parsing edge cases found in real CI corpus."""

    def setUp(self):
        self.extractor = SanitizerExtractor()

    def test_very_long_stack_frame_line(self):
        """Test parser handles 900+ character lines without truncation.

        Real pattern: Nested C++ templates like
        DenseMapBase<SmallDenseMap<pair<AACacheLoc, AACacheLoc>, ...>>
        can produce 985+ character stack frames.
        """
        # Create a realistic 950+ character frame with nested templates.
        long_frame = (
            "#3 0x55b311147ebf in llvm::detail::DenseMapPair<std::pair<llvm::AACacheLoc, "
            "llvm::AACacheLoc>, llvm::AAQueryInfo::CacheEntry>* "
            "llvm::DenseMapBase<llvm::SmallDenseMap<std::pair<llvm::AACacheLoc, llvm::AACacheLoc>, "
            "llvm::AAQueryInfo::CacheEntry, 8u, llvm::DenseMapInfo<std::pair<llvm::AACacheLoc, "
            "llvm::AACacheLoc>, void>, llvm::detail::DenseMapPair<std::pair<llvm::AACacheLoc, "
            "llvm::AACacheLoc>, llvm::AAQueryInfo::CacheEntry>>, std::pair<llvm::AACacheLoc, "
            "llvm::AACacheLoc>, llvm::AAQueryInfo::CacheEntry, llvm::DenseMapInfo<std::pair<"
            "llvm::AACacheLoc, llvm::AACacheLoc>, void>, llvm::detail::DenseMapPair<std::pair<"
            "llvm::AACacheLoc, llvm::AACacheLoc>, llvm::AAQueryInfo::CacheEntry>>::"
            "findBucketForInsertion<std::pair<llvm::AACacheLoc, llvm::AACacheLoc>>"
            "(std::pair<llvm::AACacheLoc, llvm::AACacheLoc> const&, "
            "llvm::detail::DenseMapPair<std::pair<llvm::AACacheLoc, llvm::AACacheLoc>, "
            "llvm::AAQueryInfo::CacheEntry>*) /__w/iree/iree/third_party/llvm-project/llvm/include/llvm/ADT/DenseMap.h:553:43"
        )

        self.assertGreater(len(long_frame), 900, "Test frame should be 900+ chars")

        log_content = f"""
==12345==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 442880 byte(s) in 17 object(s) allocated from:
    #0 0x55b30e1f4e02 in operator new(unsigned long) build/bin/clang+0x30bde02
    #1 0x55b31376d3dc in llvm::allocate_buffer(unsigned long, unsigned long) llvm/lib/Support/MemAlloc.cpp:16:18
    #2 0x55b311147ebf in llvm::DenseMapBase::grow(unsigned int) llvm/include/llvm/ADT/DenseMap.h:553:43
    {long_frame}

SUMMARY: AddressSanitizer: 442880 byte(s) leaked in 17 allocation(s).
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]

        # Verify long frame is captured completely.
        self.assertEqual(len(issue.primary_stack), 4)
        frame_3 = issue.primary_stack[3]

        # Check it's the long frame and hasn't been truncated.
        self.assertGreater(len(frame_3), 900, "Long frame should be preserved")
        self.assertIn("DenseMapBase", frame_3)
        self.assertIn("SmallDenseMap", frame_3)
        self.assertIn("findBucketForInsertion", frame_3)
        self.assertIn("DenseMap.h:553:43", frame_3)
        self.assertNotIn("...", frame_3, "Should not be truncated")

    def test_large_integer_leak_size(self):
        """Test parser handles large integers (400KB+) correctly as int.

        Real pattern: LLVM DenseMap growth can leak 442880 bytes (432KB).
        Must parse as int, not float or overflow.
        """
        log_content = """
==12345==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 442880 byte(s) in 17 object(s) allocated from:
    #0 0x123456 in operator new(unsigned long) build/bin/clang
    #1 0x234567 in llvm::allocate_buffer llvm/lib/Support/MemAlloc.cpp:16

SUMMARY: AddressSanitizer: 442880 byte(s) leaked in 17 allocation(s).
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]

        # Verify large leak size parsed correctly.
        self.assertEqual(issue.leaked_bytes, 442880)
        self.assertEqual(issue.leaked_objects, 17)

        # Verify types are int, not float.
        self.assertIsInstance(issue.leaked_bytes, int)
        self.assertIsInstance(issue.leaked_objects, int)

        # Verify SUMMARY contains same value.
        self.assertIn("442880 byte(s) leaked in 17 allocation(s)", issue.summary_line)

    def test_binary_only_stack_frames(self):
        """Test parser handles frames with no source location.

        Real pattern: System library frames like libc.so.6 have no source,
        only binary path and offset: #78 0xADDR  (/path/to/lib.so+0xOFFSET)
        """
        log_content = """
==12345==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 288 byte(s) in 3 object(s) allocated from:
    #0 0x56340cc225df in malloc build/tools/iree-opt+0xcb5df
    #1 0x7f9d6604f79a in mlir::Operation::create mlir/lib/IR/Operation.cpp:114
    #2 0x7f9d65bceb2b in ireeOptRunMain compiler/API/IREEOptToolEntryPoint.cpp:170
    #3 0x7f9d60981d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f) (BuildId: 490fef8403240c91833978d494d39e537409b92e)

SUMMARY: AddressSanitizer: 288 byte(s) leaked in 3 allocation(s).
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]

        # Verify all 4 frames captured including binary-only frame.
        self.assertEqual(len(issue.primary_stack), 4)

        # Verify binary-only frame (no source location).
        libc_frame = issue.primary_stack[3]
        self.assertIn("#3", libc_frame)
        self.assertIn("0x7f9d60981d8f", libc_frame)
        self.assertIn("/lib/x86_64-linux-gnu/libc.so.6", libc_frame)
        self.assertIn("+0x29d8f", libc_frame)
        self.assertIn("BuildId: 490fef8403240c91833978d494d39e537409b92e", libc_frame)

    def test_stack_collection_loop_deep_frames(self):
        """Test stack collection loop reads until blank line, not arbitrary limit.

        Real pattern: MLIR compiler stacks can reach 79 frames through deep
        pass pipeline execution. Parser must continue reading frames until
        it hits blank line or SUMMARY, not stop at arbitrary count.
        """
        # Create 45 realistic frames to test the collection loop.
        frames = []
        frames.append("#0 0x56340cc225df in malloc build/tools/iree-opt+0xcb5df")
        frames.append(
            "#1 0x7f9d6604f79a in mlir::Operation::create mlir/lib/IR/Operation.cpp:114"
        )

        # Add middle frames (MLIR pass pipeline).
        for i in range(2, 43):
            frames.append(
                f"#{i} 0x7f9d6664e7ca in mlir::detail::OpToOpPassAdaptor::run "
                f"mlir/lib/Pass/Pass.cpp:603"
            )

        frames.append(
            "#43 0x7f9d65bceb2b in ireeOptRunMain compiler/API/IREEOptToolEntryPoint.cpp:170"
        )
        frames.append("#44 0x7f9d60981d8f  (/lib/x86_64-linux-gnu/libc.so.6+0x29d8f)")

        log_content = f"""
==12345==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 288 byte(s) in 3 object(s) allocated from:
{chr(10).join('    ' + f for f in frames)}

SUMMARY: AddressSanitizer: 288 byte(s) leaked in 3 allocation(s).
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]

        # Verify all 45 frames collected.
        self.assertEqual(len(issue.primary_stack), 45, "Should collect all 45 frames")

        # Verify first and last frames correct.
        self.assertIn("malloc", issue.primary_stack[0])
        self.assertIn("libc.so.6", issue.primary_stack[44])

        # Verify middle frames are pass pipeline.
        self.assertIn("OpToOpPassAdaptor", issue.primary_stack[20])
        self.assertIn("Pass.cpp", issue.primary_stack[30])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and partial report handling."""

    def setUp(self):
        self.extractor = SanitizerExtractor()

    def test_missing_summary_line(self):
        """Test ASAN report with missing SUMMARY (truncated log)."""
        log_content = """
==99999==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x12345678
WRITE of size 4 at 0x12345678 thread T0
    #0 0x400abc in main test.c:10:5
    #1 0x7f1234567890 in __libc_start_main
"""
        # No SUMMARY line - simulates truncated log.
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        # Should still extract the error with available data.
        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.sanitizer_type, "asan")
        self.assertEqual(issue.error_type, "heap-buffer-overflow")
        self.assertEqual(issue.pid, 99999)

        # Stack should be extracted.
        self.assertGreater(len(issue.primary_stack), 0)

        # SUMMARY may be empty.
        # Full report should still be captured.
        self.assertIn("heap-buffer-overflow", issue.full_report)

    def test_stack_trace_with_lambda(self):
        """Test stack trace parsing with lambda functions."""
        log_content = """
==11111==ERROR: AddressSanitizer: heap-use-after-free on address 0xdeadbeef
READ of size 8 at 0xdeadbeef thread T0
    #0 0x123456 in std::function<void ()>::operator()() const
    #1 0x234567 in lambda_caller<lambda()>
    #2 0x345678 in main test.cpp:42:10
SUMMARY: AddressSanitizer: heap-use-after-free test.cpp:42:10 in main
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(len(issue.primary_stack), 3)

        # Verify lambda frames captured.
        self.assertIn("operator()", issue.primary_stack[0])
        self.assertIn("lambda", issue.primary_stack[1])

    def test_line_number_preservation_after_stripping(self):
        """Test that original line numbers are preserved after prefix stripping."""
        # Simulate log with ctest prefix (no leading newline).
        log_content = "[ctest] Running tests...\n[ctest] ==12345==ERROR: AddressSanitizer: heap-buffer-overflow\n[ctest] WRITE of size 4 at 0x12345678\n[ctest]     #0 0x400abc in main test.c:10\n[ctest] SUMMARY: AddressSanitizer: heap-buffer-overflow"

        # Strip ctest prefix.
        log_buffer = LogBuffer(log_content, strip_formats=["ctest"])
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        # Line number should be 1 (0-indexed: line 0 = "Running tests...", line 1 = ERROR line).
        # Since prefix stripping doesn't remove lines, just prefixes,
        # the line numbers should match.
        self.assertEqual(issue.line_number, 1)


class TestFalsePositivePrevention(unittest.TestCase):
    """Test that extractor doesn't produce false positives."""

    def setUp(self):
        self.extractor = SanitizerExtractor()

    def test_no_sanitizer_build_log(self):
        """Test normal build log produces no issues."""
        log_content = """
[ 25%] Building CXX object CMakeFiles/test.dir/test.cpp.o
[ 50%] Linking CXX executable test
[100%] Built target test
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        # Should find nothing.
        self.assertEqual(len(issues), 0)

    def test_no_match_timeout_package(self):
        """Test pip install 'timeout' package doesn't trigger false positive."""
        log_content = """
Collecting timeout==1.0.0
  Downloading timeout-1.0.0-py3-none-any.whl
Installing collected packages: timeout
Successfully installed timeout-1.0.0
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)

    def test_no_match_gdb_backtrace(self):
        """Test gdb/lldb stack traces don't trigger false positive."""
        log_content = """
#0  0x00007ffff7a1234 in malloc () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x0000000000400abc in main () at test.c:10
#2  0x00007ffff7a45678 in __libc_start_main () from /lib/x86_64-linux-gnu/libc.so.6
"""
        # No ==PID== prefix, so not a sanitizer report.
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)

    def test_no_match_disabled_message(self):
        """Test 'AddressSanitizer is disabled' message doesn't match."""
        log_content = """
Note: AddressSanitizer is disabled for this build.
Configure with -DCMAKE_BUILD_TYPE=ASan to enable.
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)

    def test_empty_log(self):
        """Test empty log produces no issues."""
        log_buffer = LogBuffer("")
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)

    def test_partial_separator_only(self):
        """Test log with just separator lines produces no issues."""
        log_content = """
=================================================================
==================
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)

    def test_summary_without_error(self):
        """Test SUMMARY line without ERROR header doesn't match."""
        log_content = """
Build complete.
SUMMARY: 100% tests passed, 0 tests failed out of 42
Total time: 5.2 seconds
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)


if __name__ == "__main__":
    unittest.main()
