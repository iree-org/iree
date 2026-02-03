# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Extractor for sanitizer failures (ASAN, LSAN, TSAN, MSAN, UBSAN).

This extractor detects and parses sanitizer reports from various formats:
- AddressSanitizer (ASAN): heap/stack buffer overflows, use-after-free
- LeakSanitizer (LSAN): memory leaks (direct and indirect)
- ThreadSanitizer (TSAN): data races, deadlocks
- MemorySanitizer (MSAN): uninitialized memory access
- UndefinedBehaviorSanitizer (UBSAN): undefined behavior

Key features:
- Graceful degradation: Extracts partial data when reports are incomplete
- Multiple reports: LSAN can have multiple leak blocks under one ERROR section
- Line number preservation: Maps stripped content to original unstripped line numbers
- Format variations: Handles ctest, LIT, and raw sanitizer output

Design philosophy:
"Better to return 80% of fields with a line number than skip entirely
because one regex doesn't match."
"""

import re

from common.extractors.base import Extractor
from common.issues import Issue, SanitizerIssue, Severity
from common.log_buffer import LogBuffer


class SanitizerExtractor(Extractor):
    """Extracts sanitizer failure reports from logs."""

    name = "sanitizer"
    activation_keywords = [
        # Sanitizers only trigger during test execution - look for sanitizer names.
        "AddressSanitizer",
        "LeakSanitizer",
        "ThreadSanitizer",
        "MemorySanitizer",
        "==ERROR:",  # ASAN/LSAN/TSAN error marker.
        "runtime error:",  # UBSAN error marker.
    ]

    def extract(self, log_buffer: LogBuffer) -> list[Issue]:
        """Extract all sanitizer reports from log.

        Args:
            log_buffer: LogBuffer with log content (may be prefix-stripped).

        Returns:
            List of SanitizerIssue objects, one per detected report.
            Returns empty list if no sanitizer reports found.
            Never raises exceptions - all failures are graceful.
        """
        issues = []

        # Each sanitizer type has independent extraction.
        # They don't interfere with each other.
        issues.extend(self._extract_asan(log_buffer))
        issues.extend(self._extract_lsan(log_buffer))
        issues.extend(self._extract_tsan(log_buffer))
        issues.extend(self._extract_msan(log_buffer))
        issues.extend(self._extract_ubsan(log_buffer))

        return issues

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_original_line_number(
        self, log_buffer: LogBuffer, stripped_line_num: int
    ) -> int:
        """Map stripped content line number to original content line number.

        When LogBuffer strips prefixes, the line count stays the same
        (stripping only removes prefix text, not entire lines). This means
        line numbers have a 1:1 correspondence between stripped and original.

        Args:
            log_buffer: LogBuffer with potentially stripped content.
            stripped_line_num: Line number in stripped content (0-indexed).

        Returns:
            Line number in original unstripped content (0-indexed).
            Since stripping is prefix-only, this is just stripped_line_num.
        """
        # Prefix stripping doesn't change line count, so line numbers are 1:1.
        # We return the same line number, but this method exists for clarity
        # and future-proofing if we add line-removing transformations.
        return stripped_line_num

    def _parse_stack_trace(
        self,
        log_buffer: LogBuffer,
        start_line: int,
        stop_patterns: list[str] | None = None,
    ) -> tuple[list[str], int]:
        """Parse stack trace frames until blank line or stop pattern.

        Handles various stack frame formats:
        - Standard: #0 0xADDR in function file.c:123:45
        - No location: #0 0xADDR in function
        - Lambda functions: #0 ... ::operator()() const
        - Symbol lookup errors: #0 0xADDR (<module>+0x123)

        Args:
            log_buffer: LogBuffer to read from.
            start_line: Line number to start parsing (0-indexed).
            stop_patterns: List of regex patterns that end the stack.
                Examples: [r'^SUMMARY:', r'^allocated', r'^Previous']

        Returns:
            Tuple of (frames_list, next_line_number):
            - frames_list: List of frame strings (stripped)
            - next_line_number: Line number after last frame (0-indexed)
        """
        frames = []
        i = start_line
        stop_patterns = stop_patterns or []

        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Stop on blank line.
            if not line or not line.strip():
                break

            # Stop on any stop pattern.
            if any(re.match(pattern, line) for pattern in stop_patterns):
                break

            # Parse frame: #NUM at start (may have whitespace before #).
            # Collect entire frame line even if format varies.
            if re.match(r"^\s*#(\d+)\s+", line):
                frames.append(line.strip())
                i += 1
            else:
                # Not a frame line - stop parsing stack.
                break

        return frames, i

    def _find_report_end(
        self, log_buffer: LogBuffer, start_line: int, sanitizer_type: str
    ) -> int:
        """Find end of sanitizer report.

        Searches for report boundaries in order of preference:
        1. SUMMARY line (most reliable)
        2. Next ==PID==ERROR or WARNING section
        3. Shadow bytes legend end (ASAN-specific)
        4. EOF

        Args:
            log_buffer: LogBuffer to search.
            start_line: Line number to start searching from.
            sanitizer_type: Type of sanitizer ("asan", "lsan", etc.).

        Returns:
            Line number of report end (exclusive - first line after report).
        """
        # Search forward for SUMMARY line (skip current line to avoid matching self).
        for i in range(start_line + 1, log_buffer.line_count):
            line = log_buffer.get_line(i)

            # SUMMARY line marks end.
            if re.match(r"^\s*SUMMARY:", line):
                return i + 1  # Include SUMMARY line in report.

            # Next error section - don't include it.
            if re.match(r"^==\d+==ERROR:", line):
                return i

            # TSAN uses WARNING instead of ERROR.
            if re.match(r"^WARNING: ThreadSanitizer:", line):
                return i

        # No SUMMARY found - use EOF.
        return log_buffer.line_count

    def _extract_memory_info_asan(
        self, log_buffer: LogBuffer, start_line: int, search_lines: int = 10
    ) -> dict[str, any]:
        """Extract ASAN-specific memory access details.

        Parses lines like:
        - READ/WRITE of size N at 0xADDR thread T0
        - 0xADDR is located N bytes after/before M-byte region [START,END)

        Args:
            log_buffer: LogBuffer to search.
            start_line: Line number to start searching.
            search_lines: Number of lines to search forward.

        Returns:
            Dict with fields: access_type, access_size, address,
            memory_offset, memory_region_size, etc.
        """
        info = {}

        # Search within limited range after error header.
        for i in range(
            start_line, min(start_line + search_lines, log_buffer.line_count)
        ):
            line = log_buffer.get_line(i)

            # READ/WRITE of size N at 0xADDR thread T0.
            if match := re.search(
                r"^\s*(READ|WRITE) of size (\d+) at (0x[0-9a-f]+)", line
            ):
                info["access_type"] = match.group(1)
                info["access_size"] = int(match.group(2))
                info["address"] = match.group(3)

                # Thread info may be on same line.
                if thread_match := re.search(r"thread (T\d+|main thread)", line):
                    info["thread_id"] = thread_match.group(1)

            # Memory location description.
            # Example: 0xDEADBEEF is located 4 bytes after 100-byte region [0x..., 0x...)
            if match := re.search(
                r"(0x[0-9a-f]+) is located (\d+) bytes? (after|before) (\d+)-byte region",
                line,
            ):
                info["address"] = match.group(1)
                info["memory_offset"] = int(match.group(2))
                # "after" = positive offset, "before" = negative.
                if match.group(3) == "before":
                    info["memory_offset"] = -info["memory_offset"]
                info["memory_region_size"] = int(match.group(4))

            # Region bounds: [START, END).
            if match := re.search(r"\[(0x[0-9a-f]+),(0x[0-9a-f]+)\)", line):
                info["memory_region_start"] = match.group(1)
                info["memory_region_end"] = match.group(2)

        return info

    # =========================================================================
    # ASAN Extraction
    # =========================================================================

    def _extract_asan(self, log_buffer: LogBuffer) -> list[SanitizerIssue]:
        """Extract AddressSanitizer reports.

        Detects errors like:
        - heap-buffer-overflow
        - stack-buffer-overflow
        - use-after-free
        - double-free

        Args:
            log_buffer: LogBuffer to search.

        Returns:
            List of SanitizerIssue objects for ASAN errors found.
        """
        issues = []
        i = 0

        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Detection: ==PID==ERROR: AddressSanitizer: ERROR_TYPE.
            if match := re.match(r"^==(\d+)==ERROR: AddressSanitizer: ([\w-]+)", line):
                pid = int(match.group(1))
                error_type = match.group(2)
                report_start_line = i

                # Extract report data (graceful - get whatever we can).
                report_data = self._parse_asan_report(log_buffer, i, pid, error_type)

                # Remove internal fields before passing to SanitizerIssue.
                end_line = report_data.pop("end_line", i + 1)

                # Create issue even if partial.
                issue = SanitizerIssue(
                    severity=Severity.CRITICAL,
                    actionable=True,
                    message=f"AddressSanitizer: {error_type}",
                    line_number=self._get_original_line_number(
                        log_buffer, report_start_line
                    ),
                    source_extractor=self.name,
                    sanitizer_type="asan",
                    error_type=error_type,
                    pid=pid,
                    **report_data,
                )
                issues.append(issue)

                # Skip past this report.
                i = end_line
            else:
                i += 1

        return issues

    def _parse_asan_report(  # noqa: C901
        self, log_buffer: LogBuffer, start_line: int, pid: int, error_type: str
    ) -> dict[str, any]:
        """Parse ASAN report starting at error header.

        Note: Complexity is intentional for single-pass parsing efficiency.

        Extracts:
        - Primary stack trace
        - Allocation stack trace
        - Memory access info (READ/WRITE, size, address)
        - Full report text

        Args:
            log_buffer: LogBuffer containing the report.
            start_line: Line number of ERROR header.
            pid: Process ID from error header.
            error_type: Error type from header (e.g., "heap-buffer-overflow").

        Returns:
            Dict with extracted fields (primary_stack, allocation_stack, etc.).
        """
        data = {}
        data["primary_stack"] = []
        data["allocation_stack"] = []

        # Single-pass parsing: check each line for patterns.
        i = start_line + 1
        in_primary_stack = False
        in_allocation_stack = False

        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Stop at report boundaries.
            if re.match(r"^\s*SUMMARY:|^Shadow bytes around|^==\d+==", line):
                break

            # Extract memory info patterns (READ/WRITE, location, etc.).
            # Use separate if statements (not elif) since multiple patterns can appear on same line.
            if match := re.search(
                r"^\s*(READ|WRITE) of size (\d+) at (0x[0-9a-f]+)", line
            ):
                data["access_type"] = match.group(1)
                data["access_size"] = int(match.group(2))
                data["address"] = match.group(3)
                if thread_match := re.search(r"thread (T\d+|main thread)", line):
                    data["thread_id"] = thread_match.group(1)

            if match := re.search(
                r"(0x[0-9a-f]+) is located (\d+) bytes? (after|before) (\d+)-byte region",
                line,
            ):
                data["address"] = match.group(1)
                data["memory_offset"] = int(match.group(2)) * (
                    1 if match.group(3) == "after" else -1
                )
                data["memory_region_size"] = int(match.group(4))

            if match := re.search(r"\[(0x[0-9a-f]+),(0x[0-9a-f]+)\)", line):
                data["memory_region_start"] = match.group(1)
                data["memory_region_end"] = match.group(2)

            # Check for stack frames.
            if (
                re.match(r"^\s*#0\s+", line)
                and not in_primary_stack
                and not in_allocation_stack
            ):
                # First frame starts primary stack.
                in_primary_stack = True

            # Check for allocation stack marker.
            if re.search(r"allocated by thread .* here:", line):
                in_primary_stack = False
                in_allocation_stack = True
                i += 1
                continue

            # Collect stack frames.
            if re.match(r"^\s*#\d+\s+", line):
                if in_primary_stack:
                    data["primary_stack"].append(line.strip())
                elif in_allocation_stack:
                    data["allocation_stack"].append(line.strip())
            else:
                # Non-frame line ends current stack.
                if (in_primary_stack or in_allocation_stack) and (
                    data["primary_stack"] or data["allocation_stack"]
                ):
                    # Only stop if we've started collecting (empty line or non-frame after frames).
                    in_primary_stack = False
                    in_allocation_stack = False

            i += 1

        # Find report end.
        end_line = self._find_report_end(log_buffer, start_line, "asan")
        data["end_line"] = end_line

        # Collect full report text.
        report_lines = []
        for line_num in range(start_line, end_line):
            report_lines.append(log_buffer.get_line(line_num))
        data["full_report"] = "\n".join(report_lines)

        # Extract summary line if present.
        for line_num in range(start_line, end_line):
            line = log_buffer.get_line(line_num)
            if line.startswith("SUMMARY:"):
                data["summary_line"] = line.strip()
                break

        return data

    # =========================================================================
    # LSAN Extraction
    # =========================================================================

    def _extract_lsan(self, log_buffer: LogBuffer) -> list[SanitizerIssue]:
        """Extract LeakSanitizer reports.

        CRITICAL: LSAN can have multiple leak blocks under ONE ERROR section:
        ==PID==ERROR: LeakSanitizer: detected memory leaks

        Direct leak of 48 byte(s) in 2 object(s) allocated from:
            #0 ...
        Direct leak of 24 byte(s) in 1 object(s) allocated from:
            #0 ...

        SUMMARY: AddressSanitizer: 72 byte(s) leaked in 3 allocation(s).

        Returns separate SanitizerIssue for EACH leak block.

        Args:
            log_buffer: LogBuffer to search.

        Returns:
            List of SanitizerIssue objects, one per leak block found.
        """
        issues = []
        i = 0

        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Detection: ==PID==ERROR: LeakSanitizer: detected memory leaks.
            if match := re.match(
                r"^==(\d+)==ERROR: LeakSanitizer: detected memory leaks", line
            ):
                pid = int(match.group(1))
                error_section_start = i

                # Collect ALL leak blocks under this ERROR section.
                leak_blocks = []
                summary_line = ""
                j = i + 1

                while j < log_buffer.line_count:
                    leak_line = log_buffer.get_line(j)

                    # Found a leak block: Direct/Indirect leak of N byte(s) in M object(s).
                    if leak_match := re.match(
                        r"^(Direct|Indirect) leak of (\d+) byte\(s\) in (\d+) object\(s\)",
                        leak_line,
                    ):
                        leak_data = self._parse_lsan_leak_block(
                            log_buffer, j, leak_match
                        )
                        leak_blocks.append((j, leak_data))
                        j = leak_data.get("next_line", j + 1)

                    # Found SUMMARY - end of this ERROR section.
                    elif re.match(r"^\s*SUMMARY:", leak_line):
                        summary_line = leak_line.strip()
                        j += 1
                        break

                    # Next ERROR section - end.
                    elif re.match(r"^==\d+==ERROR:", leak_line):
                        break

                    else:
                        j += 1

                # Collect full report for all leaks (from ERROR to SUMMARY).
                report_lines = []
                for line_num in range(error_section_start, j):
                    report_lines.append(log_buffer.get_line(line_num))
                full_report = "\n".join(report_lines)

                # Create separate issue for each leak block.
                for leak_line_num, leak_data in leak_blocks:
                    # Remove internal fields.
                    leak_data.pop("next_line", None)

                    issue = SanitizerIssue(
                        severity=Severity.HIGH,
                        actionable=True,
                        message=f"LeakSanitizer: {leak_data['leak_type']} leak of {leak_data['leaked_bytes']} bytes",
                        line_number=self._get_original_line_number(
                            log_buffer, leak_line_num
                        ),
                        source_extractor=self.name,
                        sanitizer_type="lsan",
                        error_type="memory-leak",
                        pid=pid,
                        summary_line=summary_line,
                        full_report=full_report,
                        **leak_data,
                    )
                    issues.append(issue)

                i = j
            else:
                i += 1

        return issues

    def _parse_lsan_leak_block(
        self, log_buffer: LogBuffer, start_line: int, leak_match: re.Match
    ) -> dict[str, any]:
        """Parse a single LSAN leak block.

        Args:
            log_buffer: LogBuffer containing the leak.
            start_line: Line number of "Direct/Indirect leak of..." line.
            leak_match: Regex match object for the leak header.

        Returns:
            Dict with leak_type, leaked_bytes, leaked_objects, primary_stack, etc.
        """
        data = {}

        # Extract leak info from header.
        data["leak_type"] = leak_match.group(1)  # "Direct" or "Indirect"
        data["leaked_bytes"] = int(leak_match.group(2))
        data["leaked_objects"] = int(leak_match.group(3))

        # Look for "allocated from:" on same or next line.
        i = start_line
        while i < min(start_line + 3, log_buffer.line_count):
            line = log_buffer.get_line(i)
            if "allocated from:" in line:
                # Parse stack trace starting after "allocated from:" line.
                stack, next_line = self._parse_stack_trace(
                    log_buffer,
                    i + 1,
                    stop_patterns=[
                        r"^Direct leak",
                        r"^Indirect leak",
                        r"^\s*SUMMARY:",
                        r"^==\d+==ERROR:",
                    ],
                )
                data["primary_stack"] = stack
                data["next_line"] = next_line
                return data
            i += 1

        # No stack found - just return what we have.
        data["next_line"] = start_line + 1
        return data

    # =========================================================================
    # TSAN Extraction
    # =========================================================================

    def _extract_tsan(self, log_buffer: LogBuffer) -> list[SanitizerIssue]:
        """Extract ThreadSanitizer reports.

        TSAN format uses different separator (shorter) and has dual stacks:
        ==================
        WARNING: ThreadSanitizer: data-race (pid=PID)
          Write of size 8 at 0xADDR by thread T1:
            #0 ...
          Previous read of size 8 at 0xADDR by main thread:
            #0 ...
          Thread T1 created by main thread at:
            #0 ...
        SUMMARY: ThreadSanitizer: data-race file.c:123 in func
        ==================

        Args:
            log_buffer: LogBuffer to search.

        Returns:
            List of SanitizerIssue objects for TSAN errors found.
        """
        issues = []
        i = 0

        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Detection: WARNING: ThreadSanitizer: ERROR_TYPE (pid=PID).
            if match := re.match(
                r"^WARNING: ThreadSanitizer: ([\w-]+) \(pid=(\d+)\)", line
            ):
                error_type = match.group(1)
                pid = int(match.group(2))
                report_start_line = i

                # Parse TSAN report (dual stacks).
                report_data = self._parse_tsan_report(log_buffer, i, pid, error_type)

                # Remove internal fields.
                end_line = report_data.pop("end_line", i + 1)

                # Create issue.
                issue = SanitizerIssue(
                    severity=Severity.CRITICAL,
                    actionable=True,
                    message=f"ThreadSanitizer: {error_type}",
                    line_number=self._get_original_line_number(
                        log_buffer, report_start_line
                    ),
                    source_extractor=self.name,
                    sanitizer_type="tsan",
                    error_type=error_type,
                    pid=pid,
                    **report_data,
                )
                issues.append(issue)

                # Skip past this report.
                i = end_line
            else:
                i += 1

        return issues

    def _parse_tsan_report(
        self, log_buffer: LogBuffer, start_line: int, pid: int, error_type: str
    ) -> dict[str, any]:
        """Parse TSAN report with dual stack traces.

        Args:
            log_buffer: LogBuffer containing the report.
            start_line: Line number of WARNING header.
            pid: Process ID.
            error_type: Error type (e.g., "data-race").

        Returns:
            Dict with primary_stack, conflicting_stack, thread info, etc.
        """
        data = {}
        i = start_line + 1

        # Parse primary access stack (first stack after WARNING).
        # Look for "Write/Read of size N at 0xADDR by thread TX:".
        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Primary access description.
            if match := re.search(
                r"^\s*(Write|Read|Atomic \w+) of size (\d+) at (0x[0-9a-f]+) by (thread \w+|main thread)",
                line,
            ):
                data["access_type"] = match.group(1)
                data["access_size"] = int(match.group(2))
                data["address"] = match.group(3)
                data["thread_id"] = match.group(4)

                # Parse primary stack.
                primary_stack, i = self._parse_stack_trace(
                    log_buffer,
                    i + 1,
                    stop_patterns=[
                        r"^\s*Previous",
                        r"^\s*Thread T\d+ created",
                        r"^\s*SUMMARY:",
                    ],
                )
                data["primary_stack"] = primary_stack
                break

            i += 1

        # Parse conflicting access stack ("Previous write/read...").
        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            if re.match(r"^\s*Previous (write|read)", line):
                # Parse conflicting stack.
                conflicting_stack, i = self._parse_stack_trace(
                    log_buffer,
                    i + 1,
                    stop_patterns=[
                        r"^\s*Thread T\d+ created",
                        r"^\s*SUMMARY:",
                    ],
                )
                data["conflicting_stack"] = conflicting_stack
                break

            # Stop if we hit SUMMARY.
            if re.match(r"^\s*SUMMARY:", line):
                break

            i += 1

        # Find report end.
        end_line = self._find_report_end(log_buffer, start_line, "tsan")
        data["end_line"] = end_line

        # Collect full report.
        report_lines = []
        for line_num in range(start_line, end_line):
            report_lines.append(log_buffer.get_line(line_num))
        data["full_report"] = "\n".join(report_lines)

        # Extract SUMMARY.
        for line_num in range(start_line, end_line):
            line = log_buffer.get_line(line_num)
            if line.strip().startswith("SUMMARY:"):
                data["summary_line"] = line.strip()
                break

        return data

    # =========================================================================
    # MSAN Extraction
    # =========================================================================

    def _extract_msan(self, log_buffer: LogBuffer) -> list[SanitizerIssue]:
        """Extract MemorySanitizer reports.

        MSAN detects use of uninitialized values:
        ==PID== WARNING: MemorySanitizer: use-of-uninitialized-value
            #0 ...
          Uninitialized value was created by an allocation of 'var'
            #0 ...
        SUMMARY: MemorySanitizer: use-of-uninitialized-value file.c:123

        Args:
            log_buffer: LogBuffer to search.

        Returns:
            List of SanitizerIssue objects for MSAN errors found.
        """
        issues = []
        i = 0

        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Detection: ==PID== WARNING: MemorySanitizer: ERROR_TYPE.
            if match := re.match(
                r"^==(\d+)== WARNING: MemorySanitizer: ([\w-]+)", line
            ):
                pid = int(match.group(1))
                error_type = match.group(2)
                report_start_line = i

                # Parse MSAN report.
                report_data = self._parse_msan_report(log_buffer, i, pid, error_type)

                # Remove internal fields.
                end_line = report_data.pop("end_line", i + 1)

                # Create issue.
                issue = SanitizerIssue(
                    severity=Severity.CRITICAL,
                    actionable=True,
                    message=f"MemorySanitizer: {error_type}",
                    line_number=self._get_original_line_number(
                        log_buffer, report_start_line
                    ),
                    source_extractor=self.name,
                    sanitizer_type="msan",
                    error_type=error_type,
                    pid=pid,
                    **report_data,
                )
                issues.append(issue)

                # Skip past this report.
                i = end_line
            else:
                i += 1

        return issues

    def _parse_msan_report(
        self, log_buffer: LogBuffer, start_line: int, pid: int, error_type: str
    ) -> dict[str, any]:
        """Parse MSAN report.

        Args:
            log_buffer: LogBuffer containing the report.
            start_line: Line number of WARNING header.
            pid: Process ID.
            error_type: Error type.

        Returns:
            Dict with primary_stack, origin_description, etc.
        """
        data = {}
        i = start_line + 1

        # Parse primary stack (where uninitialized value was used).
        primary_stack, i = self._parse_stack_trace(
            log_buffer,
            i,
            stop_patterns=[
                r"^\s*Uninitialized",
                r"^\s*SUMMARY:",
            ],
        )
        data["primary_stack"] = primary_stack

        # Look for origin description.
        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            if re.match(r"^\s*Uninitialized value was created", line):
                data["origin_description"] = line.strip()
                # Parse origin stack if present.
                origin_stack, i = self._parse_stack_trace(
                    log_buffer,
                    i + 1,
                    stop_patterns=[r"^\s*SUMMARY:"],
                )
                data["allocation_stack"] = origin_stack
                break

            if re.match(r"^\s*SUMMARY:", line):
                break

            i += 1

        # Find report end.
        end_line = self._find_report_end(log_buffer, start_line, "msan")
        data["end_line"] = end_line

        # Collect full report.
        report_lines = []
        for line_num in range(start_line, end_line):
            report_lines.append(log_buffer.get_line(line_num))
        data["full_report"] = "\n".join(report_lines)

        # Extract SUMMARY.
        for line_num in range(start_line, end_line):
            line = log_buffer.get_line(line_num)
            if line.strip().startswith("SUMMARY:"):
                data["summary_line"] = line.strip()
                break

        return data

    # =========================================================================
    # UBSAN Extraction
    # =========================================================================

    def _extract_ubsan(self, log_buffer: LogBuffer) -> list[SanitizerIssue]:
        """Extract UndefinedBehaviorSanitizer reports.

        UBSAN uses inline format (no separator):
        file.c:123:45: runtime error: signed integer overflow
        SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior file.c:123:45

        Args:
            log_buffer: LogBuffer to search.

        Returns:
            List of SanitizerIssue objects for UBSAN errors found.
        """
        issues = []
        i = 0

        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Detection: FILE:LINE:COL: runtime error: DESCRIPTION.
            # Must have absolute path or relative path with extension.
            if match := re.match(
                r"^([^:]+\.\w+):(\d+):(\d+): runtime error: (.+)$", line
            ):
                file_path = match.group(1)
                line_num = int(match.group(2))
                column = int(match.group(3))
                description = match.group(4)
                report_start_line = i

                # Look for SUMMARY on next few lines.
                summary_line = ""
                for j in range(i + 1, min(i + 5, log_buffer.line_count)):
                    summary_candidate = log_buffer.get_line(j)
                    if summary_candidate.strip().startswith(
                        "SUMMARY: UndefinedBehaviorSanitizer"
                    ):
                        summary_line = summary_candidate.strip()
                        break

                # Create issue.
                issue = SanitizerIssue(
                    severity=Severity.HIGH,
                    actionable=True,
                    message=f"UndefinedBehaviorSanitizer: {description}",
                    line_number=self._get_original_line_number(
                        log_buffer, report_start_line
                    ),
                    source_extractor=self.name,
                    sanitizer_type="ubsan",
                    error_type="undefined-behavior",
                    pid=0,  # UBSAN doesn't show PID.
                    ubsan_file=file_path,
                    ubsan_line=line_num,
                    ubsan_column=column,
                    full_report=f"{line}\n{summary_line}" if summary_line else line,
                    summary_line=summary_line,
                )
                issues.append(issue)

            i += 1

        return issues
