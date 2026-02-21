# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""FileCheck pattern parsing for MLIR lit tests.

Parses CHECK directives (CHECK, CHECK-NEXT, CHECK-SAME, CHECK-LABEL, CHECK-NOT)
to extract operation names and FileCheck variable captures.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from lit_tools.core.ir_index import IRIndex

logger = logging.getLogger(__name__)


@dataclass
class CheckCapture:
    """A FileCheck variable capture.

    Attributes:
        name: Capture variable name (e.g., "AWAITED" from %[[AWAITED]])
        pattern: Regex pattern if this is a definition (e.g., ".+")
                 None if this is a reference (using previously defined capture)
        position: Character offset in the CHECK line where capture appears
        is_definition: True if %[[NAME:pattern]], False if %[[NAME]]
    """

    name: str
    pattern: str | None
    position: int
    is_definition: bool


@dataclass
class CheckPattern:
    """A parsed CHECK directive.

    Attributes:
        check_type: Type of check (CHECK, CHECK-NEXT, CHECK-SAME, CHECK-LABEL, CHECK-NOT)
        line_num: Absolute line number in test file
        operation: Operation name being verified (e.g., "scf.for"), or None
        captures: List of FileCheck captures in order of appearance
        raw_pattern: The full pattern after // CHECK: (for debugging)
    """

    check_type: str
    line_num: int
    operation: str | None
    captures: list[CheckCapture]
    raw_pattern: str


class CheckPatternParser:
    """Parses CHECK directives from lit test files.

    Extracts operation names and FileCheck captures from CHECK patterns to
    enable content-based matching against IR.

    Example:
        >>> parser = CheckPatternParser()
        >>> pattern = parser.parse_check_line("// CHECK: %[[X:.+]] = arith.constant", 42)
        >>> pattern.operation
        'arith.constant'
        >>> pattern.captures[0].name
        'X'
    """

    # Pattern to match CHECK directives.
    # Matches: CHECK, CHECK-NEXT, CHECK-SAME, CHECK-LABEL, CHECK-NOT
    # Also supports custom prefixes via --check-prefix: FOO, FOO-NEXT, AMDGPU, etc.
    # Group 1: Full directive (e.g., "CHECK", "FOO-NEXT", "AMDGPU")
    # Group 2: Pattern content after the colon
    CHECK_DIRECTIVE_PATTERN = re.compile(
        r"^\s*//\s*([A-Z][A-Z0-9_]*(?:-NEXT|-SAME|-LABEL|-NOT)?)\s*:\s*(.*)$"
    )

    # Pattern to match FileCheck captures.
    # Matches: %[[NAME]] (reference) or %[[NAME:pattern]] (definition)
    # Enforces UPPER_SNAKE_CASE convention for capture names.
    # Pattern captures everything up to the closing ]] (non-greedy).
    CAPTURE_PATTERN = re.compile(r"%\[\[([A-Z_][A-Z0-9_]*)(?::(.+?))?\]\]")

    # Placeholder used when sanitizing captures for operation extraction.
    # Uses SSA syntax (%0) which won't match operation patterns (ops don't start with %).
    CAPTURE_PLACEHOLDER = "%0"

    def parse_check_line(self, line: str, line_num: int) -> CheckPattern | None:
        """Parse a single CHECK line.

        Args:
            line: Line content (may or may not be a CHECK directive)
            line_num: Absolute line number in test file

        Returns:
            CheckPattern if line is a CHECK directive, None otherwise
        """
        # Check if this is a CHECK directive.
        match = self.CHECK_DIRECTIVE_PATTERN.match(line)
        if not match:
            return None

        check_type = match.group(1)
        raw_pattern = match.group(2)

        # Extract captures from the pattern.
        captures = self._extract_captures(raw_pattern)

        # Extract operation name (if present).
        operation = self._extract_operation(raw_pattern, captures)

        return CheckPattern(
            check_type=check_type,
            line_num=line_num,
            operation=operation,
            captures=captures,
            raw_pattern=raw_pattern,
        )

    def parse_file(self, lines: list[str]) -> list[CheckPattern]:
        """Parse all CHECK directives from a test file.

        Args:
            lines: Test file content split by newlines

        Returns:
            List of CheckPattern objects in file order
        """
        patterns = []
        for i, line in enumerate(lines):
            pattern = self.parse_check_line(line, i)
            if pattern:
                patterns.append(pattern)
        return patterns

    def _extract_captures(self, pattern: str) -> list[CheckCapture]:
        """Extract all FileCheck captures from a pattern.

        Args:
            pattern: The pattern content after // CHECK:

        Returns:
            List of CheckCapture objects in order of appearance
        """
        captures = []
        for match in self.CAPTURE_PATTERN.finditer(pattern):
            name = match.group(1)
            regex_pattern = match.group(2)  # None if reference
            position = match.start()
            is_definition = regex_pattern is not None

            captures.append(
                CheckCapture(
                    name=name,
                    pattern=regex_pattern,
                    position=position,
                    is_definition=is_definition,
                )
            )

        return captures

    def _extract_operation(
        self, pattern: str, captures: list[CheckCapture]
    ) -> str | None:
        """Extract operation name, avoiding FileCheck capture syntax.

        Strategy:
        1. Sanitize captures to avoid matching FileCheck syntax
        2. Look for operations AFTER '=' (assignment pattern)
        3. If no '=', look for namespaced operations (e.g., scf.for)
        4. Single-word identifiers without '.' are likely not operations

        Args:
            pattern: The pattern content after // CHECK:
            captures: Already extracted captures (for reference)

        Returns:
            Operation name if found, None otherwise
        """
        # Sanitize: replace all captures with placeholder to avoid confusing
        # operation extraction with FileCheck syntax.
        sanitized_pattern = self.CAPTURE_PATTERN.sub(self.CAPTURE_PLACEHOLDER, pattern)

        # Strategy: Look for operation AFTER '=' if this is an SSA assignment.
        # SSA assignments START with % (e.g., "%[[X]] = arith.constant", "%x = arith.constant")
        # Operations like "scf.for %i = %c0" have % later, not at the start.
        stripped = sanitized_pattern.lstrip()
        if stripped.startswith("%") and "=" in stripped:
            # This is an SSA assignment - search after the '='.
            parts = stripped.split("=", 1)
            search_text = parts[1] if len(parts) > 1 else stripped
        else:
            # Not an SSA assignment - search the whole pattern.
            search_text = sanitized_pattern

        # Search for operation using IRIndex pattern.
        op_match = IRIndex.OPERATION_PATTERN.search(search_text)
        if not op_match:
            return None

        operation = op_match.group(1)

        # Filter out single-word identifiers that are likely not operations.
        # Real operations are either:
        # - Namespaced (contain '.'): scf.for, arith.constant
        # - Known single-word operations: call, return, yield
        if "." not in operation:
            # Check if it's a known single-word operation.
            known_single_word_ops = {"call", "return", "yield", "func"}
            if operation not in known_single_word_ops:
                # Likely not an operation (e.g., 'x', 'test', 'iter_args').
                return None

        return operation
