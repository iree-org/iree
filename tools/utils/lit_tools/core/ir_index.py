# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IR line indexing for MLIR lit tests.

Provides fast lookup of IR lines by operation type and proximity-based search.
This enables content-based matching of CHECK patterns to the IR they verify.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IRLine:
    """Represents a parsed IR line with metadata.

    Attributes:
        line_num: Absolute line number in the test case
        content: Raw line content
        operation: Operation name (e.g., "scf.for", "stream.timepoint.await")
        ssa_results: SSA value names defined by this line (e.g., ["awaited"])
        is_assignment: True if line defines SSA values (has =)
    """

    line_num: int
    content: str
    operation: str | None
    ssa_results: list[str]
    is_assignment: bool


class IRIndex:
    """Indexes IR lines for fast operation-based lookup.

    Builds three indexes:
    - ir_lines: All IR lines in order
    - by_operation: Map operation → list of IRLines
    - by_line_num: Map line number → IRLine

    Example:
        >>> lines = ['%x = arith.constant 0', 'scf.yield %x']
        >>> index = IRIndex(lines)
        >>> index.find_by_operation('arith.constant')
        [IRLine(line_num=0, operation='arith.constant', ...)]
    """

    # Pattern to match assignment lines: "%result = operation" or "%a, %b = op".
    # CRITICAL: LHS must start with % to distinguish from '=' in operation syntax.
    ASSIGNMENT_PATTERN = re.compile(r"^(\s*%[^=]*)=")

    # Pattern to extract SSA value names from LHS of assignment.
    # Matches: %name, but not %[[CAPTURE]] (FileCheck syntax).
    SSA_NAME_PATTERN = re.compile(r"%([a-zA-Z0-9_]+)")

    # Pattern to extract operation name from IR line.
    # Matches: namespace.operation (e.g., "scf.for", "stream.timepoint.await")
    # or single-word operations (e.g., "call", "return").
    # This pattern looks for the first identifier after = (for assignments)
    # or at the start of non-assignment lines.
    OPERATION_PATTERN = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\b"
    )

    def __init__(self, lines: list[str], start_line: int = 0) -> None:
        """Build index from test case content.

        Args:
            lines: Test case content split by newlines
            start_line: Starting line number offset for the test case
        """
        self.ir_lines: list[IRLine] = []
        self.by_operation: dict[str, list[IRLine]] = {}
        self.by_line_num: dict[int, IRLine] = {}

        self._build_index(lines, start_line)

    def _build_index(self, lines: list[str], start_line: int) -> None:
        """Parse lines and build indexes.

        Args:
            lines: Raw test case lines
            start_line: Line number offset
        """
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines and comments.
            if not stripped or stripped.startswith("//"):
                continue

            # Parse the line.
            ir_line = self._parse_ir_line(line, start_line + i)
            if ir_line:
                self._add_to_indexes(ir_line)

    def _parse_ir_line(self, line: str, line_num: int) -> IRLine | None:
        """Parse a single IR line.

        Args:
            line: Raw line content
            line_num: Absolute line number

        Returns:
            Parsed IRLine, or None if line cannot be parsed
        """
        stripped = line.strip()

        # Extract SSA results (if assignment).
        is_assignment = False
        ssa_results = []

        assignment_match = self.ASSIGNMENT_PATTERN.match(stripped)
        if assignment_match:
            is_assignment = True
            lhs = assignment_match.group(1)

            # Extract all SSA names from LHS.
            # CRITICAL: Do NOT filter out %0, %c0, %arg0 - all are semantic!
            for match in self.SSA_NAME_PATTERN.finditer(lhs):
                ssa_results.append(match.group(1))

        # Extract operation name.
        # For assignments, search after the '=' to avoid matching SSA names.
        # For non-assignments, search the whole line.
        operation = None
        search_text = stripped
        if is_assignment and assignment_match:
            # Search only in the RHS (after the '=').
            search_text = stripped[assignment_match.end() :]

        op_match = self.OPERATION_PATTERN.search(search_text)
        if op_match:
            operation = op_match.group(1)

        # Warn if we found SSA results but no operation - indicates parsing gap.
        if operation is None and ssa_results:
            logger.warning(
                f"Line {line_num}: SSA results found but no operation recognized: {stripped}"
            )

        # Create IRLine (even if operation is None - might be pure data flow line).
        return IRLine(
            line_num=line_num,
            content=line,
            operation=operation,
            ssa_results=ssa_results,
            is_assignment=is_assignment,
        )

    def _add_to_indexes(self, ir_line: IRLine) -> None:
        """Add IR line to all indexes.

        Args:
            ir_line: Parsed IR line to index
        """
        # Add to sequential list.
        self.ir_lines.append(ir_line)

        # Add to operation index.
        if ir_line.operation:
            if ir_line.operation not in self.by_operation:
                self.by_operation[ir_line.operation] = []
            self.by_operation[ir_line.operation].append(ir_line)

        # Add to line number index.
        self.by_line_num[ir_line.line_num] = ir_line

    def find_by_operation(
        self, operation: str, near_line: int | None = None, window: int = 50
    ) -> list[IRLine]:
        """Find IR lines with matching operation.

        Args:
            operation: Operation name (e.g., "scf.for")
            near_line: Optional line number for proximity filtering
            window: Lines before/after near_line to include (default: 50)

        Returns:
            List of matching IR lines, sorted by proximity if near_line given.
            When multiple lines are equidistant, all ties are returned.
        """
        # Get all lines with this operation.
        candidates = self.by_operation.get(operation, [])
        if not candidates:
            return []

        # If no proximity filter, return all candidates.
        if near_line is None:
            return list(candidates)

        # Filter by proximity window.
        in_window = [ir for ir in candidates if abs(ir.line_num - near_line) <= window]

        if not in_window:
            return []

        # Sort by proximity (closest first).
        # When ties occur (same distance), preserve original line order.
        in_window.sort(key=lambda ir: (abs(ir.line_num - near_line), ir.line_num))

        return in_window

    def get_line(self, line_num: int) -> IRLine | None:
        """Get IR line by line number.

        Args:
            line_num: Absolute line number

        Returns:
            IRLine at that number, or None if not found
        """
        return self.by_line_num.get(line_num)
