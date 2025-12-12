# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CHECK pattern to IR line matching for MLIR lit tests.

Matches CHECK directives to IR lines using content-based matching:
operation semantics + proximity, rather than naive positional matching.

This fixes false positives where CHECKs incorrectly match nearby IR lines
with different operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lit_tools.core.check_pattern import CheckPattern
    from lit_tools.core.ir_index import IRIndex, IRLine

logger = logging.getLogger(__name__)


@dataclass
class CheckMatch:
    """A CHECK pattern matched to an IR line.

    Attributes:
        check_pattern: The CHECK directive being matched
        ir_line: The IR line that this CHECK verifies (or None if no match)
        confidence: Confidence score (0.0-1.0) for this match
        match_reason: Human-readable explanation of why this match was chosen
    """

    check_pattern: CheckPattern
    ir_line: IRLine | None
    confidence: float
    match_reason: str


class CheckMatcher:
    """Matches CHECK patterns to IR lines using content-based matching.

    Uses operation semantics and proximity to match CHECK directives to the
    IR lines they verify, avoiding false positives from naive positional matching.

    Example:
        >>> ir_index = IRIndex(ir_lines)
        >>> check_patterns = parser.parse_file(test_lines)
        >>> matcher = CheckMatcher(ir_index, check_patterns)
        >>> matches = matcher.match_all()
    """

    def __init__(
        self,
        ir_index: IRIndex,
        check_patterns: list[CheckPattern],
        proximity_window: int = 50,
    ) -> None:
        """Initialize matcher with IR index and CHECK patterns.

        Args:
            ir_index: Index of IR lines by operation
            check_patterns: List of CHECK patterns to match
            proximity_window: Lines before/after CHECK to search (default: 50)
        """
        self.ir_index = ir_index
        self.check_patterns = check_patterns
        self.proximity_window = proximity_window

    def match_all(self) -> list[CheckMatch]:
        """Match all CHECK patterns to IR lines.

        Processes patterns sequentially to handle CHECK-NEXT and CHECK-SAME
        semantics, which depend on the previous CHECK's matched line.

        Returns:
            List of CheckMatch objects, one per CHECK pattern
        """
        matches = []
        previous_match: CheckMatch | None = None

        for check_pattern in self.check_patterns:
            match = self.match_check(check_pattern, previous_match)
            matches.append(match)

            # Track previous match for CHECK-NEXT/SAME semantics.
            # Only update if this was a successful match.
            if match.ir_line is not None:
                previous_match = match

        return matches

    def match_check(
        self, check: CheckPattern, previous_match: CheckMatch | None = None
    ) -> CheckMatch:
        """Match a single CHECK pattern to best IR line.

        Strategy:
        1. Handle CHECK-NEXT/SAME semantics using previous match
        2. If CHECK has no operation: return None match (can't verify)
        3. Find all IR lines with matching operation
        4. Score candidates by proximity and select best match

        Args:
            check: CHECK pattern to match
            previous_match: Previous CHECK's match (for -NEXT/-SAME semantics)

        Returns:
            CheckMatch with best IR line and confidence score
        """
        # Handle CHECK-NEXT: must match line immediately after previous match.
        if check.check_type.endswith("-NEXT"):
            if previous_match is None or previous_match.ir_line is None:
                return CheckMatch(
                    check_pattern=check,
                    ir_line=None,
                    confidence=0.0,
                    match_reason="CHECK-NEXT requires previous match",
                )

            # Get the next IR line after previous match.
            next_line_num = previous_match.ir_line.line_num + 1
            next_ir = self.ir_index.get_line(next_line_num)

            if next_ir is None:
                return CheckMatch(
                    check_pattern=check,
                    ir_line=None,
                    confidence=0.0,
                    match_reason=f"No IR line at {next_line_num} (after previous match)",
                )

            # TODO: Verify CHECK pattern actually matches next_ir.content.
            # For now, assume match if line exists.
            return CheckMatch(
                check_pattern=check,
                ir_line=next_ir,
                confidence=1.0,
                match_reason=f"CHECK-NEXT matched line {next_line_num}",
            )

        # Handle CHECK-SAME: must match same line as previous match.
        if check.check_type.endswith("-SAME"):
            if previous_match is None or previous_match.ir_line is None:
                return CheckMatch(
                    check_pattern=check,
                    ir_line=None,
                    confidence=0.0,
                    match_reason="CHECK-SAME requires previous match",
                )

            # Re-use the same IR line.
            return CheckMatch(
                check_pattern=check,
                ir_line=previous_match.ir_line,
                confidence=1.0,
                match_reason=f"CHECK-SAME matched same line as previous ({previous_match.ir_line.line_num})",
            )

        # For regular CHECK, CHECK-LABEL, CHECK-NOT: use operation-based matching.

        # If CHECK has no operation, we can't do content-based matching.
        if check.operation is None:
            return CheckMatch(
                check_pattern=check,
                ir_line=None,
                confidence=0.0,
                match_reason="No operation found in CHECK pattern",
            )

        # Find IR lines with matching operation.
        candidates = self.ir_index.find_by_operation(
            check.operation, near_line=check.line_num, window=self.proximity_window
        )

        if not candidates:
            return CheckMatch(
                check_pattern=check,
                ir_line=None,
                confidence=0.0,
                match_reason=f"No IR line with operation '{check.operation}' found within {self.proximity_window} lines",
            )

        # Select best candidate (IRIndex already sorted by proximity).
        best_candidate = candidates[0]

        # Compute confidence based on proximity.
        distance = abs(best_candidate.line_num - check.line_num)
        if distance == 0:
            # CHECK on same line as IR (unusual but possible).
            confidence = 0.5
            reason = (
                f"Matched '{check.operation}' on same line {best_candidate.line_num}"
            )
        elif len(candidates) == 1:
            # Only one candidate, high confidence.
            confidence = 1.0
            reason = f"Matched '{check.operation}' at line {best_candidate.line_num} (distance: {distance}, only candidate)"
        elif best_candidate == candidates[0] and len(candidates) > 1:
            # Multiple candidates, this is closest.
            confidence = 1.0
            reason = f"Matched '{check.operation}' at line {best_candidate.line_num} (distance: {distance}, closest of {len(candidates)})"
        else:
            # Fallback (shouldn't reach here given current logic).
            confidence = 0.8
            reason = f"Matched '{check.operation}' at line {best_candidate.line_num} (distance: {distance})"

        # TODO: Perform full FileCheck pattern matching (check.raw_pattern vs best_candidate.content).
        # This would validate the CHECK regex actually matches the IR line, not just the operation.
        # For now, operation + proximity matching is sufficient for the immediate bug fix.

        return CheckMatch(
            check_pattern=check,
            ir_line=best_candidate,
            confidence=confidence,
            match_reason=reason,
        )
