# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Linting infrastructure for MLIR lit test files.

Provides checkers to catch common test authoring mistakes before running
the compiler. Categorizes issues into errors (must fix), warnings (should
review), and info (suggestions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lit_tools.core.check_matcher import CheckMatcher
from lit_tools.core.check_pattern import CheckPatternParser
from lit_tools.core.ir_index import IRIndex

if TYPE_CHECKING:
    from lit_tools.core.parser import TestCase, TestFile


@dataclass
class LintIssue:
    """Represents a single lint issue found in a test case.

    Attributes:
        severity: Issue severity ("error", "warning", "info")
        code: Machine-readable issue code (e.g., "raw_ssa_identifier")
        message: Human-readable description
        line: 1-indexed line number where issue occurs
        column: 0-indexed column offset (optional)
        snippet: Code snippet showing the issue
        help: Suggested fix or explanation
        suggestions: Alternative values (optional, for semantic naming)
        context_lines: Surrounding IR lines for context (optional, for --full-json)
    """

    severity: str
    code: str
    message: str
    line: int
    column: int | None
    snippet: str
    help: str
    suggestions: list[str] | None = None
    context_lines: list[str] | None = None


def get_context_lines(
    case: TestCase, line_index: int, context_lines: int = 5
) -> list[str]:
    """Extract context lines around a specific line in the test case.

    Args:
        case: TestCase containing the lines
        line_index: 0-indexed line number within the test case content
        context_lines: Number of lines before AND after to include (default: 5)

    Returns:
        List of lines including context before and after the target line
    """
    lines = case.content.splitlines()
    start = max(0, line_index - context_lines)
    end = min(len(lines), line_index + context_lines + 1)
    return lines[start:end]


class LintChecker:
    """Base class for lint checkers.

    Each checker examines a test case and returns a list of issues found.
    Checkers should be focused on a single type of problem.
    """

    def __init__(self) -> None:
        """Initialize checker with default configuration."""
        self.context_lines = 5  # Default context window for --full-json

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check test case for issues.

        Args:
            case: TestCase to examine

        Returns:
            List of LintIssue objects (empty if no issues found)
        """
        raise NotImplementedError


class RawSSAIdentifierChecker(LintChecker):
    """Detects raw SSA identifiers in CHECK lines that should use captures.

    Examples of problems:
    - // CHECK: %0 = op           (should be %[[RESULT:.+]] = op)
    - // CHECK: %c123 = constant  (should be %[[SIZE:.+]] = constant)
    - // CHECK: arith.addi %arg0  (should be %[[INPUT:.+]])
    - // CHECK: call @foo(%buf)   (should be %[[BUFFER:.+]])

    All raw SSA identifiers should be captures for robust tests. Use NOLINT
    on a preceding line to suppress for crash reproducers or special cases.

    NOTE: MLIR constants (%c0, %c123, %c4_i32) are temporarily exempted due to
    widespread usage, but this is an ANTI-PATTERN. Constants like %c123 are not
    semantically meaningful - they don't tell you if it's a fill pattern, offset,
    size, or dimension. Prefer semantic captures like %[[OFFSET:.+]] or
    %[[FILL_PATTERN:.+]] that describe purpose, not value.
    """

    # Matches the CHECK directive and extracts everything after the colon.
    CHECK_LINE_PATTERN = re.compile(r"//\s*CHECK[^:]*:\s*(.*)")

    # Matches any raw SSA value: %name, %arg0, %0, %c123, etc.
    # Uses negative lookahead to exclude captures: %[[NAME]] or %[[NAME:.+]].
    RAW_SSA_PATTERN = re.compile(r"%(?!\[\[)[a-zA-Z0-9_][a-zA-Z0-9_$.]*")

    # MLIR constant pattern: %c0, %c123, %c4_i32, %cst, %cst_0, etc.
    # Temporarily exempted (anti-pattern but widespread).
    MLIR_CONSTANT_PATTERN = re.compile(r"^%c(st)?(\d+|st)(_[a-zA-Z0-9_]+)?$")

    # NOLINT comment pattern (case-insensitive).
    NOLINT_PATTERN = re.compile(r"//\s*NOLINT", re.IGNORECASE)

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for raw SSA identifiers in CHECK lines.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()
        nolint_active = False

        for i, line in enumerate(lines):
            # Check for NOLINT on any line (suppresses rest of case).
            if self.NOLINT_PATTERN.search(line):
                nolint_active = True
                continue

            if nolint_active:
                continue

            # Only check CHECK lines.
            check_match = self.CHECK_LINE_PATTERN.search(line)
            if not check_match:
                continue

            # Get the content after CHECK: and find the offset where it starts.
            check_content = check_match.group(1)
            content_start = check_match.start(1)

            # Find all raw SSA identifiers in the CHECK content.
            for ssa_match in self.RAW_SSA_PATTERN.finditer(check_content):
                ssa_name = ssa_match.group(0)

                # Skip MLIR constants (anti-pattern but widespread).
                # %c0, %c123, %c4_i32, %cst, %cst_0, etc.
                if self.MLIR_CONSTANT_PATTERN.match(ssa_name):
                    continue

                # Column is relative to the line start.
                col = content_start + ssa_match.start()

                # Provide targeted help for argument patterns vs general SSA.
                if ssa_name.startswith("%arg"):
                    help_text = (
                        f"Use %[[NAME:[^:]+]] instead of {ssa_name}. "
                        "For function arguments in CHECK-SAME, use [^:]+ (not .+) "
                        "to stop at the type specifier. Example: "
                        "// CHECK-SAME: %[[INPUT:[^:]+]]: tensor<4xf32>"
                    )
                else:
                    help_text = (
                        f"Use %[[NAME:.+]] instead of {ssa_name}. "
                        f"Raw SSA values like {ssa_name} are fragile - they change when "
                        "unrelated code is added/removed. Named captures verify data flow "
                        "and make tests resilient to IR structure changes."
                    )

                issues.append(
                    LintIssue(
                        severity="error",
                        code="raw_ssa_identifier",
                        message="raw SSA identifier in CHECK line",
                        line=case.start_line + i,
                        column=col,
                        snippet=line.strip(),
                        help=help_text,
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )

        return issues


class ExcessiveWildcardChecker(LintChecker):
    """Detects excessive use of {{.+}} wildcards in CHECK lines.

    Wildcards are appropriate for structural IR (device affinities, types the
    pass doesn't touch), but excessive use masks data flow verification.

    Triggers warning when:
    - More than 2 {{.+}} wildcards in a CHECK line (default threshold)
    - More than 5 {{.+}} wildcards in scf.for operations (special case for loop bounds/step)
    - Suggests capturing operands explicitly to verify data flow
    """

    WILDCARD_PATTERN = re.compile(r"\{\{\.?\+\}\}")

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for excessive use of {{.+}} wildcards.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        for i, line in enumerate(lines):
            # Only check CHECK lines.
            if "CHECK" not in line or "//" not in line:
                continue

            # Extract CHECK part of line.
            check_part = line.split("//", 1)[1] if "//" in line else line
            wildcards = self.WILDCARD_PATTERN.findall(check_part)

            # Special case: scf.for operations have many structural operands.
            # (loop bounds, step) that aren't critical to verify in tests.
            threshold = 5 if "scf.for" in check_part else 2

            if len(wildcards) > threshold:
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="excessive_wildcards",
                        message=f"excessive wildcards ({len(wildcards)} found)",
                        line=case.start_line + i,
                        column=None,
                        snippet=line.strip(),
                        help=(
                            "Capture operands explicitly to verify data flow. "
                            "Wildcards {{.+}} are appropriate for structural IR (types, attributes) "
                            "that your pass doesn't touch, but overuse masks the actual transformations "
                            "being verified. Explicit captures ensure the right values flow through operations. "
                            "(Ignore if operation has many structural attributes/types that the pass doesn't modify.)"
                        ),
                    )
                )

        return issues


class NonSemanticCaptureChecker(LintChecker):
    """Detects non-semantic capture names in CHECK lines.

    Flags capture names that just copy the SSA value name instead of describing
    semantic purpose. Constants and arguments have syntactic names that don't
    convey meaning.

    Examples of bad names:
    - %[[C0]] - named after constant (use %[[OFFSET]], %[[SIZE]], etc.)
    - %[[C123_I32]] - named after constant value and type
    - %[[ARG0]] - named after argument (use %[[INPUT]], %[[OPERAND]], etc.)

    Examples of good names:
    - %[[BUFFER_OFFSET]] - describes semantic purpose
    - %[[INPUT_SIZE]] - describes what the value represents
    """

    # Patterns for non-semantic names (ONLY match definitions with : pattern).
    # Matches: %[[C0:.+]], %[[C123:.*]], %[[C0_I32:[a-z]+]], etc.
    # Does NOT match usage/references: %[[C0]], %[[ARG1]]
    # Use non-greedy .+? to handle character classes like [a-zA-Z0-9]+
    CONSTANT_NAME_PATTERN = re.compile(r"%\[\[C\d+(?:_I\d+)?:.+?\]\]", re.DOTALL)
    # Matches: %[[ARG0:.+]], %[[ARG1:.*]], %[[ARG1:[a-z]+]], etc.
    # Does NOT match usage/references: %[[ARG0]], %[[ARG1]]
    ARGUMENT_NAME_PATTERN = re.compile(r"%\[\[ARG\d+:.+?\]\]", re.DOTALL)

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for non-semantic capture names.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        for i, line in enumerate(lines):
            # Only check CHECK lines.
            if "CHECK" not in line or "//" not in line:
                continue

            # Check for constant-based names.
            for match in self.CONSTANT_NAME_PATTERN.finditer(line):
                col = match.start()
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="non_semantic_capture",
                        message="capture name based on constant value",
                        line=case.start_line + i,
                        column=col,
                        snippet=line.strip(),
                        help=(
                            "Use semantic name like %[[OFFSET]] or %[[SIZE]] instead. "
                            "Names like %[[C0]] just copy the constant's syntactic name without "
                            "conveying meaning. If the constant value changes, the capture name "
                            "becomes misleading. Semantic names describe purpose and remain accurate "
                            "across code changes. "
                            "(Ignore if the constant value IS the semantic meaning, e.g., %[[C0]] for zero-initialization pattern.)"
                        ),
                        suggestions=["OFFSET", "SIZE", "CONSTANT", "INIT_VALUE"],
                    )
                )

            # Check for argument-based names.
            for match in self.ARGUMENT_NAME_PATTERN.finditer(line):
                col = match.start()
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="non_semantic_capture",
                        message="capture name based on argument name",
                        line=case.start_line + i,
                        column=col,
                        snippet=line.strip(),
                        help=(
                            "Use semantic name like %[[INPUT]] or %[[OPERAND]] instead. "
                            "Names like %[[ARG0]] just copy the argument position without conveying "
                            "what the value represents. If arguments are reordered, the name becomes "
                            "wrong. Semantic names describe the value's role (INPUT, LHS, BUFFER) and "
                            "remain meaningful across refactoring. "
                            "(Ignore if testing argument forwarding where position IS the semantic meaning.)"
                        ),
                        suggestions=["INPUT", "OPERAND", "LHS", "RHS"],
                    )
                )

        return issues


class ZeroCheckLinesChecker(LintChecker):
    """Detects test cases with IR but no CHECK lines.

    A test with IR but no verification is useless - it only verifies that
    the IR parses, not that any transformation occurred.
    """

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for test cases with no CHECK lines.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        if case.check_count == 0:
            # Check if there's actual IR content (not just comments/whitespace).
            lines = case.content.splitlines()
            has_ir = any(
                line.strip()
                and not line.strip().startswith("//")
                and not line.strip().startswith("#")
                for line in lines
            )

            if has_ir:
                return [
                    LintIssue(
                        severity="error",
                        code="zero_check_lines",
                        message="test case has IR but no CHECK lines",
                        line=case.start_line,
                        column=None,
                        snippet=f"Test case {case.number}",
                        help=(
                            "Add CHECK lines to verify transformation behavior. "
                            "A test with IR but no verification only confirms the IR parses, "
                            "not that your pass actually transforms it correctly. Without CHECK "
                            "lines, the test provides no value - bugs slip through undetected. "
                            "(Ignore only if this is explicitly a parse-only syntax test - add comment explaining.)"
                        ),
                    )
                ]

        return []


class NonInterleavedChecksChecker(LintChecker):
    """Detects when all CHECKs are clustered instead of interleaved with IR.

    Interleaving CHECKs with IR:
    - Proves where transformations occur
    - Anchors checks to specific locations
    - Makes test intent clearer

    This checker triggers when >80% of CHECKs appear in first or last 20% of lines.
    """

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for non-interleaved CHECKs.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        if case.check_count < 3 or case.line_count < 10:
            # Too small to meaningfully check.
            return []

        lines = case.content.splitlines()
        check_lines = []
        for i, line in enumerate(lines):
            if "CHECK" in line and "//" in line:
                check_lines.append(i)

        if not check_lines:
            return []

        # Calculate clustering.
        total_lines = len(lines)

        # CHECKs are non-interleaved if they're all in first or last 20%.
        threshold = int(total_lines * 0.2)

        all_at_start = all(i < threshold for i in check_lines)
        all_at_end = all(i >= total_lines - threshold for i in check_lines)

        if all_at_start or all_at_end:
            location = "start" if all_at_start else "end"
            return [
                LintIssue(
                    severity="warning",
                    code="non_interleaved_checks",
                    message=f"all CHECKs clustered at {location} of test",
                    line=case.start_line,
                    column=None,
                    snippet=f"Lines {case.start_line}-{case.end_line}",
                    help=(
                        "Interleave CHECKs near corresponding IR for better verification. "
                        "When CHECKs are clustered away from the IR they verify, it's harder to "
                        "understand what's being tested and maintain the test. "
                        "(Ignore if test structure requires grouped checks - e.g., all attributes at top, "
                        "or final state verification at bottom.)"
                    ),
                )
            ]

        return []


class TodoWithoutExplanationChecker(LintChecker):
    """Detects TODO/FIXME/NOTE comments without explanation.

    Bare TODO markers provide no context for future readers and will likely
    be ignored. All TODO/FIXME/NOTE comments must explain WHY and WHAT needs
    to be done, including relevant issue numbers or blockers.

    Examples of violations:
    - // TODO
    - // TODO: fix this
    - // FIXME: broken

    Valid patterns:
    - // TODO(#1234): Add support for scf.while after API lands.
    - // NOTE: Cannot eliminate - public funcs can't return timepoints.
    - // FIXME: Dominance check fails for block args in loop headers.
    """

    # Pattern: TODO/FIXME/NOTE with minimal or no explanation.
    BARE_TODO_PATTERN = re.compile(
        r"^\s*//\s*(TODO|FIXME|NOTE)\s*(:?\s*)?$", re.IGNORECASE
    )
    WEAK_TODO_PATTERN = re.compile(
        r"^\s*//\s*(TODO|FIXME|NOTE)\s*:\s*(\w+\s+){0,3}\w+\s*$", re.IGNORECASE
    )

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for TODO/FIXME comments without substantive explanation.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        for i, line in enumerate(lines):
            # Check for bare TODO/FIXME/NOTE.
            if self.BARE_TODO_PATTERN.match(line):
                marker = self.BARE_TODO_PATTERN.match(line).group(1)
                issues.append(
                    LintIssue(
                        severity="error",
                        code="todo_without_explanation",
                        message=f"{marker} comment lacks explanation",
                        line=case.start_line + i,
                        column=None,
                        snippet=line.strip(),
                        help=(
                            f"Add explanation to {marker} comment. "
                            f"Bare {marker} markers provide no context and will be ignored. "
                            "Explain WHY this needs attention, WHAT needs to be done, and any "
                            "relevant issue numbers or blockers. "
                            f"Example: // {marker}(#1234): Add X after Y lands."
                        ),
                    )
                )
            # Check for weak explanation (1-3 words).
            elif self.WEAK_TODO_PATTERN.match(line):
                marker = self.WEAK_TODO_PATTERN.match(line).group(1)
                issues.append(
                    LintIssue(
                        severity="error",
                        code="todo_without_explanation",
                        message=f"{marker} comment has insufficient explanation",
                        line=case.start_line + i,
                        column=None,
                        snippet=line.strip(),
                        help=(
                            f"Expand {marker} explanation. "
                            "Brief markers like 'fix this' or 'broken' provide no actionable "
                            "context. Explain the issue, blockers, or planned work with enough "
                            "detail for future readers to understand and act on it."
                        ),
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )

        return issues


class CHECKNOTWithoutAnchorChecker(LintChecker):
    """Detects CHECK-NOT without surrounding positive CHECK anchors.

    CHECK-NOT should be bracketed by positive CHECK lines to define the scope
    where something shouldn't appear. Without anchors, CHECK-NOT matches too
    broadly (entire file) or too narrowly (just next line), causing fragile
    tests.

    Examples of violations:
    - CHECK-LABEL: @test
      CHECK-NOT: await
      (no positive CHECK after - scope unclear)

    Valid patterns:
    - CHECK-LABEL: @test
      CHECK: scf.if
      CHECK-NOT: await
      CHECK: scf.yield
      (clearly checks between scf.if and scf.yield)
    """

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for CHECK-NOT without anchoring CHECKs.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        for i, line in enumerate(lines):
            # Only match actual CHECK-NOT directives, not comments mentioning CHECK-NOT.
            if not re.search(r"//\s*CHECK-NOT:", line):
                continue

            # Look backward for positive CHECK (within 10 lines).
            has_check_before = False
            for j in range(max(0, i - 10), i):
                check_line = lines[j]
                if re.search(r"//\s*CHECK(-LABEL)?:", check_line):
                    has_check_before = True
                    break

            # Look forward for positive CHECK (within 10 lines).
            has_check_after = False
            for j in range(i + 1, min(len(lines), i + 11)):
                check_line = lines[j]
                if re.search(r"//\s*CHECK(-LABEL)?:", check_line):
                    has_check_after = True
                    break

            if not (has_check_before and has_check_after):
                missing = []
                if not has_check_before:
                    missing.append("before")
                if not has_check_after:
                    missing.append("after")

                issues.append(
                    LintIssue(
                        severity="error",
                        code="check_not_without_anchor",
                        message=f"CHECK-NOT missing positive CHECK {' and '.join(missing)}",
                        line=case.start_line + i,
                        column=None,
                        snippet=line.strip(),
                        help=(
                            "Bracket CHECK-NOT with positive CHECK lines to define scope. "
                            "Without anchors, CHECK-NOT matches too broadly (entire file) or "
                            "too narrowly (just next line). Add CHECK/CHECK-LABEL before and "
                            "CHECK after to clearly define where the pattern should NOT appear. "
                            "Example: CHECK: scf.if / CHECK-NOT: await / CHECK: scf.yield "
                            "(Ignore if verifying file-level invariant where pattern should not appear anywhere.)"
                        ),
                    )
                )

        return issues


class MatchedCaptureNameChecker(LintChecker):
    """Detects mismatches between IR SSA names and CHECK capture names.

    When IR contains semantic SSA names like %transient_size, CHECK captures
    should match: %[[TRANSIENT_SIZE:.+]] not %[[SIZE:.+]]. Mismatched names
    break the connection between IR and verification, making it unclear what's
    being tested.

    CRITICAL requirement per CLAUDE.md: "Keep SSA variable names and CHECK
    capture names identical (e.g., %transient_size matches %[[TRANSIENT_SIZE:.+]])".

    Note: This is a warning (not error) initially to allow variance during
    dialect conversion or when abbreviations are intentional. Scan existing
    tests to assess strictness before promoting to error.

    Examples:
      IR: %transient_size = ...
      BAD:  // CHECK: %[[SIZE:.+]] =
      GOOD: // CHECK: %[[TRANSIENT_SIZE:.+]] =

      IR: %slice_ready = stream.timepoint.await
      BAD:  // CHECK: %[[READY:.+]] = stream.timepoint.await
      GOOD: // CHECK: %[[SLICE_READY:.+]] = stream.timepoint.await
    """

    # Extract semantic SSA names from IR (not %0, %c123, %arg0).
    # This pattern finds individual SSA names (used for extraction from LHS).
    SSA_NAME_PATTERN = re.compile(r"%([a-z_][a-z0-9_]+)")

    # Pattern to find assignment lines (to extract LHS).
    ASSIGNMENT_PATTERN = re.compile(r"^([^=]*)=")

    # Extract CHECK capture names (handle FileCheck tuple syntax like %[[NAME:.+]]:2).
    CHECK_CAPTURE_PATTERN = re.compile(r"%\[\[([A-Z_]+):.+?\]\](?::\d+)?")

    def _extract_ir_names_from_line(self, line: str) -> list[str]:
        """Extract SSA names from IR line in left-to-right order.

        Args:
            line: Single line of IR code

        Returns:
            List of semantic SSA names in order of appearance
        """
        assignment_match = self.ASSIGNMENT_PATTERN.match(line.strip())
        if not assignment_match:
            return []

        lhs = assignment_match.group(1)
        names = []
        for name_match in self.SSA_NAME_PATTERN.finditer(lhs):
            name = name_match.group(1)
            # Skip non-semantic names (constants and arguments).
            if not (name.startswith("c") and name[1:].isdigit()) and not (
                name.startswith("arg") and name[3:].isdigit()
            ):
                names.append(name)
        return names

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for mismatched IR SSA names and CHECK captures.

        Uses content-based matching (operation + proximity) to match CHECK
        patterns to the IR lines they verify, avoiding false positives from
        naive positional matching.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        # Build IR index for content-based matching.
        ir_index = IRIndex(lines)

        # Parse CHECK patterns from file.
        parser = CheckPatternParser()
        check_patterns = parser.parse_file(lines)

        if not check_patterns:
            return []  # No CHECK patterns to validate.

        # Match CHECK patterns to IR lines using content-based matching.
        matcher = CheckMatcher(ir_index, check_patterns)
        matches = matcher.match_all()

        # Validate capture names for each successful match.
        for match in matches:
            # Skip if no IR line matched (no operation, or not found).
            if match.ir_line is None:
                continue

            # Skip if CHECK has no captures.
            check_captures = list(
                self.CHECK_CAPTURE_PATTERN.finditer(match.check_pattern.raw_pattern)
            )
            if not check_captures:
                continue

            # Extract IR names from matched line.
            target_ir_names = self._extract_ir_names_from_line(match.ir_line.content)

            # Positional pairing: If capture count matches IR name count, zip them.
            # This handles tuple assignments correctly.
            if len(check_captures) == len(target_ir_names):
                for capture_match, ir_name in zip(
                    check_captures, target_ir_names, strict=False
                ):
                    capture_name = capture_match.group(1)
                    capture_normalized = capture_name.lower()
                    ir_normalized = ir_name.lower()
                    ir_upper = ir_name.upper()

                    # Exact match after case normalization - no issue.
                    # CHECK captures use SCREAMING_SNAKE_CASE, IR uses snake_case.
                    if capture_normalized == ir_normalized:
                        continue

                    # Check for intentional variance (semantic naming, abbreviations).
                    if self._is_acceptable_variance(capture_normalized, ir_normalized):
                        continue

                    # Flag mismatch.
                    issues.append(
                        self._create_mismatch_issue(
                            capture_name,
                            ir_name,
                            ir_upper,
                            case,
                            match.check_pattern.line_num,
                            capture_match,
                        )
                    )

        return issues

    def _is_acceptable_variance(
        self, capture_normalized: str, ir_normalized: str
    ) -> bool:
        """Check if capture/IR name difference is acceptable variance.

        Args:
            capture_normalized: CHECK capture name (lowercased)
            ir_normalized: IR SSA name (lowercased)

        Returns:
            True if variance is intentional/acceptable, False if likely error
        """
        # Split both names on underscores to get word components.
        capture_parts = set(capture_normalized.split("_"))
        ir_parts = set(ir_normalized.split("_"))

        # If all IR parts are contained in capture parts, it's likely intentional.
        # Example: IR has "tp", CHECK has "tp_then" (adds semantic context).
        if ir_parts.issubset(capture_parts):
            return True  # Intentional semantic naming.

        # If all capture parts are contained in IR parts, might be abbreviation.
        # Example: IR has "transient_size", CHECK has "size" (abbreviation).
        # Allow if missing < 2 words (minor abbreviation acceptable).
        # Return True for minor abbreviations, False otherwise.
        return (
            capture_parts.issubset(ir_parts) and len(ir_parts) - len(capture_parts) < 2
        )

    def _create_mismatch_issue(
        self,
        capture_name: str,
        ir_name: str,
        ir_upper: str,
        case: TestCase,
        check_line_idx: int,
        capture_match: re.Match,
    ) -> LintIssue:
        """Create a LintIssue for mismatched capture name.

        Args:
            capture_name: CHECK capture name (original case)
            ir_name: IR SSA name (original case)
            ir_upper: IR name uppercased
            case: TestCase being checked
            check_line_idx: Line index of CHECK comment
            capture_match: Regex match for the capture

        Returns:
            LintIssue describing the mismatch
        """
        # Normalize both names for comparison.
        capture_normalized = capture_name.lower()
        ir_normalized = ir_name.lower()

        # Calculate edit distance to detect typos.
        def edit_distance(s1: str, s2: str) -> int:
            """Simple Levenshtein distance for typo detection."""
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            prev_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                curr_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = prev_row[j + 1] + 1
                    deletions = curr_row[j] + 1
                    substitutions = prev_row[j] + (c1 != c2)
                    curr_row.append(min(insertions, deletions, substitutions))
                prev_row = curr_row
            return prev_row[-1]

        distance = edit_distance(capture_normalized, ir_normalized)

        # Construct message based on similarity.
        if distance <= 2 and len(capture_normalized) > 2 and len(ir_normalized) > 2:
            message = f"CHECK capture %[[{capture_name}]] may not match IR %{ir_name} (edit distance: {distance})"
            help_text = (
                f"Use %[[{ir_upper}:.+]] to match IR semantic naming. "
                f"IR has %{ir_name} but CHECK captures %[[{capture_name}]]. "
                f"Names are very similar (edit distance {distance}) - possible typo? "
                "Matched names maintain clarity and catch when IR changes. "
                "(Ignore if intentional variance for dialect conversion or abbreviation.)"
            )
        else:
            message = f"CHECK capture %[[{capture_name}]] may not match IR %{ir_name}"
            help_text = (
                f"Use %[[{ir_upper}:.+]] to match IR semantic naming. "
                f"IR has %{ir_name} but CHECK captures %[[{capture_name}]]. "
                "Matched names maintain clarity and catch when IR changes. "
                "Variance is acceptable for dialect conversion or intentional "
                "abbreviations, but exact matches are preferred. "
                "(Ignore if intentional - e.g., %stream_resource → %[[TENSOR]] during dialect lowering.)"
            )

        return LintIssue(
            severity="warning",
            code="mismatched_capture_name",
            message=message,
            line=case.start_line + check_line_idx,
            column=capture_match.start(),
            snippet=case.content.splitlines()[check_line_idx].strip(),
            help=help_text,
            suggestions=[ir_upper],
            context_lines=get_context_lines(case, check_line_idx, self.context_lines),
        )


class WildcardInTerminatorChecker(LintChecker):
    """Detects wildcards in terminator operations.

    Terminator operations (scf.yield, util.return, func.return, cf.br) define
    data flow out of regions/blocks - their operands are what you're testing.
    Using wildcards here means you're not actually verifying the transformation.

    Examples:
      BAD:  // CHECK: scf.yield {{.+}}, {{.+}}
      GOOD: // CHECK: scf.yield %[[RESULT]], %[[TP]]

      BAD:  // CHECK: util.return {{.+}}
      GOOD: // CHECK: util.return %[[AWAITED]]

      BAD:  // CHECK: cf.br ^bb1({{.+}} : i32)
      GOOD: // CHECK: cf.br ^bb1(%[[VAL]] : i32)
    """

    # Common MLIR terminator operations.
    TERMINATORS = [
        "scf.yield",
        "util.return",
        "func.return",
        "cf.br",
        "cf.cond_br",
        "cf.switch",
    ]

    # Wildcard pattern.
    WILDCARD_PATTERN = re.compile(r"\{\{\.?\+\}\}")

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for wildcards in terminator operations.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        for i, line in enumerate(lines):
            if "//" not in line or "CHECK" not in line:
                continue

            # Check if line contains terminator and wildcard.
            for terminator in self.TERMINATORS:
                if terminator in line and self.WILDCARD_PATTERN.search(line):
                    issues.append(
                        LintIssue(
                            severity="warning",
                            code="wildcard_in_terminator",
                            message=f"terminator {terminator} uses wildcards for operands",
                            line=case.start_line + i,
                            column=None,
                            snippet=line.strip(),
                            help=(
                                f"Capture {terminator} operands explicitly to verify data flow. "
                                "Terminators define data flow out of regions/blocks - their "
                                "operands are what transformations modify. Wildcards here mean "
                                "you're not verifying the transformation worked. Use named "
                                "captures like %[[RESULT]] instead of {{.+}}. "
                                "(Ignore if terminator forwards arguments unchanged and test focuses elsewhere.)"
                            ),
                            context_lines=get_context_lines(
                                case, i, self.context_lines
                            ),
                        )
                    )
                    break  # Only report once per line.

        return issues


class CHECKWithoutLabelContextChecker(LintChecker):
    """Detects CHECK lines appearing before any CHECK-LABEL.

    CHECK lines should follow a CHECK-LABEL that establishes which function/
    block is being verified. Without CHECK-LABEL, FileCheck may match lines
    from the wrong function, causing tests to pass for the wrong reasons.

    Example violation:
      // CHECK: %[[FOO:.+]] = some.op
      util.func @test() { ... }

    Valid pattern:
      // CHECK-LABEL: @test
      util.func @test() {
      // CHECK: %[[FOO:.+]] = some.op
    """

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for CHECK lines before first CHECK-LABEL.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        has_label = False
        for i, line in enumerate(lines):
            if "//" not in line:
                continue

            # Track if we've seen a CHECK-LABEL.
            if "CHECK-LABEL" in line:
                has_label = True
                continue

            # Check for CHECK (not CHECK-LABEL, CHECK-NOT) before any LABEL.
            if "CHECK:" in line and not has_label:
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="check_without_label_context",
                        message="CHECK appears before CHECK-LABEL",
                        line=case.start_line + i,
                        column=None,
                        snippet=line.strip(),
                        help=(
                            "Add CHECK-LABEL before CHECK lines to establish context. "
                            "Without CHECK-LABEL, FileCheck may match from the wrong function, "
                            "causing tests to pass incorrectly. Start each test with "
                            "CHECK-LABEL: @functionName to anchor verification. "
                            "(Ignore if single-case file testing module-level attributes or file-level ops.)"
                        ),
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )
                # Only report once (first occurrence).
                break

        return issues


class UnusedCaptureChecker(LintChecker):
    """Detects captured values that are never referenced.

    Captured values like %[[FOO:.+]] should be referenced later in CHECK lines
    or subsequent IR. A capture that's never used suggests incomplete test
    coverage - you captured it intending to verify something but never did.

    Note: Single-use captures for documentation are acceptable in CHECK-SAME.

    Examples:
      BAD:
        // CHECK: %[[UNUSED:.+]] = stream.async.execute
        // CHECK: util.return %[[OTHER]]

      GOOD:
        // CHECK: %[[RESULT:.+]] = stream.async.execute
        // CHECK: util.return %[[RESULT]]

      ACCEPTABLE (used in CHECK-SAME):
        // CHECK: %[[TP:.+]] = stream.async.execute
        // CHECK-SAME: await(%[[TP]]) =>
    """

    # Capture with regex definition: %[[NAME:.+]].
    CAPTURE_DEF_PATTERN = re.compile(r"%\[\[([A-Z_]+):.+?\]\]")

    # Reference to capture: %[[NAME]] (no colon).
    CAPTURE_REF_PATTERN = re.compile(r"%\[\[([A-Z_]+)\]\]")

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for unused captures.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        # Extract all capture definitions with line numbers.
        captures = {}  # {name: line_num}
        for i, line in enumerate(lines):
            if "//" not in line or "CHECK" not in line:
                continue
            for match in self.CAPTURE_DEF_PATTERN.finditer(line):
                capture_name = match.group(1)
                if capture_name not in captures:
                    captures[capture_name] = i

        if not captures:
            return []

        # Find all references to captures.
        references = set()
        for line in lines:
            if "//" not in line or "CHECK" not in line:
                continue
            for match in self.CAPTURE_REF_PATTERN.finditer(line):
                references.add(match.group(1))

        # Report unused captures.
        for capture_name, line_num in captures.items():
            if capture_name not in references:
                snippet = lines[line_num].strip()
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="unused_capture",
                        message=f"capture %[[{capture_name}]] is never referenced",
                        line=case.start_line + line_num,
                        column=None,
                        snippet=snippet,
                        help=(
                            f"Reference %[[{capture_name}]] in later CHECK lines or remove the capture. "
                            "Unused captures suggest incomplete test coverage. "
                            "(Ignore for function parameters, block arguments, or other documentation captures.)"
                        ),
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )

        return issues


class FileCheckDirectiveValidator(LintChecker):
    """Detects typos and malformed FileCheck directives in CHECK lines.

    FileCheck directives must be correctly spelled and include colons. Common
    typos like CHECK-DAD (should be CHECK-DAG) or CHECK-SMAE (should be
    CHECK-SAME) cause tests to silently fail - FileCheck ignores malformed
    directives, so the test passes without actually checking anything.

    Valid directives:
    - CHECK, CHECK-LABEL, CHECK-SAME, CHECK-DAG, CHECK-NOT, CHECK-NEXT,
      CHECK-COUNT, CHECK-EMPTY

    Common errors:
    - Typos: CHECK-DAD, CHECK-SMAE, CHECK-LABLE, CHECK-NEX
    - Missing colon: CHECK func.func (should be CHECK: func.func)
    - Malformed: CHEK, CHECKK, CHECk

    Examples:
      BAD:  // CHECK-DAD: %[[FOO]]    (typo, should be CHECK-DAG)
      BAD:  // CHECK-SMAE: foo(       (typo, should be CHECK-SAME)
      BAD:  // CHECK func.func         (missing colon)
      GOOD: // CHECK-DAG: %[[FOO]]
      GOOD: // CHECK: func.func
    """

    # Valid FileCheck directive suffixes (after CHECK-).
    VALID_DIRECTIVES = {
        "CHECK",
        "CHECK-LABEL",
        "CHECK-SAME",
        "CHECK-DAG",
        "CHECK-NOT",
        "CHECK-NEXT",
        "CHECK-COUNT",
        "CHECK-EMPTY",
    }

    # Pattern to extract CHECK directive from line.
    # Matches: // CHECK, // CHECK-LABEL:, //CHECK-DAG, etc.
    # Captures the directive part (CHECK-LABEL) and whether colon is present.
    DIRECTIVE_PATTERN = re.compile(r"//\s*(CHECK(?:-[A-Z]+)?)\s*(:?)", re.IGNORECASE)

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for malformed FileCheck directives.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        for i, line in enumerate(lines):
            # Only check lines that look like CHECK directives.
            if "//" not in line or "CHECK" not in line.upper():
                continue

            # Extract directive using pattern.
            match = self.DIRECTIVE_PATTERN.search(line)
            if not match:
                continue

            directive = match.group(1).upper()
            has_colon = match.group(2) == ":"

            # Check if directive is valid.
            if directive not in self.VALID_DIRECTIVES:
                # Find closest valid directive for suggestion.
                suggestion = self._find_closest_directive(directive)
                issues.append(
                    LintIssue(
                        severity="error",
                        code="invalid_filecheck_directive",
                        message=f"invalid FileCheck directive '{match.group(1)}'",
                        line=case.start_line + i,
                        column=match.start(1),
                        snippet=line.strip(),
                        help=(
                            f"Use '{suggestion}' instead of '{match.group(1)}'. "
                            "Invalid directives are silently ignored by FileCheck, "
                            "causing tests to pass without actually verifying anything. "
                            f"Valid directives: {', '.join(sorted(self.VALID_DIRECTIVES))}. "
                            "(Ignore if this is a custom FileCheck prefix defined in lit.cfg.py - "
                            "rare in IREE codebase.)"
                        ),
                        suggestions=[suggestion],
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )
                continue

            # Check for missing colon after directive.
            # Colon is required unless:
            # 1. Line ends immediately (standalone CHECK-NOT for example)
            # 2. Next character is whitespace then end of line
            remaining = line[match.end() :].lstrip()
            if not has_colon and remaining and not remaining.startswith("{{"):
                # There's content after directive but no colon.
                issues.append(
                    LintIssue(
                        severity="error",
                        code="missing_colon_in_directive",
                        message=f"FileCheck directive '{directive}' missing colon",
                        line=case.start_line + i,
                        column=match.end(1),
                        snippet=line.strip(),
                        help=(
                            f"Add colon after '{directive}': should be '{directive}: {remaining[:20]}...'. "
                            "FileCheck requires a colon to separate the directive from the pattern. "
                            "Without it, FileCheck may misinterpret the pattern or ignore the check entirely. "
                            "(Ignore if directive is intentionally standalone - e.g., CHECK-NOT at end of region.)"
                        ),
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )

        return issues

    def _find_closest_directive(self, invalid_directive: str) -> str:
        """Find the closest valid directive using Levenshtein distance.

        Args:
            invalid_directive: The malformed directive (e.g., "CHECK-DAD")

        Returns:
            Closest valid directive (e.g., "CHECK-DAG")
        """

        def levenshtein_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein edit distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)

            prev_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                curr_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = prev_row[j + 1] + 1
                    deletions = curr_row[j] + 1
                    substitutions = prev_row[j] + (c1 != c2)
                    curr_row.append(min(insertions, deletions, substitutions))
                prev_row = curr_row
            return prev_row[-1]

        # Find directive with minimum edit distance.
        min_distance = float("inf")
        closest = "CHECK"

        for valid in self.VALID_DIRECTIVES:
            distance = levenshtein_distance(invalid_directive, valid)
            if distance < min_distance:
                min_distance = distance
                closest = valid

        return closest


def _get_ir_lines(case: TestCase) -> list[str]:
    """Extract IR lines from test case (non-comments, non-CHECK).

    Args:
        case: TestCase to extract from

    Returns:
        List of IR line strings
    """
    lines = case.content.splitlines()
    ir_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip comment lines and empty lines.
        if not stripped or stripped.startswith("//"):
            continue
        ir_lines.append(line)
    return ir_lines


def _extract_label_text(pattern: str, function_ops: list[str]) -> str | None:
    """Extract the bare label from CHECK-LABEL pattern.

    Examples:
        "@function_name" -> "@function_name"
        "@test(" -> "@test"
        "func.func @test" -> None (has prefix, skip)

    Args:
        pattern: The CHECK-LABEL pattern text
        function_ops: List of known function operation prefixes

    Returns:
        Bare label text if pattern is bare, None if has prefix
    """
    pattern = pattern.strip()

    # Check if pattern has operation prefix.
    if any(pattern.startswith(op) for op in function_ops):
        return None  # Has operation prefix, skip check.

    # Extract bare label starting with @.
    if pattern.startswith("@"):
        # Extract just the symbol name (up to whitespace, paren, or end).
        match = re.match(r"(@[a-zA-Z0-9_.$-]+)", pattern)
        return match.group(1) if match else None

    return None


def _find_label_matches(ir_lines: list[str], label: str) -> list[tuple[int, str]]:
    """Find all IR lines containing the label text.

    Args:
        ir_lines: List of IR lines (non-comment, non-CHECK)
        label: Label text to search for (e.g., "@my_function")

    Returns:
        List of (line_index, full_line_text) for each match
    """
    # Escape special regex chars in label.
    escaped = re.escape(label)
    # Match label as separate token:
    # - Not preceded by alphanumeric/underscore (to avoid matching @test in @test_helper)
    # - Not followed by alphanumeric/underscore (to avoid matching @test in @test_2)
    # Use lookahead/lookbehind to avoid consuming characters.
    pattern = re.compile(r"(?<![a-zA-Z0-9_$.-])" + escaped + r"(?![a-zA-Z0-9_$.-])")

    matches = []
    for i, line in enumerate(ir_lines):
        if pattern.search(line):
            matches.append((i, line))
    return matches


def _extract_operation_prefix(ir_line: str, label: str) -> str | None:
    """Extract the operation prefix before the label in an IR line.

    Examples:
        "util.func @test(...)" with label "@test" -> "util.func @test"
        "  func.func private @foo" with label "@foo" -> "func.func private @foo"
        "%x = some.op @label" with label "@label" -> None (not a declaration)
        "util.global private @name" with label "@name" -> "util.global private @name"

    Args:
        ir_line: IR line containing the label
        label: Label text to find

    Returns:
        Operation prefix string, or None if not a declaration
    """
    # Find position of label in line.
    label_pos = ir_line.find(label)
    if label_pos == -1:
        return None

    # Extract everything before and including the label, up to ( or {.
    prefix_end = label_pos + len(label)
    remaining = ir_line[prefix_end:]

    # Extend to include trailing characters until ( or {.
    for char in remaining:
        if char in "({":
            break
        prefix_end += 1

    full_prefix = ir_line[:prefix_end].strip()

    # Check if this looks like a declaration (starts with operation name or attribute).
    # Skip if it's an SSA assignment (has = before the operation).
    if "=" in full_prefix:
        # Check if the = is before the label (SSA assignment) or after (not a declaration).
        eq_pos = full_prefix.find("=")
        if eq_pos < full_prefix.find(label):
            return None  # SSA assignment, not declaration.

    return full_prefix


class CheckLabelFormatChecker(LintChecker):
    """Ensures CHECK-LABEL uses proper function anchoring format.

    CHECK-LABEL should anchor to the full function operation, not just the
    function name, when there is ambiguity (multiple occurrences of the label)
    or when the label doesn't exist in the IR.

    Only warns when:
    - Label appears multiple times in IR (ambiguous)
    - Label doesn't appear in IR (likely typo)

    Does NOT warn when:
    - Label appears exactly once (unambiguous)
    - Label already has operation prefix (func.func, util.func, etc.)

    Valid patterns:
    - // CHECK-LABEL: func.func @function_name
    - // CHECK-LABEL: util.func @function_name
    - // CHECK-LABEL: util.func private @function_name
    - // CHECK-LABEL: @unique_name  (OK if appears only once)

    Examples of warnings:
      AMBIGUOUS: // CHECK-LABEL: @dispatch  (appears in multiple places)
      NOT FOUND: // CHECK-LABEL: @typo_name (doesn't exist in IR)
      GOOD:      // CHECK-LABEL: @unique_func (appears exactly once)
      GOOD:      // CHECK-LABEL: func.func @test (has operation prefix)
    """

    # Pattern to match CHECK-LABEL lines.
    CHECK_LABEL_PATTERN = re.compile(r"//\s*CHECK-LABEL:\s*(.+)$")

    # Common function operation prefixes in IREE.
    FUNCTION_OPS = [
        "func.func",
        "util.func",
        "builtin.func",
        "llvm.func",
        "spirv.func",
    ]

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for CHECK-LABEL format issues.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        # Extract IR lines once for all checks.
        ir_lines = _get_ir_lines(case)

        for i, line in enumerate(lines):
            # Only check CHECK-LABEL lines.
            if "CHECK-LABEL" not in line:
                continue

            match = self.CHECK_LABEL_PATTERN.search(line)
            if not match:
                continue

            pattern = match.group(1).strip()

            # Check if pattern is bare label (no operation prefix).
            label = _extract_label_text(pattern, self.FUNCTION_OPS)
            if label is None:
                continue  # Has operation prefix or not a bare @label, OK.

            # Find all occurrences in IR.
            matches = _find_label_matches(ir_lines, label)

            if len(matches) == 0:
                # No matches - likely typo or wrong label.
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="label_not_found_in_ir",
                        message=f"CHECK-LABEL '{label}' not found in IR",
                        line=case.start_line + i,
                        column=match.start(1),
                        snippet=line.strip(),
                        help=(
                            f"Label '{label}' does not appear in the IR. "
                            "This may indicate a typo or that the test is checking "
                            "the wrong function name. Verify the label matches an "
                            "actual symbol in the IR."
                        ),
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )
            elif len(matches) > 1:
                # Multiple matches - ambiguous!
                # Format match list (limit to first 5).
                match_lines_display = []
                for match_idx, (_line_idx, match_line) in enumerate(matches):
                    if match_idx >= 5:  # Limit display.
                        break
                    # Line number in original file (not in ir_lines).
                    # We need to find the actual line number in the case.
                    # Since ir_lines doesn't preserve line numbers, we'll just show the content.
                    match_lines_display.append(f"  {match_line.strip()}")

                if len(matches) > 5:
                    match_lines_display.append("  (and more...)")

                match_summary = "\n".join(match_lines_display)

                # Extract suggested prefix from first match.
                suggested_prefix = _extract_operation_prefix(matches[0][1], label)
                suggestions = []
                if suggested_prefix:
                    suggestions.append(suggested_prefix)

                issues.append(
                    LintIssue(
                        severity="warning",
                        code="ambiguous_label",
                        message=f"CHECK-LABEL uses ambiguous bare label '{label}'",
                        line=case.start_line + i,
                        column=match.start(1),
                        snippet=line.strip(),
                        help=(
                            f"Label '{label}' appears {len(matches)} times in the IR:\n"
                            f"{match_summary}\n\n"
                            "FileCheck may anchor to the wrong occurrence. "
                            "Include the operation prefix to make the anchor unambiguous."
                        ),
                        suggestions=suggestions,
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )
            # elif len(matches) == 1: skip - unambiguous, no warning needed.

            # Check if pattern looks like a bare identifier (no @ prefix).
            # Pattern doesn't start with known function op or @.
            # Could be a bare identifier or intentionally checking something else.
            # Only warn if it looks like a function name (alphanumeric/underscore).
            if (
                not pattern.startswith("@")
                and not any(pattern.startswith(op) for op in self.FUNCTION_OPS)
                and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", pattern)
            ):
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="missing_at_prefix_in_label",
                        message="CHECK-LABEL may be missing @ prefix",
                        line=case.start_line + i,
                        column=match.start(1),
                        snippet=line.strip(),
                        help=(
                            f"Use 'func.func @{pattern}' or 'util.func @{pattern}' instead. "
                            "Function names in MLIR use @ prefix. Without it, FileCheck "
                            "won't find the function declaration, causing the test to fail "
                            "or match unintended text. Include both the operation and @ prefix "
                            "for robust anchoring. "
                            "(Ignore if intentionally checking non-function constructs.)"
                        ),
                        suggestions=[
                            f"func.func @{pattern}",
                            f"util.func @{pattern}",
                        ],
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )

        return issues


class BracketBalanceChecker(LintChecker):
    """Validates balanced brackets in CHECK patterns.

    FileCheck patterns use brackets for captures and regex matching. Unbalanced
    brackets cause silent failures or unpredictable matching behavior. This
    checker validates:
    - Capture brackets: %[[NAME]] must have balanced [[ and ]]
    - Regex brackets: {{pattern}} must have balanced {{ and }}
    - Parentheses: (operands) must be balanced

    Common errors:
    - Missing closing bracket: %[[FOO    (should be %[[FOO]])
    - Missing opening bracket: FOO]]     (should be %[[FOO]])
    - Mismatched braces: {{pattern}      (should be {{pattern}})
    - Unbalanced parens: func.func(%arg0 (missing closing paren)

    Examples:
      BAD:  // CHECK: %[[FOO:.+] = op      (unbalanced capture)
      BAD:  // CHECK: {{.+} = op           (unbalanced regex)
      BAD:  // CHECK: func.func(%arg0      (unbalanced paren)
      GOOD: // CHECK: %[[FOO:.+]] = op
      GOOD: // CHECK: {{.+}} = op
      GOOD: // CHECK: func.func(%arg0)
    """

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for unbalanced brackets in CHECK patterns.

        Args:
            case: TestCase to examine

        Returns:
            List of issues found
        """
        issues = []
        lines = case.content.splitlines()

        for i, line in enumerate(lines):
            # Only check CHECK lines.
            if "//" not in line or "CHECK" not in line:
                continue

            # Extract the pattern part after the CHECK directive.
            check_match = re.search(r"//\s*CHECK[^:]*:\s*(.*)$", line)
            if not check_match:
                continue

            pattern = check_match.group(1)
            pattern_start = check_match.start(1)

            # Check different bracket types.
            bracket_issues = []

            # Check capture brackets [[ ]].
            capture_balance = self._check_bracket_balance(
                pattern, "[[", "]]", "capture bracket"
            )
            if capture_balance:
                bracket_issues.append(capture_balance)

            # Check regex braces {{ }}.
            regex_balance = self._check_bracket_balance(
                pattern, "{{", "}}", "regex brace"
            )
            if regex_balance:
                bracket_issues.append(regex_balance)

            # Check parentheses ( ).
            paren_balance = self._check_bracket_balance(
                pattern, "(", ")", "parenthesis"
            )
            if paren_balance:
                bracket_issues.append(paren_balance)

            # Check square brackets [ ] (used in types, attributes).
            square_balance = self._check_bracket_balance(
                pattern, "[", "]", "square bracket"
            )
            if square_balance:
                bracket_issues.append(square_balance)

            # Report all bracket issues for this line.
            for bracket_type, imbalance_msg in bracket_issues:
                issues.append(
                    LintIssue(
                        severity="error",
                        code="unbalanced_brackets",
                        message=f"unbalanced {bracket_type} in CHECK pattern",
                        line=case.start_line + i,
                        column=pattern_start,
                        snippet=line.strip(),
                        help=(
                            f"{imbalance_msg}. "
                            f"Unbalanced {bracket_type}s cause FileCheck to fail or produce "
                            "unpredictable matches. Check that every opening bracket has a "
                            "corresponding closing bracket. Common issues: "
                            "typos (%[[FOO] instead of %[[FOO]]), "
                            "copy-paste errors (missing closing brackets), "
                            "or regex escaping issues. "
                            "(Ignore if pattern intentionally matches literal brackets - "
                            "use escaping like \\[ \\] for literal brackets.)"
                        ),
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )

        return issues

    def _check_bracket_balance(
        self, pattern: str, open_bracket: str, close_bracket: str, bracket_name: str
    ) -> tuple[str, str] | None:
        """Check if brackets are balanced in a pattern.

        Args:
            pattern: The CHECK pattern to analyze
            open_bracket: Opening bracket string (e.g., "[[")
            close_bracket: Closing bracket string (e.g., "]]")
            bracket_name: Human-readable name for error messages

        Returns:
            Tuple of (bracket_name, imbalance_message) if unbalanced, None if balanced
        """
        # Count occurrences of opening and closing brackets.
        open_count = pattern.count(open_bracket)
        close_count = pattern.count(close_bracket)

        if open_count != close_count:
            if open_count > close_count:
                missing = open_count - close_count
                plural = "s" if missing > 1 else ""
                return (
                    bracket_name,
                    f"Missing {missing} closing {close_bracket}{plural}",
                )
            extra = close_count - open_count
            plural = "s" if extra > 1 else ""
            return (
                bracket_name,
                f"Extra {extra} closing {close_bracket}{plural} without matching opening",
            )

        return None


class WildcardPatternNormalizer(LintChecker):
    """Suggests simplifications for overly complex wildcard patterns.

    FileCheck supports powerful regex patterns, but overly complex patterns
    reduce test readability without adding value. This checker suggests
    simpler alternatives when possible.

    Common simplifications:
    - {{[a-zA-Z0-9_]+}} → {{.+}}  (match identifier)
    - {{[0-9]+}} → {{[0-9]+}}     (keep - specific enough)
    - {{.*}} → {{.+}}             (prefer .+ over .* for non-empty match)
    - {{.+?}} → {{.+}}            (non-greedy rarely needed in tests)

    The goal is not to catch errors, but to improve test readability by
    preferring simple patterns when they work equally well.

    Examples:
      COMPLEX: // CHECK: %[[VAL:.+]] = constant {{[a-zA-Z0-9_]+}}
      SIMPLE:  // CHECK: %[[VAL:.+]] = constant {{.+}}

      COMPLEX: // CHECK: types = {{.*}}
      SIMPLE:  // CHECK: types = {{.+}}
    """

    # Pattern to find {{...}} regex wildcards.
    WILDCARD_PATTERN = re.compile(r"\{\{(.+?)\}\}")

    # Patterns that can be simplified.
    SIMPLIFIABLE_PATTERNS = {
        # Identifier patterns (alphanumeric + underscore).
        r"[a-zA-Z0-9_]+": "{{.+}}",
        r"[A-Za-z0-9_]+": "{{.+}}",
        r"[a-z0-9_]+": "{{.+}}",
        r"[A-Z0-9_]+": "{{.+}}",
        r"[a-zA-Z_][a-zA-Z0-9_]*": "{{.+}}",
        # Word patterns.
        r"\w+": "{{.+}}",
        # Any character patterns (prefer non-empty).
        r".*": "{{.+}}",
        # Non-greedy patterns (rarely needed in tests).
        r".+?": "{{.+}}",
        r".*?": "{{.+}}",
    }

    def check(self, case: TestCase) -> list[LintIssue]:
        """Check for overly complex wildcard patterns.

        Args:
            case: TestCase to examine

        Returns:
            List of suggestions found
        """
        issues = []
        lines = case.content.splitlines()

        for i, line in enumerate(lines):
            # Only check CHECK lines.
            if "//" not in line or "CHECK" not in line:
                continue

            # Check for invalid zero-or-more patterns on SSA values and symbols.
            # %{{.*}} and @{{.*}} can match zero-length names which are never valid.
            if re.search(r"%\{\{\.\*\}\}", line):
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="invalid_ssa_zero_or_more",
                        message="SSA value uses {{.*}} which allows empty names",
                        line=case.start_line + i,
                        column=line.find("%{{.*}}"),
                        snippet=line.strip(),
                        help=(
                            "Use %{{.+}} instead of %{{.*}}. SSA values in MLIR must have "
                            "non-empty names like %0, %value, etc. The pattern {{.*}} matches "
                            "zero or more characters, allowing invalid matches like bare %. "
                            "Use {{.+}} (one or more) to ensure the name is non-empty. "
                            "(Applies to wildcards like %{{.*}}, not captures like %[[NAME:.+]])"
                        ),
                        suggestions=["%{{.+}}"],
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )

            if re.search(r"@\{\{\.\*\}\}", line):
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="invalid_symbol_zero_or_more",
                        message="symbol uses {{.*}} which allows empty names",
                        line=case.start_line + i,
                        column=line.find("@{{.*}}"),
                        snippet=line.strip(),
                        help=(
                            "Use @{{.+}} instead of @{{.*}}. Symbols in MLIR must have "
                            "non-empty names like @func, @global, etc. The pattern {{.*}} "
                            "matches zero or more characters, allowing invalid matches like "
                            "bare @. Use {{.+}} (one or more) to ensure the name is non-empty."
                        ),
                        suggestions=["@{{.+}}"],
                        context_lines=get_context_lines(case, i, self.context_lines),
                    )
                )

            # Find all wildcard patterns in line.
            for match in self.WILDCARD_PATTERN.finditer(line):
                pattern = match.group(1)

                # Check if pattern can be simplified.
                suggestion = self.SIMPLIFIABLE_PATTERNS.get(pattern)
                if suggestion:
                    issues.append(
                        LintIssue(
                            severity="info",
                            code="complex_wildcard_pattern",
                            message=f"wildcard pattern {{{{{pattern}}}}} can be simplified",
                            line=case.start_line + i,
                            column=match.start(),
                            snippet=line.strip(),
                            help=(
                                f"Use {suggestion} instead of {{{{{pattern}}}}}. "
                                f"Pattern {{{{{pattern}}}}} works but is unnecessarily complex "
                                "for typical test matching. The simpler {{.+}} pattern matches "
                                "the same content while improving readability. Reserve complex "
                                "regex patterns for cases where you need specific validation "
                                "(e.g., {{[0-9]+}} to ensure numeric values). "
                                "(Ignore if complex pattern is intentional - e.g., to document "
                                "expected format or catch specific patterns.)"
                            ),
                            suggestions=[suggestion],
                            context_lines=get_context_lines(
                                case, i, self.context_lines
                            ),
                        )
                    )

                # Additional check: suggest {{.+}} over {{.*}} for readability.
                elif pattern == ".*" and suggestion is None:
                    # Already handled above, but double-check.
                    pass

        return issues


# Registry of all available checkers.
# Tier 1: Errors (blocking) - CRITICAL requirements.
TIER_1_CHECKERS = [
    RawSSAIdentifierChecker(),  # CRITICAL: No raw %0, %c123 on LHS
    ZeroCheckLinesChecker(),  # CRITICAL: IR must have CHECKs
    TodoWithoutExplanationChecker(),  # ERROR: TODOs must explain
    CHECKNOTWithoutAnchorChecker(),  # ERROR: CHECK-NOT needs anchors
    FileCheckDirectiveValidator(),  # ERROR: Catch CHECK-DAD, CHECK-SMAE typos
    BracketBalanceChecker(),  # ERROR: Validate balanced brackets
    # Note: Separator checking done at file level (check_file_level_issues)
]

# Tier 2: Warnings (informational) - Best practices.
TIER_2_CHECKERS = [
    ExcessiveWildcardChecker(),  # WARN: Too many {{.+}}
    NonSemanticCaptureChecker(),  # WARN: %[[C0]], %[[ARG0]] anti-patterns
    MatchedCaptureNameChecker(),  # WARN: CHECK names should match IR
    WildcardInTerminatorChecker(),  # WARN: Verify terminator operands
    CHECKWithoutLabelContextChecker(),  # WARN: CHECKs should follow LABEL
    UnusedCaptureChecker(),  # WARN: Captures should be used
    NonInterleavedChecksChecker(),  # WARN: Interleave CHECKs with IR
    CheckLabelFormatChecker(),  # WARN: Ensure CHECK-LABEL: func.func @name
    WildcardPatternNormalizer(),  # INFO: Suggest {{.+}} simplifications
]

# All checkers enabled by default.
DEFAULT_CHECKERS = TIER_1_CHECKERS + TIER_2_CHECKERS


def run_all_checkers(
    case: TestCase,
    checkers: list[LintChecker] | None = None,
    context_lines: int = 5,
) -> list[LintIssue]:
    """Run all checkers on a test case.

    Args:
        case: TestCase to lint
        checkers: List of checkers to run (defaults to DEFAULT_CHECKERS)
        context_lines: Number of lines before/after for context (default: 5)

    Returns:
        Combined list of all issues found, sorted by line number
    """
    if checkers is None:
        checkers = DEFAULT_CHECKERS

    all_issues = []
    for checker in checkers:
        # Inject context_lines configuration before running checker.
        checker.context_lines = context_lines
        issues = checker.check(case)
        all_issues.extend(issues)

    # Sort by line number for consistent output.
    all_issues.sort(key=lambda issue: issue.line)
    return all_issues


def _check_separator_extra_text(
    line: str, line_num: int, extra_text: str
) -> LintIssue | None:
    """Check if separator has extra text after dashes.

    Args:
        line: The separator line
        line_num: 1-indexed line number
        extra_text: Text found after the dashes

    Returns:
        LintIssue if extra text present, None otherwise
    """
    if not extra_text:
        return None

    return LintIssue(
        severity="error",
        code="separator_with_extra_text",
        message="separator line contains extra text",
        line=line_num,
        column=None,
        snippet=line.strip(),
        help=(
            f'Separator should be "// -----" only, not "// ----- {extra_text}". '
            "Test cases are identified by CHECK-LABEL (@function_name) or "
            'ordinals from --list (e.g., "case 3"), never by text on the '
            "separator line. Remove extra text to avoid confusion with test "
            "case identification."
        ),
    )


def _check_separator_blank_before(
    lines: list[str], separator_index: int, separator_line: str
) -> LintIssue | None:
    """Check if separator has blank line before it.

    Args:
        lines: All lines in the file
        separator_index: 0-indexed position of separator
        separator_line: The separator line content

    Returns:
        LintIssue if blank line missing, None otherwise
    """
    if separator_index == 0:
        return None

    prev_line = lines[separator_index - 1].strip()
    if not prev_line:
        return None

    return LintIssue(
        severity="error",
        code="missing_blank_before_separator",
        message="separator line missing blank line before it",
        line=separator_index + 1,
        column=None,
        snippet=separator_line.strip(),
        help=(
            'Add blank line before "// -----" separator. '
            "Separators delimit test cases and should be visually distinct. "
            "A blank line before the separator improves readability and makes "
            "test case boundaries obvious. This is a consistent formatting "
            "convention across IREE lit tests."
        ),
    )


def _check_separator_blank_after(
    lines: list[str], separator_index: int, separator_line: str
) -> LintIssue | None:
    """Check if separator has blank line after it.

    Args:
        lines: All lines in the file
        separator_index: 0-indexed position of separator
        separator_line: The separator line content

    Returns:
        LintIssue if blank line missing, None otherwise
    """
    if separator_index >= len(lines) - 1:
        return None

    next_line = lines[separator_index + 1].strip()
    if not next_line:
        return None

    return LintIssue(
        severity="error",
        code="missing_blank_after_separator",
        message="separator line missing blank line after it",
        line=separator_index + 1,
        column=None,
        snippet=separator_line.strip(),
        help=(
            'Add blank line after "// -----" separator. '
            "Separators delimit test cases and should be visually distinct. "
            "A blank line after the separator improves readability and makes "
            "test case boundaries obvious. This is a consistent formatting "
            "convention across IREE lit tests."
        ),
    )


# Pattern to extract CHECK directive type from a line.
CHECK_DIRECTIVE_PATTERN = re.compile(
    r"//\s*(CHECK(?:-(?:LABEL|SAME|DAG|NOT|NEXT|COUNT(?:-\d+)?|EMPTY))?)\s*:",
    re.IGNORECASE,
)


def _extract_check_directive_type(line: str) -> str | None:
    """Extract the CHECK directive type from a line.

    Args:
        line: Source line to examine

    Returns:
        Directive type (e.g., "CHECK-LABEL", "CHECK-DAG") or None if not a CHECK
    """
    match = CHECK_DIRECTIVE_PATTERN.search(line)
    if match:
        return match.group(1).upper()
    return None


def _check_missing_split_input_file(
    delimiter_lines: tuple[int, ...],
    lines: list[str],
) -> LintIssue | None:
    """Check for // ----- delimiters without --split-input-file flag.

    Args:
        delimiter_lines: Indices of delimiter lines
        lines: All lines in the file

    Returns:
        LintIssue if delimiters present but --split-input-file missing
    """
    if not delimiter_lines:
        return None

    # Report on the first delimiter line.
    first_delim_idx = delimiter_lines[0]
    return LintIssue(
        severity="warning",
        code="missing_split_input_file",
        message="file has // ----- delimiters but no --split-input-file",
        line=first_delim_idx + 1,
        column=None,
        snippet=lines[first_delim_idx].strip() if first_delim_idx < len(lines) else "",
        help=(
            "Without --split-input-file, delimiters are cosmetic and tests run as "
            "a single module. Add --split-input-file to iree-opt/iree-compile command "
            "if tests should be isolated. Otherwise, remove the // ----- separators."
        ),
    )


def _check_first_check_not_label(
    line_idx: int,
    directive_type: str,
    line: str,
    delim_line_num: int,
) -> LintIssue:
    """Create issue for first CHECK after split not being CHECK-LABEL.

    Args:
        line_idx: 0-based line index of the CHECK directive
        directive_type: The type of CHECK directive found
        line: The source line
        delim_line_num: 1-based line number of the preceding delimiter

    Returns:
        LintIssue for the unanchored CHECK
    """
    return LintIssue(
        severity="warning",
        code="first_check_not_label_after_split",
        message=f"first CHECK after split (line {delim_line_num}) should be CHECK-LABEL",
        line=line_idx + 1,
        column=None,
        snippet=line.strip(),
        help=(
            "Without CHECK-LABEL, patterns may match output from previous test "
            "modules in --split-input-file mode. Add CHECK-LABEL: @functionName "
            "or CHECK-LABEL: module to anchor the match scope before other CHECKs."
        ),
    )


def _check_unanchored_check_dag(
    line_idx: int,
    directive_type: str,
    line: str,
    delim_line_num: int,
) -> LintIssue:
    """Create issue for CHECK-DAG/CHECK-COUNT before CHECK-LABEL after split.

    Args:
        line_idx: 0-based line index of the CHECK directive
        directive_type: The type of CHECK directive (CHECK-DAG or CHECK-COUNT-N)
        line: The source line
        delim_line_num: 1-based line number of the preceding delimiter

    Returns:
        LintIssue for the unanchored CHECK-DAG/CHECK-COUNT
    """
    return LintIssue(
        severity="warning",
        code="unanchored_check_dag_after_split",
        message=f"{directive_type} without CHECK-LABEL anchor after split (line {delim_line_num})",
        line=line_idx + 1,
        column=None,
        snippet=line.strip(),
        help=(
            f"{directive_type} can match across '// -----' boundaries in "
            "--split-input-file mode. Add CHECK-LABEL before this directive to "
            "anchor the match scope. Consider whether CHECK-DAG is necessary here - "
            "use CHECK if order is deterministic after the pass."
        ),
    )


def _check_split_boundary_issues(
    lines: list[str],
    delimiter_lines: tuple[int, ...],
    uses_split_input: bool,
) -> list[LintIssue]:
    """Check for issues related to split boundaries.

    Returns issues for:
    - Missing --split-input-file when delimiters present
    - CHECK-DAG/CHECK-COUNT before CHECK-LABEL after split
    - First CHECK not being CHECK-LABEL after split

    Args:
        lines: All lines in the file
        delimiter_lines: Indices of // ----- delimiter lines
        uses_split_input: Whether --split-input-file is in RUN lines

    Returns:
        List of issues found
    """
    issues = []

    # Check for missing --split-input-file if delimiters present.
    if delimiter_lines and not uses_split_input:
        issue = _check_missing_split_input_file(delimiter_lines, lines)
        if issue:
            issues.append(issue)
        return issues

    # Only run further checks if split-input-file is used.
    if not uses_split_input:
        return issues

    # For each delimiter, check the region after it.
    for delim_pos, delim_idx in enumerate(delimiter_lines):
        # Determine end of this region (next delimiter or EOF).
        if delim_pos + 1 < len(delimiter_lines):
            end_idx = delimiter_lines[delim_pos + 1]
        else:
            end_idx = len(lines)

        seen_check_label = False
        first_check_line_idx: int | None = None
        first_check_type: str | None = None

        # Scan lines after this delimiter.
        for i in range(delim_idx + 1, end_idx):
            line = lines[i]
            directive_type = _extract_check_directive_type(line)

            if directive_type is None:
                continue

            # Track first CHECK directive.
            if first_check_line_idx is None:
                first_check_line_idx = i
                first_check_type = directive_type

            # Check for CHECK-LABEL.
            if directive_type == "CHECK-LABEL":
                seen_check_label = True
            elif not seen_check_label and (
                directive_type == "CHECK-DAG"
                or directive_type.startswith("CHECK-COUNT")
            ):
                # CHECK-DAG or CHECK-COUNT-N before CHECK-LABEL.
                issues.append(
                    _check_unanchored_check_dag(i, directive_type, line, delim_idx + 1)
                )

        # After scanning, check if first directive was not CHECK-LABEL.
        if first_check_line_idx is not None and first_check_type != "CHECK-LABEL":
            issues.append(
                _check_first_check_not_label(
                    first_check_line_idx,
                    first_check_type,
                    lines[first_check_line_idx],
                    delim_idx + 1,
                )
            )

    return issues


def check_file_level_issues(test_file: TestFile) -> list[LintIssue]:
    """Check file-level issues that aren't tied to specific test cases.

    This includes checking separator lines, split boundary CHECK anchoring,
    and formatting issues.

    Args:
        test_file: Parsed test file

    Returns:
        List of file-level issues found
    """
    issues = []
    separator_pattern = re.compile(r"^//\s*(-{5,})\s*(.*)$")

    # Get raw lines from the document.
    lines = [line.get_full_line() for line in test_file.doc.lines]

    # Check split boundary issues (missing --split-input-file, unanchored CHECKs).
    split_issues = _check_split_boundary_issues(
        lines,
        test_file.delimiter_lines,
        test_file.uses_split_input_file,
    )
    issues.extend(split_issues)

    # Check separator formatting.
    for i, line in enumerate(lines):
        match = separator_pattern.match(line.strip())
        if not match:
            continue

        extra_text = match.group(2)

        # Check for extra text after dashes.
        issue = _check_separator_extra_text(line, i + 1, extra_text)
        if issue:
            issues.append(issue)

        # Check for blank line before separator.
        issue = _check_separator_blank_before(lines, i, line)
        if issue:
            issues.append(issue)

        # Check for blank line after separator.
        issue = _check_separator_blank_after(lines, i, line)
        if issue:
            issues.append(issue)

    # Check for excessive consecutive newlines (max 2 allowed).
    consecutive_blanks = 0
    for i, line in enumerate(lines):
        if not line.strip():
            consecutive_blanks += 1
            # Only report once when we first exceed 2 blank lines.
            if consecutive_blanks == 3:
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="excessive_blank_lines",
                        message="excessive blank lines (more than 2 consecutive)",
                        line=i + 1,
                        column=None,
                        snippet="(blank line)",
                        help=(
                            "Limit consecutive blank lines to 2 maximum. "
                            "Excessive blank lines reduce code density without improving "
                            "readability. Use 1 blank line for logical grouping within test "
                            "cases, 2 blank lines around separators. More than 2 is excessive."
                        ),
                    )
                )
        else:
            consecutive_blanks = 0

    return issues
