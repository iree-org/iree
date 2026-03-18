# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ONNX test suite failure extractor.

Extracts rich reproduction context from ONNX test failures including:
- Input MLIR program
- Compilation and run commands
- Error messages
- Test source URLs

This provides complete triage information without needing to set up ONNX.
"""

import contextlib

from common.extractors.base import Extractor
from common.issues import ONNXTestIssue, Severity
from common.log_buffer import LogBuffer


class ONNXTestExtractor(Extractor):
    """Extracts ONNX test failures with reproduction context."""

    name = "onnx_test"
    activation_keywords = [
        "Error invoking iree-run-module",
        "Error invoking iree-compile",
        "Input program:",
        "IREE compile and run:",
    ]

    def extract(self, log_buffer: LogBuffer) -> list[ONNXTestIssue]:
        """Extract ONNX test failures.

        Args:
            log_buffer: Log content to analyze

        Returns:
            List of ONNXTestIssue instances
        """
        issues = []

        i = 0
        while i < log_buffer.line_count:
            line = log_buffer.get_line(i)

            # Look for ONNX test failure marker.
            if "IREE compile and run:" in line:
                issue = self._extract_onnx_test(log_buffer, i)
                if issue:
                    issues.append(issue)
                    # Skip past this test (typical test is 20-30 lines).
                    i += 20
                    continue

            i += 1

        return issues

    def _parse_test_name_from_header(self, header_line: str) -> str:
        """Parse test name from ONNX test header line.

        Args:
            header_line: Header line with format "_ IREE compile and run: test_name::..."

        Returns:
            Extracted test name, or empty string if parsing fails
        """
        if "::" not in header_line:
            return ""

        parts = header_line.split("::")
        if len(parts) < 2:
            return ""

        # Format: "_ IREE compile and run: test_name::..."
        test_name_part = parts[0].split(":")
        if len(test_name_part) >= 2:
            return test_name_part[-1].strip()

        return ""

    def _find_test_block_end(self, log_buffer: LogBuffer, start_line: int) -> int:
        """Find the boundary of ONNX test block.

        Searches for next test header or end of log, whichever comes first.

        Args:
            log_buffer: LogBuffer with log content
            start_line: Starting line index

        Returns:
            Offset from start_line where test block ends
        """
        max_search = min(200, log_buffer.line_count - start_line)
        for offset in range(1, max_search):
            line = log_buffer.get_line(start_line + offset)
            if "IREE compile and run:" in line:
                return offset
        return max_search

    def _extract_mlir_from_block(
        self, log_buffer: LogBuffer, start_line: int, offset: int
    ) -> str:
        """Extract MLIR code from Input program block.

        Looks for ``` code fence markers and extracts content between them.

        Args:
            log_buffer: LogBuffer with log content
            start_line: Test start line index
            offset: Offset from start_line where "Input program:" was found

        Returns:
            Extracted MLIR code, or empty string if not found
        """
        mlir_lines = []
        in_code_block = False
        for j in range(1, 100):
            if start_line + offset + j >= log_buffer.line_count:
                break
            mlir_line = log_buffer.get_line(start_line + offset + j)
            if "```" in mlir_line:
                if in_code_block:
                    break  # End of MLIR.
                in_code_block = True  # Start of MLIR.
                continue
            if in_code_block:
                mlir_lines.append(mlir_line)
        return "\n".join(mlir_lines)

    def _extract_error_message_after_marker(
        self, log_buffer: LogBuffer, start_line: int, offset: int
    ) -> str:
        """Extract error message after "Stderr diagnostics:" marker.

        Args:
            log_buffer: LogBuffer with log content
            start_line: Test start line index
            offset: Offset where "Stderr diagnostics:" was found

        Returns:
            First non-empty line after marker, or empty string if not found
        """
        for j in range(1, 10):
            if start_line + offset + j >= log_buffer.line_count:
                break
            err_line = log_buffer.get_line(start_line + offset + j)
            if err_line.strip() and "Stdout diagnostics:" not in err_line:
                return err_line.strip()
        return ""

    def _scan_test_block_for_data(
        self, log_buffer: LogBuffer, start_line: int, block_end: int
    ) -> dict:
        """Scan ONNX test block and extract all data in single pass.

        Args:
            log_buffer: LogBuffer with log content
            start_line: Starting line index
            block_end: Offset where test block ends

        Returns:
            Dict with keys: error_tool, error_code, error_message, input_mlir,
            compile_command, run_command, test_source_url
        """
        data = {
            "error_tool": "",
            "error_code": None,
            "error_message": "",
            "input_mlir": "",
            "compile_command": "",
            "run_command": "",
            "test_source_url": "",
        }

        for offset in range(1, block_end):
            line = log_buffer.get_line(start_line + offset)

            # Extract error tool.
            if "Error invoking iree-run-module" in line:
                data["error_tool"] = "iree-run-module"
            elif "Error invoking iree-compile" in line:
                data["error_tool"] = "iree-compile"

            # Extract error code.
            if "Error code:" in line and data["error_code"] is None:
                with contextlib.suppress(ValueError, IndexError):
                    data["error_code"] = int(line.split("Error code:")[-1].strip())

            # Extract error message (after "Stderr diagnostics:").
            if "Stderr diagnostics:" in line and not data["error_message"]:
                data["error_message"] = self._extract_error_message_after_marker(
                    log_buffer, start_line, offset
                )

            # Extract test source URL.
            if "Test case source:" in line:
                next_line = log_buffer.get_line(start_line + offset + 1)
                if "github.com" in next_line:
                    data["test_source_url"] = next_line.strip()

            # Extract input MLIR (between ``` markers after "Input program:").
            if "Input program:" in line:
                data["input_mlir"] = self._extract_mlir_from_block(
                    log_buffer, start_line, offset
                )

            # Extract compilation command.
            if "Compiled with:" in line:
                next_line = log_buffer.get_line(start_line + offset + 1)
                if "iree-compile" in next_line:
                    data["compile_command"] = next_line.strip()

            # Extract run command.
            if "Run with:" in line:
                next_line = log_buffer.get_line(start_line + offset + 1)
                if "iree-run-module" in next_line:
                    data["run_command"] = next_line.strip()

        return data

    def _determine_onnx_severity(self, error_message: str) -> tuple[Severity, bool]:
        """Determine severity and actionability for ONNX test error.

        Args:
            error_message: Error message from test

        Returns:
            Tuple of (severity, actionable)
        """
        # Most ONNX test failures are infrastructure (GPU not available, etc.).
        infrastructure_keywords = [
            "physical device",
            "device not found",
            "vulkan",
            "out of memory",
            "timeout",
        ]

        if any(keyword in error_message.lower() for keyword in infrastructure_keywords):
            return (Severity.HIGH, False)  # Infrastructure issue.
        return (Severity.HIGH, True)  # Compilation or runtime error - actionable.

    def _extract_onnx_test(
        self, log_buffer: LogBuffer, start_line: int
    ) -> ONNXTestIssue | None:
        """Extract single ONNX test failure.

        Pattern:
            _ IREE compile and run: test_name::model.mlir::model.mlir::gpu_vulkan __
            [gw2] linux -- Python ...
            Error invoking iree-run-module  (or iree-compile)
            Error code: 1
            Stderr diagnostics:
            <error message>

            Stdout diagnostics:

            Test case source:
              <github url>

            Input program:
            ```
            <mlir code>
            ```

            Compiled with:
              <compile command>

            Run with:
              <run command>

        Returns:
            ONNXTestIssue or None if extraction fails
        """
        # Parse test name from header.
        header_line = log_buffer.get_line(start_line)
        test_name = self._parse_test_name_from_header(header_line)

        # Find test block boundary.
        block_end = self._find_test_block_end(log_buffer, start_line)

        # Scan block for all data in single pass.
        data = self._scan_test_block_for_data(log_buffer, start_line, block_end)

        # Only create issue if we found error information.
        if not data["error_tool"]:
            return None

        # Determine severity and actionability.
        severity, actionable = self._determine_onnx_severity(data["error_message"])

        return ONNXTestIssue(
            severity=severity,
            actionable=actionable,
            message=f"ONNX test '{test_name}' failed: {data['error_message'][:100]}",
            line_number=start_line,
            source_extractor=self.name,
            test_name=test_name,
            error_tool=data["error_tool"],
            error_code=data["error_code"],
            error_message=data["error_message"],
            input_mlir=data["input_mlir"],
            compile_command=data["compile_command"],
            run_command=data["run_command"],
            test_source_url=data["test_source_url"],
            context_lines=log_buffer.get_context(start_line, before=2, after=10),
        )
