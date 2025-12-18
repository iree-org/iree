# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Issue types for log analysis extractors.

This module defines a type hierarchy for structured diagnostic information
extracted from logs. Each Issue type captures specific error details for
different failure modes (sanitizers, LIT tests, build errors, etc.).

Example usage:
    # Sanitizer error.
    issue = SanitizerIssue(
        severity=Severity.CRITICAL,
        actionable=True,
        message="Data race in Device::Submit",
        sanitizer_type="TSAN",
        error_type="data-race",
        stack_trace=["#0 Device::Submit() device.cc:123", ...],
    )

    # LIT test failure.
    issue = LITTestIssue(
        severity=Severity.HIGH,
        actionable=True,
        message="FileCheck pattern mismatch",
        test_file="test.mlir",
        test_line=42,
        check_type="CHECK-SAME",
        expected="expected pattern",
        actual="actual output",
    )
"""

from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """Issue severity levels with numeric values for sorting."""

    CRITICAL = 4  # Crashes, data corruption, security issues.
    HIGH = 3  # Functional failures, test failures.
    MEDIUM = 2  # Warnings, deprecations.
    LOW = 1  # Info, suggestions.


@dataclass
class Issue:
    """Base class for all extracted diagnostic issues.

    Line number indexing convention:
    - line_number: 0-indexed (matches LogBuffer and Python list indexing)
    - Source file line numbers in subclasses (e.g., LITTestIssue.test_line):
      Should use 1-indexed to match source file conventions

    Attributes:
        severity: Issue severity level.
        actionable: True if this requires code changes (vs infrastructure flake).
        message: Human-readable summary of the issue.
        context_lines: Log lines surrounding the issue (for display).
        line_number: Line number in log where issue was found (0-indexed).
        source_extractor: Name of extractor that found this issue.
    """

    severity: Severity
    actionable: bool
    message: str
    context_lines: list[str] = field(default_factory=list)
    line_number: int | None = None
    source_extractor: str = "Unknown"


@dataclass
class SanitizerIssue(Issue):
    """Sanitizer failure (TSAN, ASAN, LSAN, UBSAN, MSAN).

    Captures comprehensive information about sanitizer-detected errors including
    stack traces, memory details, thread/mutex state, and full reports.

    Attributes:
        sanitizer_type: Type of sanitizer ("asan", "lsan", "tsan", "msan", "ubsan").
        error_type: Specific error ("heap-buffer-overflow", "data-race", etc.).
        pid: Process ID from sanitizer report.

        # Stack traces (list of formatted frame strings).
        primary_stack: Main error location stack trace.
        allocation_stack: Memory allocation stack trace (ASAN/LSAN).
        conflicting_stack: Second access stack trace (TSAN data races).

        # Memory information (ASAN).
        access_type: Read or write access ("READ"/"WRITE").
        access_size: Number of bytes accessed.
        address: Memory address involved ("0xdeadbeef").
        memory_offset: Bytes after/before region boundary.
        memory_region_size: Size of allocated region in bytes.
        memory_region_start: Start address of region.
        memory_region_end: End address of region.

        # Leak information (LSAN).
        leak_type: Direct or indirect leak ("Direct"/"Indirect").
        leaked_bytes: Total bytes leaked.
        leaked_objects: Number of leaked objects.

        # Thread/synchronization information (TSAN).
        thread_id: Thread identifier ("T0", "main thread").
        mutex_state: Mutex lock state during access.
        thread_created_by: Parent thread information.

        # Origin information (MSAN).
        origin_description: Description of uninitialized value origin.

        # UBSAN specific.
        ubsan_file: Source file where UB occurred.
        ubsan_line: Line number in source file.
        ubsan_column: Column number in source file.

        # Complete report text.
        full_report: Complete sanitizer output from error to SUMMARY.
        summary_line: SUMMARY line from report.
    """

    # Classification.
    sanitizer_type: str = ""  # "asan", "lsan", "tsan", "msan", "ubsan"
    error_type: str = ""  # "heap-buffer-overflow", "data-race", etc.
    pid: int = 0

    # Stack traces.
    primary_stack: list[str] = field(default_factory=list)
    allocation_stack: list[str] = field(default_factory=list)
    conflicting_stack: list[str] = field(default_factory=list)

    # Memory info (ASAN).
    access_type: str = ""  # "READ" or "WRITE"
    access_size: int = 0
    address: str = ""  # "0xdeadbeef"
    memory_offset: int = 0  # bytes after/before
    memory_region_size: int = 0
    memory_region_start: str = ""
    memory_region_end: str = ""

    # Leak info (LSAN).
    leak_type: str = ""  # "Direct" or "Indirect"
    leaked_bytes: int = 0
    leaked_objects: int = 0

    # Thread/mutex info (TSAN).
    thread_id: str = ""  # "T0", "main thread"
    mutex_state: str = ""
    thread_created_by: str = ""

    # Origin info (MSAN).
    origin_description: str = ""

    # UBSAN info.
    ubsan_file: str = ""
    ubsan_line: int = 0
    ubsan_column: int = 0

    # Full report.
    full_report: str = ""
    summary_line: str = ""


@dataclass
class LITTestIssue(Issue):
    """LIT test failure (FileCheck, verifier, RUN command).

    Captures detailed information about LIT test failures including the
    test file, line number, check type, and expected vs actual output.

    Attributes:
        test_file: Path to the failing test file.
        test_line: Line number in test file where failure occurred (1-indexed,
            matching source file conventions).
        check_type: Type of FileCheck directive (CHECK, CHECK-SAME, etc.).
        check_pattern: The FileCheck pattern that failed.
        expected: Expected output pattern.
        actual: Actual output received.
        run_command: The RUN: command that was executing.
    """

    test_file: str = ""
    test_line: int = 0
    check_type: str = ""  # "CHECK", "CHECK-SAME", "CHECK-NOT", etc.
    check_pattern: str | None = None
    expected: str | None = None
    actual: str | None = None
    run_command: str | None = None


@dataclass
class MLIRCompilerIssue(Issue):
    """MLIR compiler error (iree-compile, mlir-opt, iree-opt).

    Captures structured information about MLIR compiler failures including
    file location, operation details, and full diagnostic context. Compatible
    with iree-lit-test/iree-lit-extract for test file extraction.

    Line number conventions:
    - error_line: 1-indexed (matches compiler output: file.mlir:17:15)
    - error_column: 1-indexed (matches compiler output)
    - line_number (inherited): 0-indexed (log line where error was found)

    Attributes:
        file_path: Path to .mlir file with error (may be relative or absolute).
        error_line: Line number where error occurred (1-indexed).
        error_column: Column number where error occurred (1-indexed).
        error_type: Error category inferred from message
            (e.g., "out-of-bounds", "does-not-dominate", "failed-to-tile").
        operation: MLIR operation name if present
            (e.g., "linalg.generic", "stream.async.execute").
        error_message: Primary error diagnostic text.
        source_snippet: MLIR source code line(s) showing the error location.
        caret_line: Line with caret (^) marker showing exact column.
        notes: List of related diagnostic notes. Each note is a dict with:
            - 'file': str - File path
            - 'line': int - Line number (1-indexed)
            - 'column': int - Column number (1-indexed)
            - 'type': str - Note type (e.g., "see current operation:", "called from")
            - 'body': str - Note body text (may include multi-line operation dump)
        full_diagnostic: Complete multi-line error text including all notes.
        test_target: Inferred test target path if detectable
            (e.g., "compiler/.../test/foo.mlir" or Bazel target).
        compile_command: Compilation command that triggered error if available.
    """

    # Error location.
    file_path: str = ""  # "tests/e2e/linalg/argmax.mlir"
    error_line: int = 0  # 17 (1-indexed, matching compiler output)
    error_column: int = 0  # 15 (1-indexed, matching compiler output)

    # Error details.
    error_type: str = ""  # "failed-to-tile", "out-of-bounds", "does-not-dominate"
    operation: str | None = None  # "linalg.generic", "tensor.extract_slice"
    error_message: str = ""  # "Failed to analyze the reduction operation."

    # Source context.
    source_snippet: str = ""  # "  %result:2 = linalg.generic {"
    caret_line: str = ""  # "              ^"

    # Related diagnostics.
    notes: list[dict] = field(
        default_factory=list
    )  # [{"file": ..., "line": ..., "type": "called from", "body": ...}]

    # Full diagnostic.
    full_diagnostic: str = ""  # Complete multi-line error from error to last note

    # Test context (optional).
    test_target: str | None = None  # "tests/e2e/linalg/argmax.mlir"
    compile_command: str | None = None  # "iree-compile ... -o foo.vmfb"


@dataclass
class MissingDependencyIssue(Issue):
    """Bazel/CMake missing dependency error.

    Captures information about missing build dependencies including the
    header/symbol that's missing and suggested fixes.

    Attributes:
        missing_header: Name of missing header or symbol.
        target: Build target that needs the dependency.
        suggested_dep: Suggested dependency to add.
        fix_suggestion: Actionable fix instructions.
    """

    missing_header: str = ""
    target: str | None = None
    suggested_dep: str | None = None
    fix_suggestion: str | None = None


@dataclass
class DeprecatedAPIIssue(Issue):
    """Deprecated API usage warning.

    Captures information about deprecated API usage including the symbol,
    replacement, and location in code.

    Attributes:
        deprecated_symbol: Name of deprecated symbol.
        replacement: Replacement API to use.
        file_path: Source file with deprecated usage.
        line: Line number in source file (1-indexed, matching compiler output).
        column: Column number in source file (1-indexed, matching compiler output).
    """

    deprecated_symbol: str = ""
    replacement: str = ""
    file_path: str = ""
    line: int = 0
    column: int = 0


@dataclass
class PythonTestIssue(Issue):
    """Python test failure (pytest, unittest).

    Captures detailed information about Python test failures including
    exception type, message, stack trace, and assertion details.

    Attributes:
        test_name: Name of the failing test.
        exception_type: Type of exception raised.
        exception_message: Exception message.
        stack_trace: Python stack trace frames.
        assertion_details: Details of failed assertion (if applicable).
    """

    test_name: str = ""
    exception_type: str = ""
    exception_message: str = ""
    stack_trace: list[str] = field(default_factory=list)
    assertion_details: str | None = None


@dataclass
class ROCmInfrastructureIssue(Issue):
    """ROCm infrastructure failure (cleanup crash, device lost).

    Captures information about ROCm/HIP infrastructure failures that are
    typically non-actionable (driver issues, cleanup crashes, etc.).

    Attributes:
        error_pattern: Type of infrastructure error.
        test_passed: True if tests passed before the infrastructure failure.
    """

    error_pattern: str = ""  # "cleanup_crash", "device_lost", etc.
    test_passed: bool = False  # True if tests passed before crash.


@dataclass
class BuildErrorIssue(Issue):
    """C/C++ compilation or linker error.

    Attributes:
        error_type: Type of error ("compile", "link", "undefined_reference").
        file_path: Source file path.
        line: Line number in source file (1-indexed).
        column: Column number in source file (1-indexed).
        compiler_message: Compiler/linker error message.
        compiler: Compiler name if detected ("clang", "gcc", "msvc", "ld").
    """

    error_type: str = ""
    file_path: str = ""
    line: int = 0
    column: int = 0
    compiler_message: str = ""
    compiler: str | None = None


@dataclass
class CMakeErrorIssue(Issue):
    """CMake configuration error.

    Attributes:
        cmake_file: CMake file where error occurred.
        cmake_line: Line number in CMake file (1-indexed).
        error_type: Type of error ("missing_dependency", "version_mismatch", "configuration", "error").
        cmake_command: CMake command that failed (e.g., "find_package").
        cmake_stack: CMake call stack if available.
        missing_dependency: Name of missing package/dependency.
        required_version: Required version string.
        found_version: Found version string (if any).
    """

    cmake_file: str = ""
    cmake_line: int = 0
    error_type: str = ""
    cmake_command: str | None = None
    cmake_stack: list[str] = field(default_factory=list)
    missing_dependency: str | None = None
    required_version: str | None = None
    found_version: str | None = None


@dataclass
class CTestErrorIssue(Issue):
    """CTest test failure.

    Attributes:
        test_name: Name of failed test.
        test_number: CTest test number.
        exit_code: Test exit code.
        failure_reason: Reason for failure ("failed", "timeout", "exception", "crash").
        test_output_tail: Last N lines of test output.
        elapsed_time: Test execution time in seconds.
    """

    test_name: str = ""
    test_number: int = 0
    exit_code: int | None = None
    failure_reason: str = "failed"
    test_output_tail: list[str] = field(default_factory=list)
    elapsed_time: float | None = None


@dataclass
class GPUDriverIssue(Issue):
    """GPU driver error (CUDA, HIP/ROCm, Vulkan, Metal).

    Attributes:
        gpu_type: GPU API ("cuda", "hip", "vulkan", "metal").
        error_code: Driver error code ("VK_ERROR_DEVICE_LOST", "hipErrorFileNotFound", etc.).
        error_type: Error category ("device_lost", "oom", "initialization_failed", "cleanup_crash").
        test_passed_before_crash: True if test passed before driver crash (for ROCm cleanup crash).
        rocclr_file: ROCm source file for cleanup crash debugging.
    """

    gpu_type: str = ""
    error_code: str = ""
    error_type: str = ""
    test_passed_before_crash: bool | None = None
    rocclr_file: str | None = None


@dataclass
class BazelErrorIssue(Issue):
    """Bazel build system error.

    Captures Bazel-specific build failures including missing targets and
    build summary errors. Note: Compiler errors from Bazel builds are
    handled by BuildErrorExtractor.

    Attributes:
        error_type: Type of error ("missing_target", "build_failed").
        bazel_file: BUILD.bazel file where error occurred.
        bazel_line: Line number in BUILD.bazel file (1-indexed).
        workspace: Workspace name for external dependencies.
        package: Package path ("compiler/src/iree/compiler/API").
        target: Target name that failed.
        failed_targets: List of failed targets (for build summary).
    """

    error_type: str = ""
    bazel_file: str = ""
    bazel_line: int = 0
    workspace: str | None = None
    package: str | None = None
    target: str | None = None
    failed_targets: list[str] = field(default_factory=list)


@dataclass
class PrecommitErrorIssue(Issue):
    """Pre-commit hook failure.

    Captures failures from pre-commit hooks (formatting, linting, etc.).
    These are typically non-critical infrastructure issues that can be
    fixed by running pre-commit locally.

    Attributes:
        hook_id: Pre-commit hook identifier.
        hook_description: Human-readable hook description.
        error_type: Type of error ("modified_files", "check_failed").
        files_modified: True if hook modified files (needs commit).
        error_details: Additional error details if available.
    """

    hook_id: str = ""
    hook_description: str = ""
    error_type: str = ""
    files_modified: bool = False
    error_details: str | None = None


@dataclass
class CodeQLIssue(Issue):
    """CodeQL code scanning failure.

    Captures failures from GitHub CodeQL code scanning, including language
    detection failures, analysis upload failures, and fatal errors during
    analysis. These are typically infrastructure/configuration issues.

    Attributes:
        error_category: Type of CodeQL error ("language", "upload", "fatal", "status").
        expected_language: Expected language for analysis (if language error).
        found_language: Actually detected language (if language error).
        command: Command that failed (if fatal error).
        exit_code: Exit code from failed command (if fatal error).
        status: Job or upload status (if status error).
    """

    error_category: str = ""
    expected_language: str | None = None
    found_language: str | None = None
    command: str | None = None
    exit_code: int | None = None
    status: str | None = None


@dataclass
class ONNXTestIssue(Issue):
    """ONNX test failure with rich reproduction context.

    Captures ONNX test failures from iree-test-suites with complete
    reproduction information: input MLIR, compilation command, run command,
    and test source URL. This provides everything needed for triage without
    setting up the ONNX environment.

    Attributes:
        test_name: Test name (e.g., "test_mean_example").
        error_tool: Tool that failed ("iree-compile" or "iree-run-module").
        error_code: Exit code from failed tool.
        error_message: Actual error message from stderr.
        input_mlir: Complete input MLIR program.
        compile_command: Full compilation command for reproduction.
        run_command: Full run command for reproduction.
        test_source_url: GitHub URL to test case source.
    """

    test_name: str = ""
    error_tool: str = ""  # "iree-compile" or "iree-run-module"
    error_code: int | None = None
    error_message: str = ""
    input_mlir: str = ""
    compile_command: str = ""
    run_command: str = ""
    test_source_url: str = ""
