# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wrapper for running LLVM LIT on extracted test cases.

This module integrates with lit programmatically. It avoids writing into the
source tree by creating a temporary test directory and mapping its lit.cfg.py
to the real suite configuration discovered from the original test directory.
"""

import contextlib
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from common import build_detection

from lit_tools.core import rendering  # For inject_run_lines_with_case
from lit_tools.core.parser import TestCase, TestFile

# Error message length limits for readability in diagnostic output.
MAX_ERROR_LINES_FILECHECK = 10  # FileCheck error preview lines.
MAX_ERROR_LINES_IR_VALIDATION = 15  # IR verification error lines.
MAX_ERROR_LINES_IR_OPERATION = 10  # IR operation error context lines.
ASSERTION_CONTEXT_LINES = 3  # Lines of context around C assertions.
CHECK_FAILURE_CONTEXT_LINES = 5  # Lines of context around CHECK failures.


@dataclass
class LitResult:
    """Result from running lit on a test case."""

    passed: bool
    case_number: int
    case_name: str | None
    duration: float  # Execution time in seconds.
    stdout: str  # Full lit output.
    stderr: str  # Lit stderr.
    failure_summary: str | None  # Parsed failure details.
    run_commands: list[str]  # Executed RUN commands.


def inject_extra_flags(content: str, extra_flags: str) -> str:
    """Inject extra flags into first iree-* tool in each RUN line.

    Only modifies RUN lines that contain iree-* tools.
    Flags are inserted immediately after the tool name.

    Example:
        Input:
            // RUN: iree-opt --split-input-file %s | FileCheck %s

        With extra_flags="--debug --mlir-print-ir-after-all":

        Output:
            // RUN: iree-opt --debug --mlir-print-ir-after-all --split-input-file %s | FileCheck %s

    Args:
        content: Test file content (including RUN lines)
        extra_flags: Space-separated flags to inject

    Returns:
        Modified content with flags injected into RUN lines
    """
    if not extra_flags:
        return content

    lines = content.splitlines()
    modified_lines = []

    for line in lines:
        # Check if this is a RUN line.
        if re.match(r"^\s*//\s*RUN:", line):
            # Find first iree-* tool in this line.
            # Pattern: iree-{letters-digits-and-hyphens}
            # Matches: iree-opt, iree-compile, iree-run-module, iree-opt-19, etc.
            match = re.search(r"\b(iree-[a-z0-9-]+)\b", line, re.IGNORECASE)

            if match:
                tool_end = match.end(1)

                # Insert flags right after tool name.
                modified_line = line[:tool_end] + " " + extra_flags + line[tool_end:]
                modified_lines.append(modified_line)
            else:
                # No iree-* tool found, keep line unchanged.
                modified_lines.append(line)
        else:
            # Not a RUN line, keep unchanged.
            modified_lines.append(line)

    return "\n".join(modified_lines)


def _ensure_lit_importable() -> None:
    """Ensures `import lit` works either from third_party or site packages."""
    try:
        import lit  # noqa: PLC0415

        return
    except ImportError:
        pass

    # Use build_detection to find repo root, then locate lit in third_party.
    repo_root = build_detection.find_repo_root()
    if repo_root:
        lit_path = repo_root / "third_party/llvm-project/llvm/utils/lit"
        if lit_path.exists():
            sys.path.insert(0, str(lit_path))

    # Try import again (let ImportError propagate if it still fails).
    import lit  # noqa: F401, PLC0415


def extract_timeout_error(failure_text: str) -> str | None:
    """Extract timeout error from lit failure output.

    Detects when a test exceeds its time limit.

    Args:
        failure_text: Text between failure markers

    Returns:
        Timeout error message, or None if no timeout detected
    """
    # Look for timeout indicators from lit.
    if re.search(r"(TIMEOUT|Timeout|Reached timeout)", failure_text, re.IGNORECASE):
        return "Test exceeded timeout limit"
    return None


def extract_crash_error(failure_text: str) -> str | None:
    """Extract crash/segfault error from lit failure output.

    Detects crashes, segmentation faults, and abnormal terminations.

    Args:
        failure_text: Text between failure markers

    Returns:
        Crash error message, or None if no crash detected
    """
    # Check for signal termination.
    signal_match = re.search(
        r"(Command terminated with signal|Exited with signal|killed by signal)\s+(\d+)",
        failure_text,
    )
    if signal_match:
        signal_num = int(signal_match.group(2))
        signal_names = {
            6: "SIGABRT (abort)",
            11: "SIGSEGV (segmentation fault)",
            4: "SIGILL (illegal instruction)",
            8: "SIGFPE (floating point exception)",
            9: "SIGKILL (killed)",
        }
        signal_name = signal_names.get(signal_num, f"signal {signal_num}")
        return f"Process crashed with {signal_name}"

    # Check for exit codes that indicate crashes (128 + signal).
    exit_code_match = re.search(r"Command returned exit code (\d+)", failure_text)
    if exit_code_match:
        exit_code = int(exit_code_match.group(1))
        if 128 < exit_code < 160:  # Signal exits are 128 + signal_number.
            signal_num = exit_code - 128
            signal_names = {
                6: "SIGABRT (abort)",
                11: "SIGSEGV (segmentation fault)",
                4: "SIGILL (illegal instruction)",
                8: "SIGFPE (floating point exception)",
            }
            signal_name = signal_names.get(signal_num, f"signal {signal_num}")
            return f"Process crashed with {signal_name} (exit code {exit_code})"

    # Check for segfault text.
    if re.search(r"Segmentation fault", failure_text, re.IGNORECASE):
        return "Process crashed with segmentation fault"

    return None


def extract_assertion_error(failure_text: str) -> str | None:
    """Extract assertion failure from lit failure output.

    Detects both C assert() and C++ CHECK/DCHECK failures.

    Args:
        failure_text: Text between failure markers

    Returns:
        Assertion error with context, or None if no assertion found
    """
    # Look for C assertion failures.
    assert_match = re.search(
        r"(Assertion `.*?' failed|assert.*failed)", failure_text, re.MULTILINE
    )
    if assert_match:
        # Extract context lines around assertion.
        lines = failure_text.splitlines()
        for i, line in enumerate(lines):
            if assert_match.group(0) in line:
                start = max(0, i - 1)
                end = min(len(lines), i + ASSERTION_CONTEXT_LINES)
                context = "\n".join(lines[start:end])
                return f"Assertion failure:\n{context}"

    # Look for CHECK failures (glog style).
    check_match = re.search(r"Check failed:.*", failure_text, re.MULTILINE)
    if check_match:
        # Extract context around CHECK failure.
        lines = failure_text.splitlines()
        for i, line in enumerate(lines):
            if "Check failed:" in line:
                start = max(0, i)
                end = min(len(lines), i + CHECK_FAILURE_CONTEXT_LINES)
                context = "\n".join(lines[start:end])
                return f"CHECK failure:\n{context}"

    return None


def extract_invalid_ir_error(failure_text: str) -> str | None:
    """Extract MLIR IR verification error from lit failure output.

    Detects when MLIR verification fails due to invalid IR.

    Args:
        failure_text: Text between failure markers

    Returns:
        IR verification error, or None if no verification error found
    """
    # Look for MLIR verification failures.
    if re.search(r"error:.*verification failed", failure_text, re.IGNORECASE):
        # Extract the verification error details.
        error_match = re.search(
            r"(error:.*verification failed.*?)(?=\n\n|\nInput file:|\n--|\Z)",
            failure_text,
            re.DOTALL | re.IGNORECASE,
        )
        if error_match:
            error = error_match.group(1).strip()
            error_lines = error.splitlines()
            if len(error_lines) > MAX_ERROR_LINES_IR_VALIDATION:
                error_lines = error_lines[:MAX_ERROR_LINES_IR_VALIDATION] + [
                    "  ... (use -v for full output)"
                ]
            return "IR verification failed:\n" + "\n".join(error_lines)

    # Look for operation verification errors (e.g., "'func.func' op ...").
    op_error_match = re.search(r"error:.*'[\w.]+' op .*", failure_text, re.MULTILINE)
    if op_error_match:
        # Extract context around the error.
        lines = failure_text.splitlines()
        error_context = []
        found_error = False
        for line in lines:
            if op_error_match.group(0) in line:
                found_error = True
            if found_error:
                error_context.append(line)
                if len(error_context) >= MAX_ERROR_LINES_IR_OPERATION:
                    break
        if error_context:
            if len(error_context) > MAX_ERROR_LINES_IR_OPERATION:
                error_context = error_context[:MAX_ERROR_LINES_IR_OPERATION] + [
                    "  ... (use -v for full output)"
                ]
            return "IR validation error:\n" + "\n".join(error_context)

    return None


def extract_missing_file_error(failure_text: str) -> str | None:
    """Extract missing file error from lit failure output.

    Detects when a required file is not found.

    Args:
        failure_text: Text between failure markers

    Returns:
        Missing file error, or None if no file error found
    """
    # Look for "No such file or directory" errors.
    file_error_match = re.search(
        r"(.*?):.*[Nn]o such file or directory", failure_text, re.MULTILINE
    )
    if file_error_match:
        return f"File not found: {file_error_match.group(0)}"

    # Look for "unable to open file" errors.
    open_error_match = re.search(
        r"[Uu]nable to open file[: ]+['\"](.*?)['\"]", failure_text, re.MULTILINE
    )
    if open_error_match:
        return f"Unable to open file: {open_error_match.group(1)}"

    # Look for "Could not open" errors.
    could_not_open_match = re.search(
        r"[Cc]ould not open[: ]+['\"](.*?)['\"]", failure_text, re.MULTILINE
    )
    if could_not_open_match:
        return f"Could not open file: {could_not_open_match.group(1)}"

    return None


def parse_lit_failure(stdout: str, stderr: str) -> tuple[str | None, list[str]]:
    """Parse lit output to extract failure details.

    Extracts:
    - Commands that were executed (RUN lines)
    - FileCheck error messages
    - Crash/segfault information
    - Timeout information
    - Invalid IR errors
    - Missing file errors
    - Assertion failures

    Args:
        stdout: Lit's stdout
        stderr: Lit's stderr

    Returns:
        (failure_summary, run_commands)
        failure_summary: Concise error message, or None if no failure
        run_commands: List of executed commands
    """
    # Extract section between "**********" markers.
    failure_match = re.search(r"\*+ TEST .* FAILED \*+\n(.*?)\n\*+", stdout, re.DOTALL)

    if not failure_match:
        return None, []

    failure_text = failure_match.group(1)

    # Extract RUN commands (lines starting with "+").
    run_commands = [
        line.strip("+ ") for line in failure_text.splitlines() if line.startswith("+ ")
    ]

    # Try different error extractors in priority order.
    error = (
        extract_timeout_error(failure_text)
        or extract_crash_error(failure_text)
        or extract_assertion_error(failure_text)
        or extract_invalid_ir_error(failure_text)
        or extract_missing_file_error(failure_text)
        or extract_filecheck_error(failure_text)
    )

    # Build concise summary.
    summary_parts = []

    if run_commands:
        summary_parts.append(f"Failed command:\n  {run_commands[-1]}")

    if error:
        summary_parts.append(f"\n{error}")

    return "\n".join(summary_parts) if summary_parts else None, run_commands


def extract_filecheck_error(failure_text: str) -> str | None:
    """Extract the key FileCheck error message from lit failure output.

    Args:
        failure_text: Text between failure markers

    Returns:
        Concise FileCheck error, or None if no FileCheck error found
    """
    # Look for FileCheck error pattern:
    # test.mlir:42:11: error: CHECK: expected string not found
    error_match = re.search(
        r"([\w/.-]+\.mlir:\d+:\d+: error:.*?)(?=\n\n|\n<stdin>|\nInput file:|\n--)",
        failure_text,
        re.DOTALL,
    )

    if error_match:
        error = error_match.group(1).strip()
        # Limit to first few lines for conciseness.
        error_lines = error.splitlines()
        if len(error_lines) > MAX_ERROR_LINES_FILECHECK:
            error_lines = error_lines[:MAX_ERROR_LINES_FILECHECK] + [
                "  ... (use -v for full output)"
            ]
        return "\n".join(error_lines)

    return None


def run_lit_on_case(
    case: TestCase,
    test_file_obj: "TestFile",
    build_dir: Path,
    timeout: int = 60,
    extra_flags: str | None = None,
    verbose: bool = False,
    keep_temps: bool = False,
) -> LitResult:
    """Run lit on extracted test case.

    Args:
        case: Test case to run
        test_file_obj: Parsed test file object (contains path and cached run lines)
        build_dir: IREE build directory
        timeout: Test timeout in seconds (0 = no timeout)
        extra_flags: Extra flags to inject into iree-* tools
        verbose: Pass -a to lit (show all output)
        keep_temps: Don't delete temp file after test

    Returns:
        LitResult with test outcome and parsed details
    """
    # Extract file path from test_file_obj
    test_file_path = test_file_obj.doc.path
    # 1. Prepare test content with line preservation.
    # Use render_for_testing() which preserves exact line count (blanks RUN lines).
    # Prepend blank lines so line numbers match original file.
    blank_lines = "\n" * (case.start_line - 1)
    test_content = blank_lines + case.render_for_testing()

    # 2. Extract and inject RUN lines from cached test file object.
    header_runs = test_file_obj.extract_run_lines()
    case_runs_with_indices = case.extract_local_run_lines()
    case_runs = [(idx, cmd) for idx, cmd in case_runs_with_indices]
    test_content = rendering.inject_run_lines_with_case(
        test_content, header_runs, case_runs
    )

    # 3. Inject extra flags if provided.
    if extra_flags:
        test_content = inject_extra_flags(test_content, extra_flags)

    # 3. Prepare a temp directory outside the source tree.
    temp_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"iree_lit_test_{os.getpid()}"
    temp_root.mkdir(parents=True, exist_ok=True)
    # Include thread ID to prevent collisions when running in parallel with --workers.
    temp_file = temp_root / f"case{case.number}_t{threading.get_ident()}.mlir"
    temp_file.write_text(test_content)

    try:
        # 4. Programmatic lit run with config mapping to the real suite config.
        # Initialize env_backup early so finally block always has it defined.
        env_backup: dict[str, str | None] = {}

        _ensure_lit_importable()
        from lit import LitConfig  # noqa: PLC0415
        from lit import discovery as lit_discovery  # noqa: PLC0415
        from lit import run as lit_run  # noqa: PLC0415

        # Try to find lit.cfg.py by walking up from test file location.
        # Use lit's discovery to check the directory first.
        real_cfg_path = lit_discovery.dirContainsTestSuite(
            str(test_file_path.parent),
            LitConfig.LitConfig(
                progname="iree-lit-test",
                path=[],
                diagnostic_level="error",
                useValgrind=False,
                valgrindLeakCheck=False,
                valgrindArgs=[],
                noExecute=False,
                debug=False,
                isWindows=(os.name == "nt"),
                order=None,
                params={},
                config_prefix=None,
            ),
        )
        if not real_cfg_path:
            # Fall back to walking upwards for lit.cfg.py.
            for parent in [test_file_path.parent] + list(test_file_path.parent.parents):
                probe = parent / "lit.cfg.py"
                if probe.exists():
                    real_cfg_path = str(probe)
                    break

        lit_cfg = temp_root / "lit.cfg.py"
        config_map = {}

        if real_cfg_path:
            # Found lit.cfg.py - create stub and map to real config.
            # lit will load real_cfg_path instead of the stub via config_map.
            lit_cfg.write_text("# stub config mapped via config_map\n")
            config_map = {os.path.normcase(str(lit_cfg)): str(real_cfg_path)}
        else:
            # No lit.cfg.py found - generate complete config in temp_root.
            # This handles stdin mode, test files in arbitrary locations, etc.
            # No temp path checking needed - works anywhere!
            lit_cfg.write_text(
                f"""# Generated lit configuration (no lit.cfg.py found in directory tree).
import os
import lit.formats

config.name = "iree-lit-test"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.test_exec_root = r"{temp_root}"
config.environment = os.environ.copy()
"""
            )

        # Compose PATH for tools (like CTest does) and apply to process env so
        # suite configs that copy os.environ see it.
        build_bin_paths = [
            str(build_dir / "llvm-project" / "bin"),
            str(build_dir / "tools"),
        ]

        # Save environment for restoration in finally block.
        env_backup = {
            "PATH": os.environ.get("PATH"),
            "FILECHECK_OPTS": os.environ.get("FILECHECK_OPTS"),
            "TEST_TMPDIR": os.environ.get("TEST_TMPDIR"),
        }

        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(build_bin_paths + [old_path])
        os.environ.setdefault("FILECHECK_OPTS", "--enable-var-scope")
        os.environ.setdefault("TEST_TMPDIR", str(temp_root))

        # Lit configuration; use our per-test timeout as maxIndividualTestTime.
        lit_config = LitConfig.LitConfig(
            progname="iree-lit-test",
            path=build_bin_paths,
            diagnostic_level="note" if verbose else "error",
            useValgrind=False,
            valgrindLeakCheck=False,
            valgrindArgs=[],
            noExecute=False,
            debug=verbose,
            isWindows=(os.name == "nt"),
            order=None,
            params={
                # Map stub config to real config (if needed for source tree tests).
                "config_map": config_map
            },
            config_prefix=None,
            maxIndividualTestTime=timeout if timeout > 0 else 0,
        )

        # Discover tests for our temp file.
        tests = lit_discovery.find_tests_for_inputs(lit_config, [str(temp_file)])
        # Execute with 1 worker; use a reasonable overall run deadline if desired.
        start_time = time.time()

        runner = lit_run.Run(
            tests=tests,
            lit_config=lit_config,
            workers=1,
            progress_callback=lambda _: None,
            max_failures=1,
            timeout=None,
        )
        # Expected when test fails and max_failures=1; results still populated.
        with contextlib.suppress(lit_run.MaxFailuresError):
            runner.execute()
        duration = time.time() - start_time

        # Collect results (single test).
        test = tests[0] if tests else None
        passed = bool(test and test.result and not test.result.code.isFailure)
        stdout = test.result.output if test and test.result else ""
        stderr = ""  # lit mixes streams in result.output for ShTest
        failure_summary = None
        run_commands: list[str] = []
        if not passed or verbose:
            failure_summary, run_commands = parse_lit_failure(stdout, stderr)

        return LitResult(
            passed=passed,
            case_number=case.number,
            case_name=case.name,
            duration=duration,
            stdout=stdout,
            stderr=stderr,
            failure_summary=failure_summary,
            run_commands=run_commands,
        )

    finally:
        # Restore environment to prevent pollution across test runs.
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        # 8. Clean up temp file.
        if not keep_temps and temp_file.exists():
            with contextlib.suppress(Exception):
                temp_file.unlink()
        elif keep_temps and temp_file.exists() and verbose:
            # Note: Only printed if verbose, otherwise clutters output.
            from common import (  # noqa: PLC0415
                console as _console,
            )  # local import to avoid cycles

            _console.note(f"Temp file kept: {temp_file}")
