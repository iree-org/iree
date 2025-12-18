# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Infrastructure flake extractor (GPU, filesystem, permissions, etc).

This extractor identifies infrastructure issues that are not code bugs.
Most importantly, it detects the ROCm cleanup crash (rocclr_memobj) which causes
false CI failures when tests pass but the driver crashes during teardown.

To add a new flake: Add it to SUBSTRING_PATTERNS below, write a test, done!

Example usage:
    from common.extractors.infrastructure_flake import InfrastructureFlakeExtractor
    from common.log_buffer import LogBuffer

    extractor = InfrastructureFlakeExtractor()
    log_buffer = LogBuffer(log_content, auto_detect_format=True)
    issues = extractor.extract(log_buffer)

    for issue in issues:
        if issue.error_type == "cleanup_crash":
            print(f"ROCm cleanup crash (test passed: {issue.test_passed_before_crash})")
"""

import re
from typing import Any

from common.extractors.base import Extractor
from common.issues import GPUDriverIssue, Severity
from common.log_buffer import LogBuffer

# =============================================================================
# Easy flake additions - just add a line here + write a test!
# =============================================================================
# Format: (substring, gpu_type, error_code, message, severity)
SUBSTRING_PATTERNS = [
    # Filesystem infrastructure flakes.
    (
        "Permission denied: '/shark-cache/data'",
        "filesystem",
        "SHARK_CACHE_PERMISSION",
        "Permission denied on /shark-cache/data (infrastructure issue)",
        Severity.HIGH,
    ),
    # Vulkan device initialization failures.
    (
        "physical device 0 invalid",  # From vulkan_driver.cc NOT_FOUND error.
        "vulkan",
        "VULKAN_DEVICE_INVALID",
        "Vulkan physical device not found/invalid (infrastructure issue)",
        Severity.HIGH,
    ),
    # Network/GitHub infrastructure flakes.
    (
        "##[error]fatal: unable to access 'https://",
        "network",
        "GIT_FETCH_ERROR",
        "Git fetch failed (network/GitHub infrastructure issue)",
        Severity.HIGH,
    ),
    (
        "rate limit exceeded",
        "network",
        "RATE_LIMIT_EXCEEDED",
        "API rate limit exceeded (infrastructure issue)",
        Severity.HIGH,
    ),
    (
        "Unable to download artifact",
        "network",
        "ARTIFACT_DOWNLOAD_FAILED",
        "Artifact download failed (GitHub infrastructure issue)",
        Severity.HIGH,
    ),
    (
        "internal compiler error: Bus error",
        "hardware",
        "COMPILER_BUS_ERROR",
        "Compiler crash with bus error (bad hardware/CI runner)",
        Severity.HIGH,
    ),
    (
        "git', 'lfs', 'pull",
        "network",
        "GIT_LFS_PULL_ERROR",
        "Git LFS pull failed (network/GitHub LFS infrastructure issue)",
        Severity.HIGH,
    ),
    # Add more simple patterns here!
]


class InfrastructureFlakeExtractor(Extractor):
    """Extracts infrastructure flake errors from logs.

    Handles GPU driver errors (CUDA, HIP/ROCm, Vulkan, Metal), filesystem
    permissions, and other infrastructure issues. The most critical pattern
    is the ROCm cleanup crash (rocclr_memobj + Aborted) which creates false
    CI failures.
    """

    name = "infrastructure_flake"
    activation_keywords = [
        # GPU errors only happen during actual execution, not during builds.
        # Use specific error patterns, not just API names (which appear in paths).
        "CUDA error:",  # CUDA runtime error.
        "HIP error:",  # HIP runtime error.
        "hipError",  # HIP error code prefix.
        "VK_ERROR_",  # Vulkan error code.
        "HSA_STATUS_ERROR",  # HSA runtime error.
        "rocclr",  # ROCm runtime (only in error messages, not paths).
        "iree-run-module",  # Actual IREE execution tool.
        "MTLCreate",  # Metal API calls (only during execution).
        "vkQueueSubmit failed",  # Vulkan execution error.
        "device lost",  # GPU device lost error.
        # Filesystem/permission errors.
        "PermissionError",  # Python permission errors.
        "Permission denied",  # General permission errors.
        # Network/GitHub infrastructure errors.
        "fatal: unable to access",  # Git fetch failures.
        "rate limit exceeded",  # GitHub API rate limits.
        "Unable to download artifact",  # GitHub artifact download failures.
        "internal compiler error",  # Compiler crashes (gcc/g++/clang).
        "git', 'lfs'",  # Git LFS failures (network/GitHub LFS issues).
    ]

    # CUDA patterns.
    _CUDA_ERROR_RE = re.compile(
        r"CUDA error.*?:?\s*(?P<code>cuda\w+|CUDA_ERROR_\w+|device\s+lost)",
        re.IGNORECASE,
    )
    _CUDA_OOM_RE = re.compile(r"CUDA out of memory", re.IGNORECASE)

    # HIP/ROCm patterns.
    _HIP_ERROR_RE = re.compile(r"hipError(?P<code>\w+)")
    _HIP_FILE_NOT_FOUND_RE = re.compile(r"hipErrorFileNotFound")
    _ROCM_ERROR_RE = re.compile(
        r"ROCm error|HSA_STATUS_ERROR|hsa.*(?:error|failed)|amd.*smi.*error",
        re.IGNORECASE,
    )
    # No regex needed - just substring checks for HSA queue destroy errors.
    _ROCCLR_ERROR_RE = re.compile(r"rocclr/[^\s:]+:\d+.*error|rocclr.*failed")
    _HIP_DEVICE_LOST_RE = re.compile(
        r"HIP.*device.*lost|device.*lost.*HIP", re.IGNORECASE
    )

    # Vulkan patterns.
    _VK_ERROR_RE = re.compile(r"VK_ERROR_(?P<code>\w+)")
    _VK_INIT_FAILED_RE = re.compile(r"VK_ERROR_INITIALIZATION_FAILED")
    _VK_GENERIC_ERROR_RE = re.compile(r"vulkan.*error|vkCreate.*failed", re.IGNORECASE)

    # Metal patterns.
    _METAL_ERROR_RE = re.compile(r"MTL.*Error|Metal.*error")

    # Dispatch table: maps regex patterns to handler methods.
    # Format: (regex, handler_method_name, needs_match_object)
    _REGEX_HANDLERS = [
        # CUDA patterns (most specific first).
        (_CUDA_OOM_RE, "_extract_cuda_oom", False),
        (_CUDA_ERROR_RE, "_extract_cuda_error", True),
        # HIP/ROCm patterns (most specific first).
        (_HIP_FILE_NOT_FOUND_RE, "_extract_hip_file_not_found", False),
        (_HIP_DEVICE_LOST_RE, "_extract_hip_device_lost", False),
        (_HIP_ERROR_RE, "_extract_hip_error", True),
        (_ROCM_ERROR_RE, "_extract_rocm_error", False),
        (_ROCCLR_ERROR_RE, "_extract_rocclr_error", False),
        # Vulkan patterns (most specific first).
        (_VK_INIT_FAILED_RE, "_extract_vk_init_failed", False),
        (_VK_ERROR_RE, "_extract_vulkan_error", True),
        (_VK_GENERIC_ERROR_RE, "_extract_vulkan_generic", False),
        # Metal patterns.
        (_METAL_ERROR_RE, "_extract_metal_error", False),
    ]

    # Known flaky CI runners (higher false positive rate).
    _FLAKY_RUNNERS = {"shark10-ci", "amdgpu-ci"}

    def extract(self, log_buffer: LogBuffer) -> list[GPUDriverIssue]:
        """Extract GPU driver issues from log.

        Args:
            log_buffer: LogBuffer with log content.

        Returns:
            List of GPUDriverIssue objects.
        """
        issues = []

        # Check for ROCm cleanup crash (highest priority).
        rocm_crash = self._check_rocm_cleanup_crash(log_buffer)
        if rocm_crash:
            issues.append(rocm_crash)

        # Scan for all GPU driver errors.
        for line_idx, line in enumerate(log_buffer.get_lines()):
            # HSA queue destroy errors (compound condition, handle separately).
            # Patterns: "error in hsa_queue_destroy:" or "hsa_queue_destroy threw an exception"
            if "hsa_queue_destroy" in line.lower() and (
                "error" in line.lower() or "threw an exception" in line.lower()
            ):
                issues.append(self._extract_hsa_queue_destroy(log_buffer, line_idx))
                continue

            # Check all regex patterns via dispatch table.
            matched = False
            for regex, handler_name, needs_match in self._REGEX_HANDLERS:
                match = regex.search(line)
                if match:
                    handler = getattr(self, handler_name)
                    if needs_match:
                        issues.append(handler(log_buffer, line_idx, match))
                    else:
                        issues.append(handler(log_buffer, line_idx))
                    matched = True
                    break

            if matched:
                continue

            # Check simple substring patterns.
            issue = self._handle_substring_pattern(log_buffer, line_idx, line)
            if issue:
                issues.append(issue)

        return issues

    def extract_from_annotations(
        self, annotations: list[dict[str, Any]], run_id: str, job_id: str
    ) -> list[GPUDriverIssue]:
        """Extract infrastructure flakes from GitHub job annotations.

        Annotations contain error messages from GitHub Actions infrastructure,
        such as runner crashes, disk full errors, and system failures.

        Args:
            annotations: List of annotation dictionaries from GitHub API
            run_id: Run ID for context
            job_id: Job ID for context

        Returns:
            List of GPUDriverIssue objects representing infrastructure failures
        """
        issues = []

        # Infrastructure flake patterns in annotations.
        patterns = [
            ("No space left on device", "runner", "RUNNER_DISK_FULL", Severity.HIGH),
            ("System.IO.IOException", "runner", "RUNNER_IO_ERROR", Severity.HIGH),
            ("GitHub.Runner.Worker", "runner", "RUNNER_CRASH", Severity.HIGH),
            (
                "Unhandled exception",
                "runner",
                "RUNNER_UNHANDLED_EXCEPTION",
                Severity.HIGH,
            ),
            ("The self-hosted runner", "runner", "RUNNER_LOST", Severity.HIGH),
            (
                "connection to the server was lost",
                "runner",
                "RUNNER_CONNECTION_LOST",
                Severity.HIGH,
            ),
        ]

        for annotation in annotations:
            message = annotation.get("message", "")
            title = annotation.get("title", "")
            annotation_level = annotation.get("annotation_level", "")

            # Only process error-level annotations.
            if annotation_level != "failure":
                continue

            # Check for infrastructure flake patterns.
            for pattern, gpu_type, error_code, severity in patterns:
                if pattern in message or pattern in title:
                    issues.append(
                        GPUDriverIssue(
                            severity=severity,
                            actionable=False,  # Infrastructure issues not actionable.
                            message=f"GitHub runner infrastructure failure: {pattern}",
                            line_number=0,  # No line number for annotations.
                            gpu_type=gpu_type,
                            error_code=error_code,
                            error_type="infrastructure",
                            context_lines=[
                                f"Annotation from run {run_id}, job {job_id}:",
                                f"  Title: {title}",
                                f"  Message: {message}",
                                f"  Level: {annotation_level}",
                            ],
                        )
                    )
                    break  # Only match first pattern per annotation.

        return issues

    def _check_rocm_cleanup_crash(self, log_buffer: LogBuffer) -> GPUDriverIssue | None:
        """Check for ROCm cleanup crash pattern.

        This is CRITICAL: ROCm driver has a bug where it crashes during teardown
        AFTER tests pass. This creates false CI failures. We must detect this
        and mark as infrastructure (non-actionable).

        Pattern:
        1. Find rocclr_memobj.cpp pattern.
        2. Check if followed by "Aborted" or "core dumped".
        3. Check if test passed before crash.

        Returns:
            GPUDriverIssue if cleanup crash detected, None otherwise.
        """
        # Search for unique ROCm memobj error message.
        rocclr_line_idx = None
        for line_idx, line in enumerate(log_buffer.get_lines()):
            if "Memobj map does not have ptr" in line:
                rocclr_line_idx = line_idx
                break

        if rocclr_line_idx is None:
            return None

        # Check if followed by "Aborted" or "core dumped" within 10 lines.
        lines = log_buffer.get_lines()
        for i in range(
            rocclr_line_idx, min(rocclr_line_idx + 10, log_buffer.line_count)
        ):
            if "Aborted" in lines[i] or "core dumped" in lines[i]:
                # Found ROCm cleanup crash.

                # Check if test passed before crash (look backward up to 200 lines).
                test_passed = self._check_test_passed_before(
                    log_buffer, rocclr_line_idx
                )

                return GPUDriverIssue(
                    severity=Severity.HIGH if test_passed else Severity.CRITICAL,
                    actionable=False,  # Infrastructure bug, not code bug!
                    message="ROCm driver cleanup crash (known infrastructure issue)",
                    line_number=rocclr_line_idx,
                    gpu_type="hip",
                    error_code="rocclr_memobj",
                    error_type="cleanup_crash",
                    test_passed_before_crash=test_passed,
                    rocclr_file="rocclr_memobj.cpp",
                    context_lines=log_buffer.get_context(
                        rocclr_line_idx, before=10, after=10
                    ),
                )

        return None

    def _check_test_passed_before(
        self, log_buffer: LogBuffer, crash_line_idx: int
    ) -> bool | None:
        """Check if test passed before crash (look backward)."""
        lines = log_buffer.get_lines()
        for i in range(crash_line_idx - 1, max(0, crash_line_idx - 200), -1):
            line = lines[i]
            if "PASSED" in line or "[       OK ]" in line:
                return True
            if "FAILED" in line or "[  FAILED  ]" in line:
                return False
        return None

    def _handle_substring_pattern(
        self, log_buffer: LogBuffer, line_idx: int, line: str
    ) -> GPUDriverIssue | None:
        """Handle simple substring pattern matching.

        Checks line against SUBSTRING_PATTERNS and creates issue if matched.
        Special case: COMPILER_BUS_ERROR requires system compiler verification.

        Args:
            log_buffer: LogBuffer with log content
            line_idx: Current line index
            line: Current line text

        Returns:
            GPUDriverIssue if pattern matched, None otherwise
        """
        for substring, gpu_type, error_code, message, severity in SUBSTRING_PATTERNS:
            if substring not in line:
                continue

            # Special case: Compiler bus error requires context verification.
            # Ensure it's /usr/bin/cc or /usr/bin/g++ (system compiler),
            # not iree-compile (which might also say "internal compiler error").
            if error_code == "COMPILER_BUS_ERROR":
                # Check previous 10 lines for system compiler invocation.
                context = log_buffer.get_context(line_idx, before=10, after=0)
                has_system_compiler = any(
                    "/usr/bin/cc" in ctx_line
                    or "/usr/bin/g++" in ctx_line
                    or "/usr/bin/c++" in ctx_line
                    or "/usr/bin/clang" in ctx_line
                    for ctx_line in context
                )
                if not has_system_compiler:
                    continue  # Skip - might be iree-compile crash.

            return GPUDriverIssue(
                severity=severity,
                actionable=False,  # All substring patterns are infrastructure.
                message=message,
                line_number=line_idx,
                gpu_type=gpu_type,
                error_code=error_code,
                error_type="infrastructure",
                context_lines=log_buffer.get_context(line_idx, before=5, after=5),
            )

        return None

    def _extract_cuda_error(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> GPUDriverIssue:
        """Extract CUDA error."""
        error_code = match.group("code")

        # Determine if device lost (non-actionable) or other error.
        is_device_lost = "lost" in error_code.lower()

        return GPUDriverIssue(
            severity=Severity.CRITICAL if not is_device_lost else Severity.HIGH,
            actionable=not is_device_lost,  # Device lost is infrastructure.
            message=f"CUDA error: {error_code}",
            line_number=line_idx,
            gpu_type="cuda",
            error_code=error_code,
            error_type="device_lost" if is_device_lost else "error",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_hip_error(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> GPUDriverIssue:
        """Extract HIP error."""
        error_code = f"hipError{match.group('code')}"

        # Common non-actionable errors.
        is_infrastructure = error_code in [
            "hipErrorDeviceLost",
            "hipErrorOutOfMemory",
        ]

        return GPUDriverIssue(
            severity=Severity.HIGH,
            actionable=not is_infrastructure,
            message=f"HIP error: {error_code}",
            line_number=line_idx,
            gpu_type="hip",
            error_code=error_code,
            error_type="device_lost"
            if "Lost" in error_code
            else "oom"
            if "Memory" in error_code
            else "error",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_vulkan_error(
        self, log_buffer: LogBuffer, line_idx: int, match: re.Match
    ) -> GPUDriverIssue:
        """Extract Vulkan error."""
        error_code = f"VK_ERROR_{match.group('code')}"

        # Common non-actionable errors.
        is_infrastructure = error_code in [
            "VK_ERROR_DEVICE_LOST",
            "VK_ERROR_OUT_OF_DEVICE_MEMORY",
            "VK_ERROR_OUT_OF_HOST_MEMORY",
        ]

        return GPUDriverIssue(
            severity=Severity.HIGH,
            actionable=not is_infrastructure,
            message=f"Vulkan error: {error_code}",
            line_number=line_idx,
            gpu_type="vulkan",
            error_code=error_code,
            error_type="device_lost"
            if "LOST" in error_code
            else "oom"
            if "MEMORY" in error_code
            else "error",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_cuda_oom(self, log_buffer: LogBuffer, line_idx: int) -> GPUDriverIssue:
        """Extract CUDA OOM error."""
        return GPUDriverIssue(
            severity=Severity.MEDIUM,
            actionable=False,  # Infrastructure - not enough GPU memory.
            message="CUDA out of memory",
            line_number=line_idx,
            gpu_type="cuda",
            error_code="CUDA_OOM",
            error_type="oom",
            context_lines=log_buffer.get_context(line_idx, before=3, after=3),
        )

    def _extract_hip_file_not_found(
        self, log_buffer: LogBuffer, line_idx: int
    ) -> GPUDriverIssue:
        """Extract HIP file not found error (missing device libraries)."""
        return GPUDriverIssue(
            severity=Severity.HIGH,
            actionable=True,  # Missing device libraries - installation issue.
            message="HIP file not found (missing device libraries)",
            line_number=line_idx,
            gpu_type="hip",
            error_code="hipErrorFileNotFound",
            error_type="missing_library",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_hip_device_lost(
        self, log_buffer: LogBuffer, line_idx: int
    ) -> GPUDriverIssue:
        """Extract HIP device lost error."""
        return GPUDriverIssue(
            severity=Severity.CRITICAL,
            actionable=False,  # Infrastructure - driver crash.
            message="HIP device lost (driver crash)",
            line_number=line_idx,
            gpu_type="hip",
            error_code="HIP_DEVICE_LOST",
            error_type="device_lost",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_rocm_error(
        self, log_buffer: LogBuffer, line_idx: int
    ) -> GPUDriverIssue:
        """Extract generic ROCm/HSA error."""
        return GPUDriverIssue(
            severity=Severity.HIGH,
            actionable=True,  # Could be code or infrastructure.
            message="ROCm/HSA runtime error",
            line_number=line_idx,
            gpu_type="hip",
            error_code="ROCM_ERROR",
            error_type="error",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_hsa_queue_destroy(
        self, log_buffer: LogBuffer, line_idx: int
    ) -> GPUDriverIssue:
        """Extract HSA queue destroy error (ROCm runtime queue teardown failure)."""
        return GPUDriverIssue(
            severity=Severity.HIGH,
            actionable=False,  # ROCm runtime error.
            message="HSA queue destroy failed (ROCm runtime error)",
            line_number=line_idx,
            gpu_type="hip",
            error_code="HSA_QUEUE_DESTROY_ERROR",
            error_type="queue_teardown",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_rocclr_error(
        self, log_buffer: LogBuffer, line_idx: int
    ) -> GPUDriverIssue:
        """Extract ROCm CLR internal error."""
        return GPUDriverIssue(
            severity=Severity.HIGH,
            actionable=False,  # Internal ROCm runtime error.
            message="ROCm CLR internal error",
            line_number=line_idx,
            gpu_type="hip",
            error_code="ROCCLR_ERROR",
            error_type="internal_error",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_vk_init_failed(
        self, log_buffer: LogBuffer, line_idx: int
    ) -> GPUDriverIssue:
        """Extract Vulkan initialization failed error."""
        return GPUDriverIssue(
            severity=Severity.HIGH,
            actionable=True,  # Could be config or driver issue.
            message="Vulkan initialization failed",
            line_number=line_idx,
            gpu_type="vulkan",
            error_code="VK_ERROR_INITIALIZATION_FAILED",
            error_type="initialization",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_vulkan_generic(
        self, log_buffer: LogBuffer, line_idx: int
    ) -> GPUDriverIssue:
        """Extract generic Vulkan error."""
        return GPUDriverIssue(
            severity=Severity.HIGH,
            actionable=True,  # Generic Vulkan API error.
            message="Vulkan API error",
            line_number=line_idx,
            gpu_type="vulkan",
            error_code="VULKAN_ERROR",
            error_type="error",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )

    def _extract_metal_error(
        self, log_buffer: LogBuffer, line_idx: int
    ) -> GPUDriverIssue:
        """Extract Metal API error."""
        return GPUDriverIssue(
            severity=Severity.HIGH,
            actionable=True,  # Metal API error.
            message="Metal API error",
            line_number=line_idx,
            gpu_type="metal",
            error_code="METAL_ERROR",
            error_type="error",
            context_lines=log_buffer.get_context(line_idx, before=5, after=5),
        )
