# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for InfrastructureFlakeExtractor."""

import unittest

from common.extractors.infrastructure_flake import InfrastructureFlakeExtractor
from common.issues import Severity
from common.log_buffer import LogBuffer


class TestInfrastructureFlakeExtractor(unittest.TestCase):
    """Test InfrastructureFlakeExtractor for all infrastructure flakes."""

    def setUp(self):
        """Set up extractor for tests."""
        self.extractor = InfrastructureFlakeExtractor()

    def test_rocm_cleanup_crash_with_passed_test(self):
        """Test ROCm cleanup crash detection after passing test."""
        log = """
[       OK ] TestSuite.TestCase (15 ms)
[  PASSED  ] 10 tests.
/opt/rocm/rocclr/device/rocm/rocmemory.cpp:2345: int rocmemobj::mapMemory()
Assertion 'Memobj map does not have ptr' failed
Aborted (core dumped)
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "hip")
        self.assertEqual(issue.error_code, "rocclr_memobj")
        self.assertEqual(issue.error_type, "cleanup_crash")
        self.assertFalse(issue.actionable)  # Infrastructure bug!
        self.assertEqual(
            issue.severity, Severity.HIGH
        )  # Passed test = HIGH not CRITICAL.
        self.assertTrue(issue.test_passed_before_crash)

    def test_rocm_cleanup_crash_with_failed_test(self):
        """Test ROCm cleanup crash detection after failed test."""
        log = """
[  FAILED  ] TestSuite.TestCase (timeout)
:0:/long_pathname/src/external/clr/rocclr/device/device.cpp:321 : 12074 us: [pid:123] Memobj map does not have ptr: 0x0
Aborted (core dumped)
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertFalse(issue.actionable)  # Still infrastructure!
        self.assertEqual(issue.severity, Severity.CRITICAL)  # Failed test = CRITICAL.
        self.assertFalse(issue.test_passed_before_crash)

    def test_hip_file_not_found(self):
        """Test hipErrorFileNotFound detection (missing device libraries)."""
        log = """
[IREE-PJRT] HIP error: hipErrorFileNotFound
Failed to find device kernel library at /opt/rocm/lib/bitcode
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "hip")
        self.assertEqual(issue.error_code, "hipErrorFileNotFound")
        self.assertEqual(issue.error_type, "missing_library")
        self.assertTrue(issue.actionable)  # Missing libraries - should fix!
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_hip_device_lost(self):
        """Test HIP device lost detection (driver crash)."""
        log = """
iree-run-module: HIP device lost during execution
Error: device communication failed
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "hip")
        self.assertEqual(issue.error_code, "HIP_DEVICE_LOST")
        self.assertEqual(issue.error_type, "device_lost")
        self.assertFalse(issue.actionable)  # Driver crash - infrastructure!
        self.assertEqual(issue.severity, Severity.CRITICAL)

    def test_hip_error_generic(self):
        """Test generic HIP error detection."""
        log = """
Runtime error: hipErrorInvalidValue at line 123
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "hip")
        self.assertEqual(issue.error_code, "hipErrorInvalidValue")
        self.assertTrue(issue.actionable)  # Code bug.
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_hip_device_lost_alternate_pattern(self):
        """Test HIP device lost with alternate wording."""
        log = """
Error: device lost in HIP runtime
Device reset required
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.error_type, "device_lost")
        self.assertFalse(issue.actionable)

    def test_rocm_hsa_error(self):
        """Test ROCm HSA error detection."""
        log = """
HSA_STATUS_ERROR_INVALID_QUEUE at line 456
ROCm runtime initialization failed
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "hip")
        self.assertEqual(issue.error_code, "ROCM_ERROR")
        self.assertTrue(issue.actionable)  # Could be code or config.
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_rocclr_internal_error(self):
        """Test ROCm CLR internal error detection."""
        log = """
rocclr/device/gpu/gpudevice.cpp:789: error: internal assertion failed
Device initialization error
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "hip")
        self.assertEqual(issue.error_code, "ROCCLR_ERROR")
        self.assertEqual(issue.error_type, "internal_error")
        self.assertFalse(issue.actionable)  # Internal ROCm error.
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_hsa_queue_destroy_error(self):
        """Test HSA queue destroy error detection (ROCm runtime teardown)."""
        log = """
error in hsa_queue_destroy: Invalid PCI BUS ID
terminate called after throwing an instance of 'std::runtime_error'
  what():  Invalid PCI BUS ID
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "hip")
        self.assertEqual(issue.error_code, "HSA_QUEUE_DESTROY_ERROR")
        self.assertEqual(issue.error_type, "queue_teardown")
        self.assertFalse(issue.actionable)  # ROCm runtime error.
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_hsa_queue_destroy_exception(self):
        """Test HSA queue destroy exception variant (ERROR: threw an exception)."""
        log = """
ERROR: hsa_queue_destroy threw an exception: Invalid PCI BUS ID
ERROR: hsa_queue_destroy threw an exception: Invalid PCI BUS ID
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        # Should extract both occurrences.
        self.assertEqual(len(issues), 2)
        for issue in issues:
            self.assertEqual(issue.gpu_type, "hip")
            self.assertEqual(issue.error_code, "HSA_QUEUE_DESTROY_ERROR")
            self.assertEqual(issue.error_type, "queue_teardown")
            self.assertFalse(issue.actionable)
            self.assertEqual(issue.severity, Severity.HIGH)

    def test_cuda_out_of_memory(self):
        """Test CUDA OOM detection."""
        log = """
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
cudaMalloc failed: out of memory
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "cuda")
        self.assertEqual(issue.error_code, "CUDA_OOM")
        self.assertEqual(issue.error_type, "oom")
        self.assertFalse(issue.actionable)  # Infrastructure - not enough memory.
        self.assertEqual(issue.severity, Severity.MEDIUM)

    def test_cuda_device_lost(self):
        """Test CUDA device lost detection."""
        log = """
CUDA error: device lost during kernel execution
cudaDeviceSynchronize returned cudaErrorDeviceLost
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "cuda")
        self.assertIn("lost", issue.error_code.lower())
        self.assertEqual(issue.error_type, "device_lost")
        self.assertFalse(issue.actionable)  # Driver issue!
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_cuda_error_generic(self):
        """Test generic CUDA error (actionable)."""
        log = """
CUDA error: cudaErrorInvalidConfiguration at line 234
Invalid block/grid dimensions
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "cuda")
        self.assertTrue(issue.actionable)  # Code bug.
        self.assertEqual(issue.severity, Severity.CRITICAL)

    def test_vulkan_device_lost(self):
        """Test Vulkan device lost (infrastructure flake)."""
        log = """
vkQueueSubmit failed with VK_ERROR_DEVICE_LOST
Device reset detected by driver
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "vulkan")
        self.assertEqual(issue.error_code, "VK_ERROR_DEVICE_LOST")
        self.assertEqual(issue.error_type, "device_lost")
        self.assertFalse(issue.actionable)  # Driver crash - infrastructure!
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_vulkan_out_of_memory(self):
        """Test Vulkan OOM (infrastructure flake)."""
        log = """
vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY
Failed to allocate 512MB device memory
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "vulkan")
        self.assertEqual(issue.error_code, "VK_ERROR_OUT_OF_DEVICE_MEMORY")
        self.assertEqual(issue.error_type, "oom")
        self.assertFalse(issue.actionable)  # Infrastructure - not enough memory.
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_vulkan_initialization_failed(self):
        """Test Vulkan initialization failure (could be code or infrastructure)."""
        log = """
vkCreateInstance failed: VK_ERROR_INITIALIZATION_FAILED
Failed to initialize Vulkan runtime
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "vulkan")
        self.assertEqual(issue.error_code, "VK_ERROR_INITIALIZATION_FAILED")
        self.assertEqual(issue.error_type, "initialization")
        self.assertTrue(issue.actionable)  # Could be config issue - investigate!
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_vulkan_error_actionable(self):
        """Test Vulkan error that is actionable (code bug)."""
        log = """
vkCreateGraphicsPipeline failed: VK_ERROR_INVALID_SHADER
Shader compilation failed
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "vulkan")
        self.assertEqual(issue.error_code, "VK_ERROR_INVALID_SHADER")
        self.assertTrue(issue.actionable)  # Code bug!
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_vulkan_generic_error(self):
        """Test generic Vulkan error from vkCreate failure."""
        log = """
vulkan runtime error: vkCreateCommandPool failed
Command pool creation error
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "vulkan")
        self.assertEqual(issue.error_code, "VULKAN_ERROR")
        self.assertTrue(issue.actionable)  # Generic API error - investigate.
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_metal_error(self):
        """Test Metal API error detection."""
        log = """
MTLCreateSystemDefaultDevice failed
Metal initialization error: device not available
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "metal")
        self.assertEqual(issue.error_code, "METAL_ERROR")
        self.assertTrue(issue.actionable)  # Metal API error.
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_multiple_gpu_errors_in_log(self):
        """Test detection of multiple different GPU errors in one log."""
        log = """
Test starting...
hipErrorFileNotFound: missing device library
Test continues...
VK_ERROR_DEVICE_LOST: vulkan device reset
Test ending...
CUDA error: cudaErrorInvalidValue
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        # Should find all 3 errors.
        self.assertEqual(len(issues), 3)

        # Verify each type was detected.
        error_codes = {issue.error_code for issue in issues}
        self.assertIn("hipErrorFileNotFound", error_codes)
        self.assertIn("VK_ERROR_DEVICE_LOST", error_codes)
        self.assertIn("cudaErrorInvalidValue", error_codes)

    def test_no_gpu_errors(self):
        """Test log with no GPU errors."""
        log = """
Running tests...
All tests passed successfully
No GPU errors detected
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)

    def test_permission_denied_shark_cache(self):
        """Test permission denied on /shark-cache/data (infrastructure flake)."""
        log = """
Test Torch / torch_models tests :: amdgpu_mi325_gfx942    UNKNOWN STEP    2025-10-14T15:34:17.3331033Z INTERNALERROR> PermissionError: [Errno 13] Permission denied: '/shark-cache/data'
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "filesystem")
        self.assertEqual(issue.error_code, "SHARK_CACHE_PERMISSION")
        self.assertEqual(issue.error_type, "infrastructure")
        self.assertFalse(issue.actionable)  # Infrastructure flake!
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_git_fetch_error(self):
        """Test Git fetch failure (network infrastructure flake)."""
        log = """
setup / setup    UNKNOWN STEP    2025-11-18T20:58:35.5778693Z ##[group]Fetching the repository
setup / setup    UNKNOWN STEP    2025-11-18T21:00:30.5839972Z ##[error]fatal: unable to access 'https://github.com/iree-org/iree/': The requested URL returned error: 502
setup / setup    UNKNOWN STEP    2025-11-18T21:00:30.5847541Z The process '/usr/bin/git' failed with exit code 128
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "network")
        self.assertEqual(issue.error_code, "GIT_FETCH_ERROR")
        self.assertEqual(issue.error_type, "infrastructure")
        self.assertFalse(issue.actionable)
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_rate_limit_exceeded(self):
        """Test API rate limit exceeded (infrastructure flake)."""
        log = """
publish_website    UNKNOWN STEP    2025-10-30T15:58:19.8199355Z Traceback (most recent call last):
publish_website    UNKNOWN STEP    2025-10-30T15:58:19.8222430Z RuntimeError: Request was not successful, reason: rate limit exceeded
publish_website    UNKNOWN STEP    2025-10-30T15:58:19.8239891Z Error: Process completed with exit code 1.
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "network")
        self.assertEqual(issue.error_code, "RATE_LIMIT_EXCEEDED")
        self.assertEqual(issue.error_type, "infrastructure")
        self.assertFalse(issue.actionable)
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_artifact_download_failed(self):
        """Test GitHub artifact download failure (infrastructure flake)."""
        log = """
Test Sharktank / sharktank_tests :: amdgpu_rocm_mi250_gfx90a    UNKNOWN STEP    2025-11-03T23:51:23.0495851Z ##[error]Unable to download artifact(s): Unable to download and extract artifact: Artifact download failed after 5 retries.
Test Sharktank / sharktank_tests :: amdgpu_rocm_mi250_gfx90a    UNKNOWN STEP    2025-11-03T23:54:18.2244146Z Post job cleanup.
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "network")
        self.assertEqual(issue.error_code, "ARTIFACT_DOWNLOAD_FAILED")
        self.assertEqual(issue.error_type, "infrastructure")
        self.assertFalse(issue.actionable)
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_compiler_bus_error(self):
        """Test compiler crash with bus error (hardware issue on CI runner)."""
        log = """
Test AMD W7900 / test_w7900    UNKNOWN STEP    2025-10-13T23:24:54.1865087Z /usr/bin/cc -DCPUINFO_LOG_LEVEL=2 -o third_party/cpuinfo/CMakeFiles/cpuinfo.dir/src/init.c.o -c /home/svcnod/actions-runner-2/_work/iree/iree/third_party/cpuinfo/src/init.c
Test AMD W7900 / test_w7900    UNKNOWN STEP    2025-10-13T23:24:54.1867472Z /home/svcnod/actions-runner-2/_work/iree/iree/third_party/cpuinfo/src/init.c:65:1: internal compiler error: Bus error
Test AMD W7900 / test_w7900    UNKNOWN STEP    2025-10-13T23:24:54.1868111Z    65 | void CPUINFO_ABI cpuinfo_deinitialize(void) {}
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.gpu_type, "hardware")
        self.assertEqual(issue.error_code, "COMPILER_BUS_ERROR")
        self.assertEqual(issue.error_type, "infrastructure")
        self.assertFalse(issue.actionable)
        self.assertEqual(issue.severity, Severity.HIGH)


if __name__ == "__main__":
    unittest.main()
