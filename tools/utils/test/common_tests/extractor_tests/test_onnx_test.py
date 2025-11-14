# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for ONNXTestExtractor."""

import unittest

from common.extractors.onnx_test import ONNXTestExtractor
from common.issues import Severity
from common.log_buffer import LogBuffer


class TestONNXTestExtractor(unittest.TestCase):
    """Test ONNXTestExtractor for ONNX test failures."""

    def setUp(self):
        """Set up extractor for tests."""
        self.extractor = ONNXTestExtractor()

    def test_onnx_test_with_full_context(self):
        """Test ONNX test extraction with complete reproduction context."""
        log = """
_ IREE compile and run: test_mean_example::model.mlir::model.mlir::gpu_vulkan __
[gw2] linux -- Python 3.11.13 /home/svcnod/actions-runner-2/_work/iree/iree/venv/bin/python
Error invoking iree-run-module
Error code: 1
Stderr diagnostics:
iree/runtime/src/iree/hal/drivers/vulkan/vulkan_driver.cc:546: NOT_FOUND; physical device 0 invalid; 1 physical devices available; 0 visible; creating device 'vulkan'; resolving dependencies for 'module'; creating VM context; creating run context

Stdout diagnostics:

Test case source:
  https://github.com/iree-org/iree-test-suites/blob/main/onnx_ops/onnx/node/generated/test_mean_example

Input program:
```
module {
  func.func @test_mean_example(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Mean"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }
}
```

Compiled with:
  cd /home/svcnod/actions-runner-2/_work/iree/iree/iree-test-suites/onnx_ops/onnx/node/generated/test_mean_example && iree-compile model.mlir --iree-hal-target-device=vulkan --iree-input-demote-f64-to-f32 --iree-opt-level=O0 -o model_gpu_vulkan.vmfb

Run with:
  cd /home/svcnod/actions-runner-2/_work/iree/iree/iree-test-suites/onnx_ops/onnx/node/generated/test_mean_example && iree-run-module --module=model_gpu_vulkan.vmfb --device=vulkan --flagfile=run_module_io_flags.txt
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]

        # Verify test name extraction.
        self.assertEqual(issue.test_name, "test_mean_example")

        # Verify error details.
        self.assertEqual(issue.error_tool, "iree-run-module")
        self.assertEqual(issue.error_code, 1)
        self.assertIn("physical device 0 invalid", issue.error_message)

        # Verify MLIR extraction.
        self.assertIn("func.func @test_mean_example", issue.input_mlir)
        self.assertIn("onnx.Mean", issue.input_mlir)
        self.assertEqual(len(issue.input_mlir.splitlines()), 7)

        # Verify command extraction.
        self.assertIn("iree-compile model.mlir", issue.compile_command)
        self.assertIn("--iree-hal-target-device=vulkan", issue.compile_command)
        self.assertIn("iree-run-module", issue.run_command)
        self.assertIn("--device=vulkan", issue.run_command)

        # Verify test source URL.
        self.assertEqual(
            issue.test_source_url,
            "https://github.com/iree-org/iree-test-suites/blob/main/onnx_ops/onnx/node/generated/test_mean_example",
        )

        # Verify severity and actionability (infrastructure error in this case).
        self.assertEqual(issue.severity, Severity.HIGH)
        self.assertFalse(issue.actionable)

    def test_onnx_compile_error(self):
        """Test ONNX test with compilation error (actionable)."""
        log = """
______ IREE compile and run: test_conv_example::model.mlir::model.mlir::gpu_vulkan ______
[gw0] linux -- Python 3.11.13 /home/svcnod/actions-runner-2/_work/iree/iree/venv/bin/python
Error invoking iree-compile
Error code: 1
Stderr diagnostics:
mlir/lib/IR/Verifier.cpp:123: Verification failed: unexpected operand type

Stdout diagnostics:

Test case source:
  https://github.com/iree-org/iree-test-suites/blob/main/onnx_ops/onnx/node/generated/test_conv_example

Input program:
```
module {
  func.func @test_conv_example(%arg0: !torch.vtensor<[1,1,5,5],f32>) -> !torch.vtensor<[1,1,5,5],f32> {
    return %arg0 : !torch.vtensor<[1,1,5,5],f32>
  }
}
```

Compiled with:
  cd /home/svcnod/actions-runner-2/_work/iree/iree/iree-test-suites/onnx_ops/onnx/node/generated/test_conv_example && iree-compile model.mlir --iree-hal-target-device=vulkan -o model_gpu_vulkan.vmfb

Run with:
  cd /home/svcnod/actions-runner-2/_work/iree/iree/iree-test-suites/onnx_ops/onnx/node/generated/test_conv_example && iree-run-module --module=model_gpu_vulkan.vmfb --device=vulkan --flagfile=run_module_io_flags.txt
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]

        # Verify compilation error extraction.
        self.assertEqual(issue.test_name, "test_conv_example")
        self.assertEqual(issue.error_tool, "iree-compile")
        self.assertEqual(issue.error_code, 1)
        self.assertIn("Verification failed", issue.error_message)

        # Compilation errors are actionable (code bug).
        self.assertTrue(issue.actionable)
        self.assertEqual(issue.severity, Severity.HIGH)

    def test_onnx_multiple_tests_in_log(self):
        """Test extraction of multiple ONNX test failures in one log."""
        log = """
_ IREE compile and run: test_abs::model.mlir::model.mlir::gpu_vulkan _
[gw1] linux -- Python 3.11.13 /home/svcnod/actions-runner-2/_work/iree/iree/venv/bin/python
Error invoking iree-run-module
Error code: 1
Stderr diagnostics:
vulkan_driver.cc:546: NOT_FOUND; physical device 0 invalid

Stdout diagnostics:

Test case source:
  https://github.com/iree-org/iree-test-suites/blob/main/onnx_ops/onnx/node/generated/test_abs

Input program:
```
module {
  func.func @test_abs(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> {
    %0 = torch.operator "onnx.Abs"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }
}
```

Compiled with:
  iree-compile model.mlir -o model.vmfb

Run with:
  iree-run-module --module=model.vmfb --device=vulkan


______ IREE compile and run: test_acos::model.mlir::model.mlir::gpu_vulkan ______
[gw2] linux -- Python 3.11.13 /home/svcnod/actions-runner-2/_work/iree/iree/venv/bin/python
Error invoking iree-run-module
Error code: 1
Stderr diagnostics:
vulkan_driver.cc:546: NOT_FOUND; physical device 0 invalid

Stdout diagnostics:

Test case source:
  https://github.com/iree-org/iree-test-suites/blob/main/onnx_ops/onnx/node/generated/test_acos

Input program:
```
module {
  func.func @test_acos(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> {
    %0 = torch.operator "onnx.Acos"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }
}
```

Compiled with:
  iree-compile model.mlir -o model.vmfb

Run with:
  iree-run-module --module=model.vmfb --device=vulkan
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        # Should extract both tests.
        self.assertEqual(len(issues), 2)

        # Verify test names.
        test_names = {issue.test_name for issue in issues}
        self.assertEqual(test_names, {"test_abs", "test_acos"})

        # Both should have same error (infrastructure flake).
        for issue in issues:
            self.assertEqual(issue.error_tool, "iree-run-module")
            self.assertIn("physical device", issue.error_message)
            self.assertFalse(issue.actionable)

    def test_onnx_test_missing_some_context(self):
        """Test ONNX test with partial context (missing MLIR or commands)."""
        log = """
_ IREE compile and run: test_partial::model.mlir::model.mlir::gpu_vulkan _
[gw0] linux -- Python 3.11.13 /venv/bin/python
Error invoking iree-run-module
Error code: 1
Stderr diagnostics:
Some error occurred

Stdout diagnostics:
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        # Should still extract even without full context.
        self.assertEqual(len(issues), 1)
        issue = issues[0]

        self.assertEqual(issue.test_name, "test_partial")
        self.assertEqual(issue.error_tool, "iree-run-module")
        self.assertEqual(issue.error_code, 1)
        self.assertIn("Some error occurred", issue.error_message)

        # Context may be missing.
        self.assertEqual(issue.input_mlir, "")
        self.assertEqual(issue.compile_command, "")
        self.assertEqual(issue.run_command, "")

    def test_no_onnx_tests(self):
        """Test log with no ONNX test failures."""
        log = """
Running some other tests...
All tests passed successfully
No ONNX test errors detected
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 0)

    def test_vulkan_device_invalid_infrastructure_classification(self):
        """Test that Vulkan device invalid errors are classified as infrastructure flakes."""
        log = """
_ IREE compile and run: test_gpu::model.mlir::model.mlir::gpu_vulkan _
[gw0] linux -- Python 3.11.13 /venv/bin/python
Error invoking iree-run-module
Error code: 1
Stderr diagnostics:
vulkan_driver.cc:546: NOT_FOUND; physical device 0 invalid

Stdout diagnostics:

Test case source:
  https://github.com/iree-org/iree-test-suites/blob/main/onnx_ops/test_gpu

Input program:
```
module { }
```

Compiled with:
  iree-compile model.mlir -o model.vmfb

Run with:
  iree-run-module --module=model.vmfb --device=vulkan
"""
        log_buffer = LogBuffer(log)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        issue = issues[0]

        # Verify infrastructure flake classification.
        self.assertFalse(issue.actionable)
        self.assertEqual(issue.severity, Severity.HIGH)
        self.assertIn("physical device", issue.error_message)


if __name__ == "__main__":
    unittest.main()
