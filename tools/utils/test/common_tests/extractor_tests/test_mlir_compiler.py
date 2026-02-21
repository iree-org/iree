# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for MLIRCompilerExtractor.

Tests cover:
- Simple errors with source context
- Multi-line errors with operation dumps
- Operation name extraction (from message and source)
- Note collection (see current operation, called from)
- FileCheck error exclusion
- Error type inference
- Test target inference
"""

import unittest

from common.extractors.mlir_compiler import MLIRCompilerExtractor
from common.issues import Severity
from common.log_buffer import LogBuffer


class TestMLIRCompilerExtraction(unittest.TestCase):
    """Test MLIR compiler error extraction."""

    def setUp(self):
        self.extractor = MLIRCompilerExtractor()

    def test_simple_error_with_source_context(self):
        """Test simple MLIR error with source snippet and caret."""
        log_content = """
artifacts/model_zoo/validated/vision/classification/vgg19-7.mlir:80:11: error: slice along dimension 3 runs out-of-bounds: 255 >= 16
    %76 = torch.operator "onnx.Gemm"(%75, %32, %33) {...}
          ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(
            issue.file_path,
            "artifacts/model_zoo/validated/vision/classification/vgg19-7.mlir",
        )
        self.assertEqual(issue.error_line, 80)
        self.assertEqual(issue.error_column, 11)
        self.assertEqual(
            issue.error_message, "slice along dimension 3 runs out-of-bounds: 255 >= 16"
        )
        self.assertEqual(issue.error_type, "out-of-bounds")
        self.assertEqual(issue.severity, Severity.HIGH)
        self.assertTrue(issue.actionable)

        # Check source snippet and caret.
        self.assertIn('torch.operator "onnx.Gemm"', issue.source_snippet)
        self.assertIn("^", issue.caret_line)

        # Check full diagnostic.
        self.assertIn("error: slice along dimension", issue.full_diagnostic)

    def test_error_with_operation_dump_note(self):
        """Test error with 'note: see current operation:' and multi-line dump."""
        log_content = """
model.mlir:4:10: error: operand #0 does not dominate this use
    %0 = torch.operator "onnx.TfIdfVectorizer"(%arg0) {...}
         ^
model.mlir:4:10: note: see current operation:
%38:3 = "stream.async.execute"(%23#0, %6, %5, %5, %23#2) <{affinity = #hal.device.affinity<@dev>}> ({
^bb0(%arg7: !stream.resource<external>):
  %50 = "stream.async.slice"(%arg7, %6, %33, %34, %5) : (...)
  %51 = "stream.async.dispatch"(...) : (...) -> !stream.resource<external>
  stream.yield %50, %51 : !stream.resource<external>
}) : (...) -> (!stream.timepoint, !stream.resource<external>, !stream.resource<external>)
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.file_path, "model.mlir")
        self.assertEqual(issue.error_line, 4)
        self.assertEqual(issue.error_column, 10)
        self.assertEqual(issue.error_message, "operand #0 does not dominate this use")
        self.assertEqual(issue.error_type, "operand-does-not-dominate")

        # Check notes.
        self.assertEqual(len(issue.notes), 1)
        note = issue.notes[0]
        self.assertEqual(note["file"], "model.mlir")
        self.assertEqual(note["line"], 4)
        self.assertEqual(note["column"], 10)
        self.assertEqual(note["type"], "see current operation:")

        # Check operation dump in note body.
        self.assertIn("stream.async.execute", note["body"])
        self.assertIn("stream.async.slice", note["body"])
        self.assertIn("stream.yield", note["body"])

    def test_multiple_notes_chain(self):
        """Test error with multiple notes (called from chain)."""
        log_content = """
/home/runner/_work/iree/iree/tests/e2e/linalg/argmax.mlir:17:15: error: 'linalg.generic' op Failed to analyze the reduction operation.
  %result:2 = linalg.generic {
              ^
/home/runner/_work/iree/iree/tests/e2e/linalg/argmax.mlir:17:15: note: called from
  %result:2 = linalg.generic {
              ^
/home/runner/_work/iree/iree/tests/e2e/linalg/argmax.mlir:17:15: note: failed to tile operation
  %result:2 = linalg.generic {
              ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(
            issue.file_path, "/home/runner/_work/iree/iree/tests/e2e/linalg/argmax.mlir"
        )
        self.assertEqual(issue.error_line, 17)
        self.assertEqual(issue.error_column, 15)
        self.assertEqual(issue.operation, "linalg.generic")
        self.assertEqual(issue.error_type, "failed-to-analyze")

        # Check multiple notes.
        self.assertEqual(len(issue.notes), 2)
        self.assertEqual(issue.notes[0]["type"], "called from")
        self.assertEqual(issue.notes[1]["type"], "failed to tile operation")

        # Check test target inference.
        self.assertEqual(issue.test_target, "tests/e2e/linalg/argmax.mlir")

    def test_operation_name_extraction_from_message(self):
        """Test operation name extracted from error message: 'op_name' op."""
        log_content = """
/compiler/test/fold_unit_dims.mlir:175:8: error: 'iree_linalg_ext.map_scatter' op expected input type to have non-zero rank
  %0 = iree_linalg_ext.map_scatter %input into %empty {
       ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.operation, "iree_linalg_ext.map_scatter")
        self.assertIn("iree_linalg_ext.map_scatter", issue.error_message)

    def test_operation_name_extraction_from_source(self):
        """Test operation name extracted from source when not in message."""
        log_content = """
test.mlir:10:5: error: failed to materialize conversion
    %1 = tensor.extract_slice %0[0, 0] [16, 32] [1, 1] : tensor<16x32xf32>
    ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        # Operation name extracted from source: %1 = tensor.extract_slice ...
        self.assertEqual(issue.operation, "tensor.extract_slice")

    def test_exclude_filecheck_errors(self):
        """Test that FileCheck errors are excluded (test infrastructure, not compiler)."""
        log_content = """
test.mlir:307:12: error: CHECK: expected string not found in input
 // CHECK: %[[IF_RESULT:.+]] = scf.if
           ^
<stdin>:248:56: note: scanning from here
  %25 = scf.if %24 -> (i32) {
                                                       ^
test.mlir:307:12: note: possible intended match here
 // CHECK: %[[IF_RESULT:.+]] = scf.if
           ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        # Should be zero issues (FileCheck errors excluded).
        self.assertEqual(len(issues), 0)

    def test_error_type_inference(self):
        """Test error type categorization from message."""
        test_cases = [
            ("slice runs out-of-bounds", "out-of-bounds"),
            ("does not dominate this use", "does-not-dominate"),
            ("Failed to tile operation", "failed-to-tile"),
            ("Failed to analyze the reduction", "failed-to-analyze"),
            ("type mismatch in operation", "type-mismatch"),
            ("invalid operation detected", "invalid-operation"),
            ("expected input type to have", "expectation-failed"),
        ]

        for error_msg, expected_type in test_cases:
            inferred = self.extractor._infer_error_type(error_msg)
            self.assertEqual(
                inferred,
                expected_type,
                f"Error message '{error_msg}' should infer type '{expected_type}', got '{inferred}'",
            )

    def test_test_target_inference(self):
        """Test inference of test target from file paths."""
        test_cases = [
            (
                "/home/runner/_work/iree/iree/tests/e2e/linalg/argmax.mlir",
                "tests/e2e/linalg/argmax.mlir",
            ),
            (
                "/path/to/iree/compiler/src/iree/compiler/Dialect/test/foo.mlir",
                "compiler/src/iree/compiler/Dialect/test/foo.mlir",
            ),
            (
                "artifacts/model_zoo/validated/vision/vgg19.mlir",
                "artifacts/model_zoo/validated/vision/vgg19.mlir",
            ),
            ("model.mlir", None),  # No test context - can't infer
        ]

        for file_path, expected_target in test_cases:
            inferred = self.extractor._infer_test_target(file_path)
            self.assertEqual(
                inferred,
                expected_target,
                f"File path '{file_path}' should infer target '{expected_target}', got '{inferred}'",
            )

    def test_multiple_errors_in_log(self):
        """Test extraction of multiple separate errors from same log."""
        log_content = """
test1.mlir:10:5: error: type mismatch
    %1 = test.op %0 : i32
    ^

test2.mlir:20:10: error: invalid operation
    %2 = invalid.op %1
         ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 2)

        # First error.
        self.assertEqual(issues[0].file_path, "test1.mlir")
        self.assertEqual(issues[0].error_line, 10)
        self.assertEqual(issues[0].error_message, "type mismatch")

        # Second error.
        self.assertEqual(issues[1].file_path, "test2.mlir")
        self.assertEqual(issues[1].error_line, 20)
        self.assertEqual(issues[1].error_message, "invalid operation")

    def test_graceful_degradation_missing_source(self):
        """Test extraction works even without source snippet."""
        log_content = """
test.mlir:10:5: error: compilation failed
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        # Should still extract the error, just without source snippet.
        self.assertEqual(len(issues), 1)

        issue = issues[0]
        self.assertEqual(issue.file_path, "test.mlir")
        self.assertEqual(issue.error_line, 10)
        self.assertEqual(issue.error_message, "compilation failed")
        self.assertEqual(issue.source_snippet, "")  # No source available
        self.assertEqual(issue.caret_line, "")  # No caret available

    def test_full_diagnostic_completeness(self):
        """Test that full_diagnostic captures complete error text."""
        log_content = """
test.mlir:10:5: error: failed to compile
    %1 = test.op %0
    ^
test.mlir:10:5: note: see operation dump
  %1 = "test.op"(%0) : (i32) -> i32
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)

        issue = issues[0]
        # Full diagnostic should include error + source + note.
        self.assertIn("error: failed to compile", issue.full_diagnostic)
        self.assertIn("%1 = test.op %0", issue.full_diagnostic)
        self.assertIn("note: see operation dump", issue.full_diagnostic)
        self.assertIn('"test.op"', issue.full_diagnostic)

    def test_error_type_operand_dominate_specific(self):
        """Test operand-does-not-dominate is distinct from does-not-dominate."""
        # Specific operand error should get specific type.
        operand_msg = "operand #0 does not dominate this use"
        self.assertEqual(
            self.extractor._infer_error_type(operand_msg),
            "operand-does-not-dominate",
        )

        # General dominate error should get general type.
        general_msg = "value does not dominate this use"
        self.assertEqual(
            self.extractor._infer_error_type(general_msg),
            "does-not-dominate",
        )

    def test_compile_command_compiled_with(self):
        """Test extraction of compile command with 'Compiled with:' marker (Pattern 1)."""
        log_content = """
model.mlir:4:10: error: 'linalg.generic' op Failed to analyze the reduction operation.
    %0 = torch.operator "onnx.ArgMax"(%arg0) {...}
         ^
model.mlir:4:10: note: see current operation:
%7:2 = "linalg.generic"(...) : (...) -> (tensor<2xf32>, tensor<2xi64>)

Compiled with:
  cd /home/nod/actions-runner/_work/iree/iree/iree-test-suites/onnx_ops && iree-compile model.mlir --iree-hal-target-device=vulkan --iree-input-demote-f64-to-f32 --iree-opt-level=O0 -o model_gpu_vulkan.vmfb
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        self.assertIsNotNone(issues[0].compile_command)
        self.assertIn("iree-compile model.mlir", issues[0].compile_command)
        self.assertIn("--iree-hal-target-device=vulkan", issues[0].compile_command)
        self.assertIn("cd /home/nod/actions-runner", issues[0].compile_command)

    def test_compile_command_cmake_cd(self):
        """Test extraction of compile command from CMake 'cd ... &&' pattern (Pattern 2)."""
        log_content = """
cd /groups/aig_sharks/actions-runner/_work/iree/iree/build-tests/tests/e2e/linalg && /groups/aig_sharks/actions-runner/_work/iree/iree/.venv/bin/iree-compile --output-format=vm-bytecode --iree-hal-target-backends=rocm --iree-hip-target=gfx90a /groups/aig_sharks/actions-runner/_work/iree/iree/tests/e2e/linalg/argmax.mlir -o check_rocm_hip_argmax.mlir_module.vmfb --iree-hal-executable-object-search-path=\\"/groups/aig_sharks/actions-runner/_work/iree/iree/build-tests\\"
/groups/aig_sharks/actions-runner/_work/iree/iree/tests/e2e/linalg/argmax.mlir:17:15: error: 'linalg.generic' op Failed to analyze the reduction operation.
  %result:2 = linalg.generic {
              ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        self.assertIsNotNone(issues[0].compile_command)
        self.assertIn(
            "iree-compile --output-format=vm-bytecode", issues[0].compile_command
        )
        self.assertIn("--iree-hal-target-backends=rocm", issues[0].compile_command)
        # Verify quote unescaping worked.
        self.assertIn(
            '--iree-hal-executable-object-search-path="/groups',
            issues[0].compile_command,
        )
        self.assertNotIn('\\"', issues[0].compile_command)

    def test_compile_command_lit_xtrace(self):
        """Test extraction of compile command from lit shell xtrace (Pattern 3)."""
        log_content = """
+ iree-opt '--pass-pipeline=builtin.module(func.func(iree-codegen-gpu-convert-to-coalesced-dma,canonicalize))' /__w/iree/iree/compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_convert_to_coalesced_dma.mlir --split-input-file
/__w/iree/iree/compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_convert_to_coalesced_dma.mlir:10:5: error: failed to materialize conversion
    %1 = tensor.extract_slice %0[0, 0] [16, 32] [1, 1] : tensor<16x32xf32>
    ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        self.assertIsNotNone(issues[0].compile_command)
        self.assertIn("iree-opt", issues[0].compile_command)
        self.assertIn("--pass-pipeline=", issues[0].compile_command)
        # Verify shell xtrace prefix was stripped.
        self.assertFalse(issues[0].compile_command.startswith("+ "))

    def test_compile_command_not_found(self):
        """Test graceful handling when no compile command found."""
        log_content = """
model.mlir:10:5: error: compilation failed
    %1 = test.op %0
    ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        self.assertIsNone(issues[0].compile_command)

    def test_compile_command_ci_prefix_stripped(self):
        """Test that CI log prefixes are properly stripped from commands."""
        log_content = """
Test ONNX / test_onnx_ops :: amdgpu_vulkan_O0\tUNKNOWN STEP\t2025-11-12T03:13:55.5396889Z model.mlir:4:10: error: 'linalg.generic' op Failed to analyze
Test ONNX / test_onnx_ops :: amdgpu_vulkan_O0\tUNKNOWN STEP\t2025-11-12T03:13:55.5410291Z Compiled with:
Test ONNX / test_onnx_ops :: amdgpu_vulkan_O0\tUNKNOWN STEP\t2025-11-12T03:13:55.5410687Z   cd /home/nod/iree-test-suites && iree-compile model.mlir -o output.vmfb
"""
        log_buffer = LogBuffer(log_content, auto_detect_format=True)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        self.assertIsNotNone(issues[0].compile_command)
        # Verify CI prefix was stripped.
        self.assertNotIn("Test ONNX", issues[0].compile_command)
        self.assertNotIn("UNKNOWN STEP", issues[0].compile_command)
        self.assertIn("cd /home/nod/iree-test-suites", issues[0].compile_command)

    def test_source_snippet_stops_at_diagnostic(self):
        """Test source snippet collection stops at next error/warning."""
        log_content = """
test.mlir:10:5: error: first error
    %1 = test.op %0
test.mlir:11:5: error: second error immediately after
    %2 = another.op %1
    ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 2)
        self.assertIn("%1 = test.op %0", issues[0].source_snippet)
        self.assertNotIn("second error", issues[0].source_snippet)
        self.assertNotIn("%2", issues[0].source_snippet)

    def test_compile_command_onnx_compiled_with_large_ir(self):
        """Test ONNX 'Compiled with:' extraction with large IR dump (Pattern 1 + adaptive boundary)."""
        log_content = """
Error invoking iree-compile
Error code: 1
Stderr diagnostics:
model.mlir:4:10: error: 'linalg.generic' op Failed to anaysis the reduction operation.
    %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64>
         ^
model.mlir:4:10: note: see current operation:
%7:2 = "linalg.generic"(%2, %6, %4) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<reduction>]}> ({
^bb0(%arg3: f32, %arg4: f32, %arg5: i64):
  %11 = "linalg.index"() <{dim = 0 : i64}> : () -> index
  "linalg.yield"(%13, %15) : (f32, i64) -> ()
}) : (tensor<2x2xf32>, tensor<2xf32>, tensor<2xi64>) -> (tensor<2xf32>, tensor<2xi64>)
model.mlir:4:10: error: 'linalg.generic' op failed to tile using scf.forall
    %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64>
         ^
model.mlir:4:10: note: see current operation:
%7:2 = "linalg.generic"(%2, %6, %4) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>]}> ({
^bb0(%arg3: f32, %arg4: f32, %arg5: i64):
  "linalg.yield"(%13, %15) : (f32, i64) -> ()
}) : (tensor<2x2xf32>, tensor<2xf32>, tensor<2xi64>) -> (tensor<2xf32>, tensor<2xi64>)

Stdout diagnostics:

Test case source:
  https://github.com/iree-org/iree-test-suites/blob/main/onnx_ops/onnx/node/generated/test_argmax_default_axis_example

Input program:
```
module {
  func.func @test_argmax_default_axis_example(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],si64> {
    %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.keepdims = 1 : si64}
    return %0 : !torch.vtensor<[1,2],si64>
  }
}
```

Compiled with:
  cd /home/runner/work/iree/iree/iree-test-suites/onnx_ops/onnx/node/generated/test_argmax_default_axis_example && iree-compile model.mlir --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-input-demote-f64-to-f32=false --iree-opt-level=O0 -o model_cpu_llvm_sync.vmfb
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 2)  # Two errors extracted.
        # Both errors should have same compile command.
        for issue in issues:
            self.assertIsNotNone(issue.compile_command)
            self.assertIn(
                "cd /home/runner/work/iree/iree/iree-test-suites", issue.compile_command
            )
            self.assertIn("iree-compile model.mlir", issue.compile_command)
            self.assertIn("--iree-hal-target-device=local", issue.compile_command)
        # Verify adaptive boundary detection worked (marker is ~35 lines after first error).

    def test_compile_command_python_api_invoked_with(self):
        """Test Python API 'Invoked with:' extraction (Pattern 2)."""
        log_content = """
Traceback (most recent call last):
  File "/__w/iree/iree/samples/sink_callback/sink_callback_log.py", line 97, in <module>
    main()
  File "/__w/iree/iree/samples/sink_callback/sink_callback_log.py", line 19, in main
    vmfb_contents = compiler.compile_file(
  File "/__w/iree/iree/build-byo-llvm/iree/compiler/bindings/python/iree/compiler/tools/binaries.py", line 201, in invoke_immediate
    raise CompilerToolError(process)
iree.compiler.tools.binaries.CompilerToolError: Error invoking IREE compiler tool iree-compile
Error code: 1
Diagnostics:
IREE was not built with support for LLD
Linking failed; escaped command line returned exit code 256:

/__w/iree/iree/samples/sink_callback/model.mlir:0:0: error: failed to link executable and generate target dylib (check above for more specific error messages)
/__w/iree/iree/samples/sink_callback/model.mlir:0:0: error: failed to serialize executable for target backend llvm-cpu
/__w/iree/iree/samples/sink_callback/model.mlir:0:0: error: failed to serialize executables


Invoked with:
 iree-compile /__w/iree/iree/build-byo-llvm/iree/compiler/bindings/python/iree/compiler/tools/../_mlir_libs/iree-compile /__w/iree/iree/samples/sink_callback/model.mlir --iree-input-type=auto --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=llvm-cpu --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --iree-llvmcpu-target-cpu=host
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 3)  # Three errors extracted.
        # All should have same compile command.
        for issue in issues:
            self.assertIsNotNone(issue.compile_command)
            self.assertIn("iree-compile", issue.compile_command)
            self.assertIn("model.mlir", issue.compile_command)
            self.assertIn("--iree-input-type=auto", issue.compile_command)
            # Verify leading space was stripped.
            self.assertFalse(issue.compile_command.startswith(" "))

    def test_compile_command_cmake_backward_search(self):
        """Test CMake 'cd &&' backward proximity search (Pattern 3)."""
        log_content = """
[8413/8447] Generating simple_mul_module.vmfb from simple_mul.mlir
FAILED: runtime/src/iree/runtime/demo/simple_mul_module.vmfb
cd /__w/iree/iree/build-tsan/runtime/src/iree/runtime/demo && /__w/iree/iree/build-tsan/tools/iree-compile --output-format=vm-bytecode --mlir-print-op-on-diagnostic=false --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx /__w/iree/iree/runtime/src/iree/runtime/demo/simple_mul.mlir -o simple_mul_module.vmfb --iree-hal-executable-object-search-path=\"/__w/iree/iree/build-tsan\"
simple_mul.mlir:5:10: error: 'arith.addi' op result type 'i32' does not match operand type 'i64'
  %result = arith.addi %arg0, %arg1 : i64
          ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        self.assertIsNotNone(issues[0].compile_command)
        # Verify command extracted from line BEFORE error (backward search).
        self.assertIn("cd /__w/iree/iree/build-tsan", issues[0].compile_command)
        self.assertIn(
            "iree-compile --output-format=vm-bytecode", issues[0].compile_command
        )
        # Verify quote unescaping worked.
        self.assertIn(
            '--iree-hal-executable-object-search-path="/__w/iree/iree/build-tsan"',
            issues[0].compile_command,
        )
        self.assertNotIn(r"\"", issues[0].compile_command)

    def test_compile_command_lit_shell_xtrace(self):
        """Test Lit/Bazel shell xtrace extraction (Pattern 4)."""
        log_content = """
FAIL: IREE :: src/iree/compiler/Codegen/Dialect/GPU/IR/test/iree_gpu_ops.mlir (1 of 1)
******************** TEST 'IREE :: src/iree/compiler/Codegen/Dialect/GPU/IR/test/iree_gpu_ops.mlir' FAILED ********************
Exit Code: 1

Command Output (stderr):
--
ERROR: ld.so: object 'libvulkan.so.1' from LD_PRELOAD cannot be preloaded: ignored.
+ iree-opt /dev/shm/bazel-sandbox.77efa1afd21a6a756aca4546a3434b69c714acccffdac2a35c681913d3c6fbf7/processwrapper-sandbox/13947/execroot/_main/bazel-out/k8-opt/bin/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/test/iree_gpu_ops.mlir.test.runfiles/_main/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/test/iree_gpu_ops.mlir --split-input-file --verify-diagnostics
+ FileCheck /dev/shm/bazel-sandbox.77efa1afd21a6a756aca4546a3434b69c714acccffdac2a35c681913d3c6fbf7/processwrapper-sandbox/13947/execroot/_main/bazel-out/k8-opt/bin/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/test/iree_gpu_ops.mlir.test.runfiles/_main/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/test/iree_gpu_ops.mlir
ERROR: ld.so: object 'libvulkan.so.1' from LD_PRELOAD cannot be preloaded: ignored.
/dev/shm/bazel-sandbox.77efa1afd21a6a756aca4546a3434b69c714acccffdac2a35c681913d3c6fbf7/iree_gpu_ops.mlir:104:7: error: unexpected error: 'iree_gpu.coalesced_gather_dma' op operand #1 must be variadic of ranked tensor of 32-bit signless integer values
      iree_gpu.coalesced_gather_dma %source[%idx0] into %out lane(%lane)
      ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        self.assertIsNotNone(issues[0].compile_command)
        # Verify shell xtrace command extracted (backward search).
        self.assertIn("iree-opt", issues[0].compile_command)
        self.assertIn("/dev/shm/bazel-sandbox", issues[0].compile_command)
        self.assertIn(
            "--split-input-file --verify-diagnostics", issues[0].compile_command
        )
        # Verify "+ " prefix was stripped.
        self.assertFalse(issues[0].compile_command.startswith("+ "))

    def test_compile_command_windows_cmake(self):
        """Test Windows-style CMake command extraction (Pattern 3 with cmd.exe wrapper)."""
        log_content = """
[8413/8447] Generating simple_mul_module.vmfb from simple_mul.mlir
FAILED: runtime/src/iree/runtime/demo/simple_mul_module.vmfb
C:\\Windows\\system32\\cmd.exe /C "cd /D B:\\tmpbuild\\runtime\\src\\iree\\runtime\\demo && B:\\tmpbuild\\tools\\iree-compile.exe --output-format=vm-bytecode --mlir-print-op-on-diagnostic=false --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx B:\\iree\\runtime\\src\\iree\\runtime\\demo\\simple_mul.mlir -o simple_mul_module.vmfb"
simple_mul.mlir:5:10: error: 'arith.addi' op result type 'i32' does not match operand type 'i64'
  %result = arith.addi %arg0, %arg1 : i64
          ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        self.assertIsNotNone(issues[0].compile_command)
        # Verify Windows command extracted (cd is NOT at line start due to cmd.exe wrapper).
        self.assertIn("cd /D B:\\tmpbuild", issues[0].compile_command)
        self.assertIn("iree-compile.exe", issues[0].compile_command)
        # Verify this works because we use search() not match().
        self.assertIn("--output-format=vm-bytecode", issues[0].compile_command)

    def test_compile_command_pattern_priority(self):
        """Test that marker-based patterns (1,2) take precedence over proximity-based (3,4)."""
        log_content = """
+ iree-compile /tmp/test.mlir --iree-hal-target-backends=llvm-cpu
test.mlir:10:5: error: 'linalg.generic' op failed to tile
    %0 = linalg.generic {
         ^

[... 40 lines of IR dump ...]

Compiled with:
  cd /home/user/iree && iree-compile /tmp/test.mlir --iree-hal-target-backends=vulkan --iree-vulkan-target-triple=valhall-unknown-android31
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        self.assertEqual(len(issues), 1)
        self.assertIsNotNone(issues[0].compile_command)
        # Should extract marker-based command (Pattern 1), NOT proximity-based (Pattern 4).
        self.assertIn("--iree-hal-target-backends=vulkan", issues[0].compile_command)
        self.assertIn("--iree-vulkan-target-triple=valhall", issues[0].compile_command)
        self.assertNotIn("llvm-cpu", issues[0].compile_command)


class TestFalsePositivePrevention(unittest.TestCase):
    """Test that extractor doesn't produce false positives."""

    def setUp(self):
        self.extractor = MLIRCompilerExtractor()

    def test_no_match_non_mlir_file(self):
        """Test C++ compiler errors don't match."""
        log_content = """
test.cpp:10:5: error: expected ';' at end of declaration
    int x = 42
    ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        # Should find nothing (not .mlir file).
        self.assertEqual(len(issues), 0)

    def test_no_match_normal_log(self):
        """Test normal build log produces no issues."""
        log_content = """
[ 25%] Building MLIR test.mlir
[ 50%] Compiling model.mlir
[100%] Built target iree-compile
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        # Should find nothing.
        self.assertEqual(len(issues), 0)

    def test_no_match_filecheck_only(self):
        """Test log with only FileCheck errors produces no issues."""
        log_content = """
test.mlir:10:12: error: CHECK: expected string not found
 // CHECK: scf.if
           ^
"""
        log_buffer = LogBuffer(log_content)
        issues = self.extractor.extract(log_buffer)

        # Should find nothing (FileCheck excluded).
        self.assertEqual(len(issues), 0)


if __name__ == "__main__":
    unittest.main()
