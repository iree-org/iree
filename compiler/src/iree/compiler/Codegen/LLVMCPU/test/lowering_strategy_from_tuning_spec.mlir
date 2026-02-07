// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(iree-codegen-llvmcpu-configuration-pipeline)' \
// RUN:   --iree-codegen-tuning-spec-path=%p/tuning_spec_matmul.mlir \
// RUN:   --iree-codegen-test-notify-transform-strategy-application \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Make sure we can apply the lowering strategy from the specified tuning spec.

// CHECK:      #[[CONFIG:.+]] = #iree_cpu.lowering_config<
// CHECK-SAME:   distribution = [64, 64, 0]
// CHECK-SAME:   vector_common_parallel = [8, 16, 0]
// CHECK-SAME:   vector_reduction = [0, 0, 8]
// CHECK:      #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
// CHECK:      func.func @matmul
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK:        linalg.matmul
// CHECK-SAME:     __custom_tuning_spec_applied__
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// expected-remark@+1 {{Applied transform configuration strategy @iree_linked_tuning_spec::@__kernel_config}}
func.func @matmul(%lhs: tensor<128x256xf32>, %rhs: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<128x512xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x512xf32>) -> tensor<128x512xf32>
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%fill : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %result : tensor<128x512xf32>
}
