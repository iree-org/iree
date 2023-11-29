// RUN: iree-opt --iree-global-opt-infer-numeric-narrowing %s | FileCheck %s
// This does not test all of the analysis logic, just that the annotations
// are inserted at proper points in the right way. Probe points checked:
//   - Every operand of a LinalgOp

// CHECK-LABEL: @probe_linalg_op
// Checks as a by-product:
//   - Infering ui0 for [0, 0] range
//   - Infering unsigned for >= 0 range
func.func @probe_linalg_op(%arg0 : tensor<5x3xf32>) -> tensor<5x1xf32> {
  // CHECK-DAG: %[[RHS:.*]] = arith.constant dense
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: util.numeric.optional_narrow %[[ZERO]] : f32 as ui0
  // CHECK-DAG: util.numeric.optional_narrow %[[RHS]] : tensor<3x1xf32> as ui7 {max_value = 127 : ui7, min_value = 0 : ui7}
  // CHECK-DAG: %[[FILL:.*]] = linalg.fill
  // CHECK-DAG: util.numeric.optional_narrow %[[FILL]] : tensor<5x1xf32> as ui0
  %rhs = arith.constant dense<
    [[3.900000e+01], [0.000000e+00], [1.270000e+02]]> : tensor<3x1xf32>
  %init_value = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.fill ins(%init_value : f32) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %2 = linalg.matmul ins(%arg0, %rhs : tensor<5x3xf32>, tensor<3x1xf32>) outs(%1 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %2 : tensor<5x1xf32>
}

// CHECK-LABEL: @infer_symmetric_signed
// CHECK: util.numeric.optional_narrow %{{.*}} : tensor<3x1xf32> as si8 {max_value = 127 : si8, min_value = -39 : si8}
func.func @infer_symmetric_signed(%arg0 : tensor<5x3xf32>) -> tensor<5x1xf32> {
  %rhs = arith.constant dense<
    [[-3.900000e+01], [0.000000e+00], [1.270000e+02]]> : tensor<3x1xf32>
  %init_value = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.fill ins(%init_value : f32) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %2 = linalg.matmul ins(%arg0, %rhs : tensor<5x3xf32>, tensor<3x1xf32>) outs(%1 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %2 : tensor<5x1xf32>
}

// CHECK-LABEL: @infer_i1_signed
// Signed i1 is a silly boundary condition worth checking.
// CHECK: util.numeric.optional_narrow %{{.*}} : tensor<3x1xf32> as si1 {max_value = 0 : si1, min_value = -1 : si1}
func.func @infer_i1_signed(%arg0 : tensor<5x3xf32>) -> tensor<5x1xf32> {
  %rhs = arith.constant dense<
    [[0.000000e+00], [0.000000e+00], [-1.000000e+00]]> : tensor<3x1xf32>
  %init_value = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.fill ins(%init_value : f32) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %2 = linalg.matmul ins(%arg0, %rhs : tensor<5x3xf32>, tensor<3x1xf32>) outs(%1 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %2 : tensor<5x1xf32>
}

// CHECK-LABEL: @infer_positive_non_straddling_zero
// A range that does not straddle zero is a special case in the code.
// CHECK: util.numeric.optional_narrow %{{.*}} : tensor<3x1xf32> as ui2 {max_value = 2 : ui2, min_value = 1 : ui2}
func.func @infer_positive_non_straddling_zero(%arg0 : tensor<5x3xf32>) -> tensor<5x1xf32> {
  %rhs = arith.constant dense<
    [[1.000000e+00], [1.000000e+00], [2.000000e+00]]> : tensor<3x1xf32>
  %init_value = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.fill ins(%init_value : f32) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %2 = linalg.matmul ins(%arg0, %rhs : tensor<5x3xf32>, tensor<3x1xf32>) outs(%1 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %2 : tensor<5x1xf32>
}

// CHECK-LABEL: @infer_negative_non_straddling_zero
// A range that does not straddle zero is a special case in the code.
// CHECK: util.numeric.optional_narrow %{{.*}} : tensor<3x1xf32> as si2 {max_value = -1 : si2, min_value = -2 : si2}
func.func @infer_negative_non_straddling_zero(%arg0 : tensor<5x3xf32>) -> tensor<5x1xf32> {
  %rhs = arith.constant dense<
    [[-1.000000e+00], [-1.000000e+00], [-2.000000e+00]]> : tensor<3x1xf32>
  %init_value = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.fill ins(%init_value : f32) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %2 = linalg.matmul ins(%arg0, %rhs : tensor<5x3xf32>, tensor<3x1xf32>) outs(%1 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %2 : tensor<5x1xf32>
}
