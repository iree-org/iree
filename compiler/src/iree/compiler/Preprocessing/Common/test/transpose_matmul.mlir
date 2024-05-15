// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-preprocessing-transpose-matmul-pass{input=lhs}))" %s | FileCheck %s --check-prefixes=CHECK,LHS
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-preprocessing-transpose-matmul-pass{input=rhs}))" %s | FileCheck %s --check-prefixes=CHECK,RHS

// CHECK-LABEL: @matmul
// LHS: linalg.matmul_transpose_a
// RHS: linalg.matmul_transpose_b
func.func @matmul(%A: tensor<16x8xf32>, %B: tensor<8x16xf32>) -> (tensor<16x16xf32>) {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<16x16xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<16x8xf32>, tensor<8x16xf32>) outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}
