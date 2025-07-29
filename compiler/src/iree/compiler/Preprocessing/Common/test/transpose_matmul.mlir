// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-preprocessing-transpose-matmul-pass{input=lhs}))" %s | FileCheck %s --check-prefixes=CHECK,LHS
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-preprocessing-transpose-matmul-pass{input=rhs}))" %s | FileCheck %s --check-prefixes=CHECK,RHS

// LHS-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// LHS-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// LHS-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// RHS-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// RHS-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// RHS-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: @matmul
// LHS: linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
// RHS: linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
func.func @matmul(%A: tensor<16x8xf32>, %B: tensor<8x16xf32>) -> (tensor<16x16xf32>) {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<16x16xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<16x8xf32>, tensor<8x16xf32>) outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}
