// RUN: iree-opt -iree-transform-dialect-interpreter %s | FileCheck %s

// CHECK-LABEL: @matmul
func.func @matmul(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  //      CHECK: %[[RESULT:.+]] = linalg.matmul
  // CHECK-SAME:     tile_sizes = 64 : index
  %result = linalg.matmul ins(%lhs, %rhs: tensor<128x128xf32>, tensor<128x128xf32>)
             outs(%fill: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %workgroup_tile_sizes, %problem_specific_sizes = transform.iree.create_matmul_mfma_tile_sizes %matmul : (!transform.any_op) -> (!transform.any_param, !transform.any_param)
    %doubled_matmul = transform.merge_handles %matmul, %matmul : !transform.any_op
    transform.annotate %doubled_matmul "tile_sizes" = %workgroup_tile_sizes : !transform.any_op, !transform.any_param
    transform.yield
  }
}
