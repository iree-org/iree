// RUN: iree-opt -iree-transform-dialect-interpreter %s | FileCheck %s

func.func @matmul(%lhs: tensor<128x128xf16>, %rhs: tensor<128x128xf16>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x128xf32>
  %inital_result = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<128x128xf16>, tensor<128x128xf16>)
             outs(%inital_result: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tile_sizes0, %tile_sizes1 = transform.iree.create_matmul_mfma_tile_sizes %matmul : (!transform.any_op) -> (!transform.any_param, !transform.any_param)
    // %1:2 = transform.structured.tile_using_forall %matmul tile_sizes *(%tile_sizes0 : !transform.any_param)
    //        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
