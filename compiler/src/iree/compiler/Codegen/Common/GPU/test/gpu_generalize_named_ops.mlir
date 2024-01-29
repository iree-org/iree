// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-generalize-named-ops))" %s | FileCheck %s

module {
  func.func @transpose_matmul(%arg0: tensor<1x4096xf32>, %arg1: tensor<32000x4096xf32>) -> tensor<1x32000xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x32000xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
    %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<1x4096xf32>, tensor<32000x4096xf32>) outs(%1 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
    return %2 : tensor<1x32000xf32>
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:      func.func @transpose_matmul(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x4096xf32>, %[[ARG1:[a-zA-Z0-9_]+]]: tensor<32000x4096xf32>) -> tensor<1x32000xf32>
// CHECK:      %[[FILL:.+]] = linalg.fill
// CHECK-SAME: -> tensor<1x32000xf32>
// CHECK:      %[[GEN:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<1x4096xf32>, tensor<32000x4096xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<1x32000xf32>)
// CHECK:      ^bb0(%[[IN:.+]]: f32, %[[IN0:.+]]: f32, %[[OUT:.+]]: f32)
// CHECK:      %[[A0:.+]] = arith.mulf %[[IN]], %[[IN0]] : f32
// CHECK:      %[[A1:.+]] = arith.addf %[[OUT]], %[[A0]] : f32
// CHECK:      linalg.yield %[[A1]] : f32
// CHECK:      return %[[GEN]] : tensor<1x32000xf32>

// -----

module {
  func.func @matvec(%arg0: tensor<32000x4096xf32>, %arg1: tensor<4096xf32>) -> tensor<32000xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32000xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32000xf32>) -> tensor<32000xf32>
    %2 = linalg.matvec ins(%arg0, %arg1 : tensor<32000x4096xf32>, tensor<4096xf32>) outs(%1 : tensor<32000xf32>) -> tensor<32000xf32>
    return %2 : tensor<32000xf32>
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK:      func.func @matvec
// CHECK:        linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction"]
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN0:.+]]: f32, %[[OUT:.+]]: f32)
// CHECK:          %[[A0:.+]] = arith.mulf %[[IN]], %[[IN0]] : f32
// CHECK:          %[[A1:.+]] = arith.addf %[[OUT]], %[[A0]] : f32
// CHECK:          linalg.yield %[[A1]] : f32


// -----

module {
  func.func @vecmat(%arg0: tensor<4096xf32>, %arg1: tensor<4096x32000xf32>) -> tensor<32000xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32000xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32000xf32>) -> tensor<32000xf32>
    %2 = linalg.vecmat ins(%arg0, %arg1 : tensor<4096xf32>, tensor<4096x32000xf32>) outs(%1 : tensor<32000xf32>) -> tensor<32000xf32>
    return %2 : tensor<32000xf32>
  }
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK:      func.func @vecmat
// CHECK:        linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction"]
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN0:.+]]: f32, %[[OUT:.+]]: f32)
// CHECK:          %[[A0:.+]] = arith.mulf %[[IN]], %[[IN0]] : f32
// CHECK:          %[[A1:.+]] = arith.addf %[[OUT]], %[[A0]] : f32
// CHECK:          linalg.yield %[[A1]] : f32

// -----

module {
  func.func @transpose_batch_matmul(%arg0: tensor<32x1x128xf16>, %arg1: tensor<32x?x128xf16>, %dim: index) -> tensor<32x1x?xf16> {
    %f0 = arith.constant 0.0 : f16
    %empty = tensor.empty(%dim) : tensor<32x1x?xf16>
    %fill = linalg.fill ins(%f0 : f16) outs(%empty : tensor<32x1x?xf16>) -> tensor<32x1x?xf16>
    %2 = linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : tensor<32x1x128xf16>, tensor<32x?x128xf16>) outs(%fill : tensor<32x1x?xf16>) -> tensor<32x1x?xf16>
    return %2 : tensor<32x1x?xf16>
  }
}

//       CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//       CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//       CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @transpose_batch_matmul
//       CHECK:  linalg.generic
//  CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//       CHECK:  ^bb0(%[[A:.+]]: f16, %[[B:.+]]: f16, %[[OUT:.+]]: f16):
//       CHECK:    %[[MUL:.+]] = arith.mulf %[[A]], %[[B]] : f16
//       CHECK:    %[[ADD:.+]] = arith.addf %[[OUT]], %[[MUL]] : f16
//       CHECK:    linalg.yield %[[ADD]] : f16
