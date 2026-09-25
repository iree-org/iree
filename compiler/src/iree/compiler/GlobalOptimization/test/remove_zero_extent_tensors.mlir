// RUN: iree-opt --split-input-file --allow-unregistered-dialect \
// RUN:          --pass-pipeline="builtin.module(util.func(iree-global-opt-remove-zero-extent-tensors))" \
// RUN:          %s | FileCheck %s

util.func public @zero_sized_operands(%arg0 : tensor<?x0xf32>, %arg1 : index) -> tensor<?x?xf32> {
  %0 = tensor.empty(%arg1): tensor<0x?xf32>
  %1 = "some_op"(%arg0, %0) : (tensor<?x0xf32>, tensor<0x?xf32>) -> tensor<?x?xf32>
  util.return %1 : tensor<?x?xf32>
}
//      CHECK: util.func public @zero_sized_operands
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x0xf32>
// CHECK-SAME:     %[[ARG1:.+]]: index
//      CHECK:   %[[EMPTY0:.+]] = tensor.empty(%[[ARG1]])
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG0]]
//      CHECK:   %[[EMPTY1:.+]] = tensor.empty(%[[DIM]])
//      CHECK:   %[[RESULT:.+]] = "some_op"(%[[EMPTY1]], %[[EMPTY0]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @zero_sized_tensor_insert(%arg0 : tensor<?x?xf32>, %arg1 : tensor<0x?xf32>,
    %arg2 : index) -> tensor<?x?xf32> {
  %1 = tensor.insert_slice %arg1 into %arg0[0, 0] [0, %arg2] [1, 1] : tensor<0x?xf32> into tensor<?x?xf32>
  util.return %1 : tensor<?x?xf32>
}
// CHECK: util.func public @zero_sized_tensor_insert(%[[ARG0:.+]]: tensor<?x?xf32>
// CHECK:   util.return %[[ARG0]]

// -----

// A reduction over a zero-sized dimension (2x0x4 -> 2x1x4) has no input
// elements combined, so the result is exactly the init operand. Here the init
// is a *non-identity* accumulator (filled with 5.0), so folding must forward
// the init and preserve that value -- it must NOT synthesize a fresh
// identity-filled tensor.
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
util.func public @zero_extent_reduction_non_identity_init(
    %arg0 : tensor<2x0x4xf32>) -> tensor<2x1x4xf32> {
  %cst = arith.constant 5.000000e+00 : f32
  %empty = tensor.empty() : tensor<2x1x4xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<2x1x4xf32>) -> tensor<2x1x4xf32>
  %0 = linalg.generic {
      indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%arg0 : tensor<2x0x4xf32>)
      outs(%init : tensor<2x1x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<2x1x4xf32>
  util.return %0 : tensor<2x1x4xf32>
}
//      CHECK: util.func public @zero_extent_reduction_non_identity_init
//  CHECK-DAG:   %[[CST:.+]] = arith.constant 5.000000e+00 : f32
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x1x4xf32>
//      CHECK:   %[[INIT:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[EMPTY]]
//  CHECK-NOT:   linalg.generic
//      CHECK:   util.return %[[INIT]]

// -----

// Same fold with dynamic output dimensions: the reduced input dim is zero
// (?x0x4 -> ?x1x4) and the init carries the dynamic size. Folding forwards the
// init directly; no linalg.generic and no new fill are introduced.
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
util.func public @zero_extent_reduction_dynamic(
    %arg0 : tensor<?x0x4xf32>, %init : tensor<?x1x4xf32>) -> tensor<?x1x4xf32> {
  %0 = linalg.generic {
      indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%arg0 : tensor<?x0x4xf32>)
      outs(%init : tensor<?x1x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<?x1x4xf32>
  util.return %0 : tensor<?x1x4xf32>
}
//      CHECK: util.func public @zero_extent_reduction_dynamic
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x0x4xf32>
// CHECK-SAME:     %[[INIT:.+]]: tensor<?x1x4xf32>
//  CHECK-NOT:   linalg.generic
//  CHECK-NOT:   linalg.fill
//      CHECK:   util.return %[[INIT]]
