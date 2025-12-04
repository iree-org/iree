// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-linalg-generic))" %s | FileCheck %s

// CHECK-LABEL: @parallel_with_broadcast_dynamic
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<64xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x64xf32>
func.func @parallel_with_broadcast_dynamic(
    %arg0 : tensor<64xf32>, %arg1 : tensor<?x64xf32>) -> tensor<?x64xf32> {
  // Verify that the decomposition pattern splits the generic into two operations:
  // First generic computes the add, second generic computes the multiply using
  // the result of the first.
  //       CHECK: %[[EMPTY:.+]] = tensor.empty
  //       CHECK: %[[ADD:.+]] = linalg.generic
  //  CHECK-SAME:   ins(%[[ARG0]] :
  //  CHECK-SAME:   outs(%[[EMPTY]] :
  //       CHECK:   %[[ADDF:.+]] = arith.addf
  //       CHECK:   linalg.yield %[[ADDF]]
  //       CHECK: %[[MUL:.+]] = linalg.generic
  //  CHECK-SAME:   ins(%[[ARG0]], %[[ADD]] :
  //  CHECK-SAME:   outs(%[[ARG1]] :
  //       CHECK:   %[[MULF:.+]] = arith.mulf
  //       CHECK:   linalg.yield %[[MULF]]
  //       CHECK: return %[[MUL]]
  %12 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<64xf32>) outs(%arg1 : tensor<?x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %13 = arith.addf %arg2, %arg2 : f32
    %14 = arith.mulf %13, %arg2 : f32
    linalg.yield %14 : f32
  } -> tensor<?x64xf32>
  func.return %12 : tensor<?x64xf32>
}
