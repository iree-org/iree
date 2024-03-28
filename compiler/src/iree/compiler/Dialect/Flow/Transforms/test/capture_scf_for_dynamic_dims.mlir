// RUN: iree-opt --iree-flow-capture-dynamic-dims %s | FileCheck %s

// CHECK-LABEL: @captureDims
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0_DIM0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0_DIM1:[[:alnum:]]+]]
util.func public @captureDims(%arg0: tensor<?x?xf32>, %arg0_dim0: index, %arg0_dim1: index, %ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[TENSOR:.*]] = flow.tensor.tie_shape %[[ARG0]]
  %tensor = flow.tensor.tie_shape %arg0 : tensor<?x?xf32>{%arg0_dim0, %arg0_dim1}
  // CHECK: %[[RET:.*]]:3 = scf.for
  // CHECK-SAME: iter_args(
  // CHECK-SAME: %[[ITER:.*]] = %[[TENSOR]]
  // CHECK-SAME: %[[ITER_DIM0:.*]] = %[[ARG0_DIM0]]
  // CHECK-SAME: %[[ITER_DIM1:.*]] = %[[ARG0_DIM1]]
  %ret = scf.for %ind = %c0 to %ub step %c1 iter_args(%arg1 = %tensor) -> tensor<?x?xf32> {
    // CHECK: %[[ITER_TIED:.*]] = flow.tensor.tie_shape %[[ITER]] : tensor<?x?xf32>{%[[ITER_DIM0]], %[[ITER_DIM1]]}
    %0 = flow.tensor.empty : tensor<?x?xf32>{%arg0_dim0, %arg0_dim1}
    // CHECK: linalg.generic
    // CHECK: ins(%[[ITER_TIED]] : tensor<?x?xf32>)
    %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %3 = arith.addf %b0, %b0 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    // CHECK: yield %{{.*}}, %[[ARG0_DIM0]], %[[ARG0_DIM1]]
    scf.yield %2 : tensor<?x?xf32>
  }
  // CHECK: flow.tensor.tie_shape %[[RET]]#0 : tensor<?x?xf32>{%[[RET]]#1, %[[RET]]#2}
  util.return
}
