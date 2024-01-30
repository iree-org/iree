// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-util-hoist-into-globals{max-size-increase-threshold=1024})' %s | FileCheck %s
// Spot verification that policies for linalg ops is respected.

// CHECK-LABEL: @compute_hoisted
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @compute_hoisted {
  // CHECK: util.global private @[[HOISTED:.*]] : tensor<5x6xf32>
  // CHECK: func.func @main
  func.func @main() -> (tensor<5x6xf32>) {
    %cst_0 = arith.constant dense<1.270000e+02> : tensor<f32>

    // A non-leaf broadcast.
    %0 = tensor.empty() : tensor<5x6xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<f32>) outs(%0 : tensor<5x6xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<5x6xf32>

    // A leaf-compute.
    %2 = tensor.empty() : tensor<5x6xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%2 : tensor<5x6xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
      %42 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %42 : f32
    } -> tensor<5x6xf32>

    // CHECK: %[[RESULT:.*]] = util.global.load @[[HOISTED]] : tensor<5x6xf32>
    // CHECK: return %[[RESULT]]
    return %3 : tensor<5x6xf32>
  }
  // CHECK: util.initializer
}

// -----
// Verifies that projected permutations (broadcasts) will never be materialized
// as a leaf. Also verifies that empty operands, which can be considered
// const-expr, are not materialized as a leaf.
// CHECK-LABEL: @broadcast_treated_as_leaf
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @broadcast_treated_as_leaf {
  // CHECK-NOT: util.global
  // CHECK: func.func @main
  func.func @main() -> (tensor<5x6xf32>) {
    %cst_0 = arith.constant dense<1.270000e+02> : tensor<f32>
    // CHECK: tensor.empty()
    %0 = tensor.empty() : tensor<5x6xf32>
    // A broadcast.
    // CHECK: linalg.generic
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<f32>) outs(%0 : tensor<5x6xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<5x6xf32>
    // CHECK: return
    return %1 : tensor<5x6xf32>
  }
  // CHECK-NOT: util.initializer
}
