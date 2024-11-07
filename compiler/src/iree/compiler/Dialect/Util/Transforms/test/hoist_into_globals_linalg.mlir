// RUN: iree-opt --split-input-file --iree-util-hoist-into-globals %s | FileCheck %s

// Spot verification that policies for linalg ops is respected.

// CHECK-LABEL: @compute_hoisted
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @compute_hoisted {
  // CHECK: util.global private @[[HOISTED:.*]] : tensor<5x6xf32>
  // CHECK: util.initializer
  // CHECK: util.func public @main
  util.func public @main() -> (tensor<5x6xf32>) {
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

    // CHECK: %[[RESULT:.*]] = util.global.load immutable @[[HOISTED]] : tensor<5x6xf32>
    // CHECK: util.return %[[RESULT]]
    util.return %3 : tensor<5x6xf32>
  }
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
  // CHECK-NOT: util.initializer
  // CHECK: util.func public @main
  util.func public @main() -> (tensor<5x6xf32>) {
    %cst_0 = arith.constant dense<1.270000e+02> : tensor<f32>
    // CHECK: tensor.empty()
    %0 = tensor.empty() : tensor<5x6xf32>
    // A broadcast.
    // CHECK: linalg.generic
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<f32>) outs(%0 : tensor<5x6xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<5x6xf32>
    // CHECK: util.return
    util.return %1 : tensor<5x6xf32>
  }
}

// -----

// Checks that a ConstantOp which only has a consumer within a nest gets hoisted.
module @nested_consumer {
  // CHECK: util.global private @[[HOISTED:.*]] : tensor<2xf32>
  // CHECK: util.initializer
  // CHECK: util.func public @main
  util.func @main(%arg0: tensor<2xindex>) -> tensor<1x2xf32> {
    %const_t = arith.constant dense<[0.0, 1.0]> : tensor<2xf32>
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<2xf32>
    %add_one = linalg.generic { //constOp
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%const_t : tensor<2xf32>) outs(%empty: tensor<2xf32>) {
      ^bb0(%in : f32, %out : f32):
        %added = arith.addf %in, %one : f32
        linalg.yield %added : f32
    } -> tensor<2xf32>
    // CHECK: %[[CONST:.*]] = util.global.load immutable @[[HOISTED]] : tensor<2xf32>
    %empty2 = tensor.empty() : tensor<2xf32>
    %loaded = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<2xindex>) outs(%empty2 : tensor<2xf32>) {
      ^bb0(%in: index, %out: f32):
        //CHECK %[[.*]] = tensor.extract %[[CONST]][%in] : tensor<2xf32>
        %extracted = tensor.extract %add_one[%in] : tensor<2xf32> // consumer
        linalg.yield %extracted : f32
    } -> tensor<2xf32>
    %reshaped = tensor.expand_shape %loaded [[0, 1]] output_shape[1, 2]: tensor<2xf32> into tensor<1x2xf32>
    util.return %reshaped : tensor<1x2xf32>
  }
}
