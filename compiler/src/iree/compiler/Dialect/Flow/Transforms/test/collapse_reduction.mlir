// RUN: iree-opt --split-input-file -iree-flow-collapse-dims %s | FileCheck %s

func.func @multi_reduce_dim(%arg0: tensor<2x32x10x4096xf32>) -> tensor<2x32x1x1xf32> {
  %cst = arith.constant -0.000000e+00 : f32
  %1 = tensor.empty() : tensor<2x32xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg0 : tensor<2x32x10x4096xf32>) outs(%2 : tensor<2x32xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %6 = arith.addf %arg1, %arg2 : f32
    linalg.yield %6 : f32
  } -> tensor<2x32xf32>
  %4 = tensor.expand_shape %3 [[0], [1, 2, 3]] : tensor<2x32xf32> into tensor<2x32x1x1xf32>
  return %4 : tensor<2x32x1x1xf32>
}

// Check that we collapse dimensions.
// CHECK: @multi_reduce_dim
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]

// -----

// Collapsing is not supported when an input is broadcasted; we can't collapse
// the input from tensor<4xf32> to tensor<32xf32> for example.

func.func @input_broadcast(%arg0: tensor<4x8xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
  %empty = tensor.empty() : tensor<f32>
  %reduce = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> ()>], iterator_types = ["reduction", "reduction"]} ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4xf32>) outs(%empty : tensor<f32>) {
  ^bb0(%arg2: f32, %arg3: f32, %out: f32):
    %div = arith.divf %arg2, %arg3 : f32
    %add = arith.addf %out, %div : f32
    linalg.yield %add : f32
  } -> tensor<f32>
  return %reduce : tensor<f32>
}

// CHECK: @input_broadcast
// CHECK-NOT: tensor.collapse_shape

// -----

// Collapsing should not happen to ops in flow.dispatch.region or flow.dispatch.workgroups

func.func @multi_reduce_dim_dispatch(%arg0: tensor<2x32x10x4096xf32>) -> tensor<2x32x1x1xf32> {
  %cst = arith.constant -0.000000e+00 : f32
  %1 = tensor.empty() : tensor<2x32xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %3 = flow.dispatch.region -> (tensor<2x32xf32>) {
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg0 : tensor<2x32x10x4096xf32>) outs(%2 : tensor<2x32xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.addf %arg1, %arg2 : f32
      linalg.yield %7 : f32
    } -> tensor<2x32xf32>
    flow.return %6 : tensor<2x32xf32>
  }
  %4 = tensor.expand_shape %3 [[0], [1, 2, 3]] : tensor<2x32xf32> into tensor<2x32x1x1xf32>
  return %4 : tensor<2x32x1x1xf32>
}

// CHECK: @multi_reduce_dim_dispatch
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction", "reduction"]
