// RUN: iree-opt --split-input-file --iree-util-test-float-range-analysis --allow-unregistered-dialect %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @linalg_generic_traversal(%arg0 : tensor<5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>) {
  %cst_min = arith.constant dense<-1.270000e+02> : tensor<f32>
  %cst_max = arith.constant dense<1.270000e+02> : tensor<f32>
  %init = tensor.empty() : tensor<5x6xf32>

  %broadcast_min = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_min : tensor<f32>) outs(%init : tensor<5x6xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    linalg.yield %arg1 : f32
  } -> tensor<5x6xf32>
  %broadcast_max = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_max : tensor<f32>) outs(%init : tensor<5x6xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    linalg.yield %arg1 : f32
  } -> tensor<5x6xf32>

  %floor = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x6xf32>) outs(%init : tensor<5x6xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    %27 = math.floor %arg1 : f32
    linalg.yield %27 : f32
  } -> tensor<5x6xf32>
  %max = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%floor, %broadcast_min : tensor<5x6xf32>, tensor<5x6xf32>) outs(%init : tensor<5x6xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
    %27 = arith.maximumf %arg1, %arg2 : f32
    linalg.yield %27 : f32
  } -> tensor<5x6xf32>
  %min = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%max, %broadcast_max : tensor<5x6xf32>, tensor<5x6xf32>) outs(%init : tensor<5x6xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
    %27 = arith.minimumf %arg1, %arg2 : f32
    linalg.yield %27 : f32
  } -> tensor<5x6xf32>

  // CHECK: fp-range: [-127.000000, 127.000000, TRUNC]
  %result_range = "iree_unregistered.test_fprange"(%min) : (tensor<5x6xf32>) -> tensor<5x6xf32>
  // CHECK: fp-range: [-127.000000, inf, TRUNC]
  %max_range = "iree_unregistered.test_fprange"(%max) : (tensor<5x6xf32>) -> tensor<5x6xf32>
  // CHECK: fp-range: [-inf, inf, TRUNC]
  %floor_range = "iree_unregistered.test_fprange"(%floor) : (tensor<5x6xf32>) -> tensor<5x6xf32>
  return %result_range, %max_range, %floor_range : tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>
}
