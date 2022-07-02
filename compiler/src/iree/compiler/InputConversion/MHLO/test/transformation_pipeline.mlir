// RUN: iree-opt --split-input-file --iree-mhlo-input-transformation-pipeline %s | FileCheck %s

// CHECK-LABEL: @empty
func.func @empty() {
  // CHECK-NEXT: return
  return
}

// -----

func.func @mhloElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  %1 = mhlo.subtract %0, %arg0 : tensor<4xf32>
  %2 = mhlo.multiply %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK:      #map = affine_map<(d0) -> (d0)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func @mhloElementwiseOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %0 = linalg.init_tensor [4] : tensor<4xf32>
// CHECK-NEXT:     %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) {
// CHECK-NEXT:     ^bb0(%arg1: f32, %arg2: f32):
// CHECK-NEXT:       %6 = arith.addf %arg1, %arg1 : f32
// CHECK-NEXT:       linalg.yield %6 : f32
// CHECK-NEXT:     } -> tensor<4xf32>
// CHECK-NEXT:     %2 = linalg.init_tensor [4] : tensor<4xf32>
// CHECK-NEXT:     %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%1, %arg0 : tensor<4xf32>, tensor<4xf32>) outs(%2 : tensor<4xf32>) {
// CHECK-NEXT:     ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
// CHECK-NEXT:       %6 = arith.subf %arg1, %arg2 : f32
// CHECK-NEXT:       linalg.yield %6 : f32
// CHECK-NEXT:     } -> tensor<4xf32>
// CHECK-NEXT:     %4 = linalg.init_tensor [4] : tensor<4xf32>
// CHECK-NEXT:     %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %arg0 : tensor<4xf32>, tensor<4xf32>) outs(%4 : tensor<4xf32>) {
// CHECK-NEXT:     ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
// CHECK-NEXT:       %6 = arith.mulf %arg1, %arg2 : f32
// CHECK-NEXT:       linalg.yield %6 : f32
// CHECK-NEXT:     } -> tensor<4xf32>
// CHECK-NEXT:     return %5 : tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

func.func @interleavedDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mhlo.add %arg0, %arg0 : tensor<4x4xf32>
  %1 = "mhlo.dot"(%0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mhlo.multiply %1, %arg0 : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// CHECK:      #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func @interleavedDot(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = linalg.init_tensor [4, 4] : tensor<4x4xf32>
// CHECK-NEXT:     %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) {
// CHECK-NEXT:     ^bb0(%arg1: f32, %arg2: f32):
// CHECK-NEXT:       %7 = arith.addf %arg1, %arg1 : f32
// CHECK-NEXT:       linalg.yield %7 : f32
// CHECK-NEXT:     } -> tensor<4x4xf32>
// CHECK-NEXT:     %2 = linalg.init_tensor [4, 4] : tensor<4x4xf32>
// CHECK-NEXT:     %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     %4 = linalg.matmul ins(%1, %arg0 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%3 : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     %5 = linalg.init_tensor [4, 4] : tensor<4x4xf32>
// CHECK-NEXT:     %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %arg0 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%5 : tensor<4x4xf32>) {
// CHECK-NEXT:     ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
// CHECK-NEXT:       %7 = arith.mulf %arg1, %arg2 : f32
// CHECK-NEXT:       linalg.yield %7 : f32
// CHECK-NEXT:     } -> tensor<4x4xf32>
// CHECK-NEXT:     return %6 : tensor<4x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }


// -----

func.func @reduction(%arg0 : tensor<4x8xf32>) -> tensor<4xf32> {
  %0 = arith.constant dense<0.0> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1 : tensor<f32>, %arg2 : tensor<f32>):
    %2 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK:      #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT: #map1 = affine_map<(d0, d1) -> (d0)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func @reduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = linalg.init_tensor [4] : tensor<4xf32>
// CHECK-NEXT:     %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<4x8xf32>) outs(%1 : tensor<4xf32>) {
// CHECK-NEXT:     ^bb0(%arg1: f32, %arg2: f32):
// CHECK-NEXT:       %3 = arith.addf %arg1, %arg2 : f32
// CHECK-NEXT:       linalg.yield %3 : f32
// CHECK-NEXT:     } -> tensor<4xf32>
// CHECK-NEXT:     return %2 : tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
