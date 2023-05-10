// RUN: iree-opt --iree-stablehlo-to-stablehlo-preprocessing %s | FileCheck %s

// CHECK-LABEL: @dot_general_2d
func.func public @dot_general_2d(%arg0: tensor<4x3xf32> {stablehlo.sharding = ""}, %arg1: tensor<4x3xf32> {stablehlo.sharding = ""}) -> tensor<3xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
         batching_dims = [1] x [1],
         contracting_dims = [0] x [0],
         precision= [HIGHEST, HIGHEST] : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<3xf32>

  // CHECK:      %[[LHS:.+]] = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
  // CHECK:      %[[RHS:.+]] = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
  // CHECK:      %[[DOT:.+]] = stablehlo.dot_general %[[LHS]], %[[RHS]]
  // CHECK-SAME:                 batching_dims = [0] x [0]
  // CHECK-SAME:                 contracting_dims = [1] x [1]
  // CHECK-SAME:                 precision = [HIGHEST, HIGHEST]
  // CHECK-NEXT: return %[[DOT]]
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: @dot_general_4d
func.func public @dot_general_4d(%arg0: tensor<1x2x3xf32> {stablehlo.sharding = ""}, %arg1: tensor<1x4x2x3xf32> {stablehlo.sharding = ""}) -> tensor<1x2x4xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
         batching_dims = [0, 1] x [0, 2],
         contracting_dims = [2] x [3],
         precision = [HIGHEST, HIGHEST] : (tensor<1x2x3xf32>, tensor<1x4x2x3xf32>) -> tensor<1x2x4xf32>

  // CHECK-DAG:  %[[RHS_T:.+]] = stablehlo.transpose %arg1, dims = [0, 2, 3, 1] : (tensor<1x4x2x3xf32>) -> tensor<1x2x3x4xf32>
  // CHECK-DAG:  %[[LHS_R:.+]] = stablehlo.reshape %arg0 : (tensor<1x2x3xf32>) -> tensor<2x1x3xf32>
  // CHECK-DAG:  %[[RHS_R:.+]] = stablehlo.reshape %[[RHS_T]] : (tensor<1x2x3x4xf32>) -> tensor<2x3x4xf32>
  // CHECK-NEXT: %[[DOT:.+]]   = stablehlo.dot_general %[[LHS_R]], %[[RHS_R]]
  // CHECK-SAME:                   batching_dims = [0] x [0]
  // CHECK-SAME:                   contracting_dims = [2] x [1]
  // CHECK-SAME:                   precision = [HIGHEST, HIGHEST]
  // CHECK-NEXT: %[[RES:.+]]   = stablehlo.reshape %[[DOT]] : (tensor<2x1x4xf32>) -> tensor<1x2x4xf32>
  // CHECK-NEXT: return %[[RES]]
  return %0 : tensor<1x2x4xf32>
}
