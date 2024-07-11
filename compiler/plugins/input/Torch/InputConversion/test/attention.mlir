// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(torch-iree-tm-tensor-to-linalg-ext))" %s | FileCheck %s

func.func @attention(%arg0: tensor<5x2x3x4xf32>, %arg1: tensor<5x2x3x4xf32>, %arg2: tensor<5x2x3x4xf32>, %arg3: tensor<5x2x3x4xf32>) -> (tensor<5x2x3x4xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>) outs(%arg3: tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32>
  return %0 : tensor<5x2x3x4xf32>
}

// CHECK-DAG: #[[$MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
// CHECK-DAG: #[[$MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
// CHECK-DAG: #[[$MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>

// CHECK-LABEL:         func.func @attention(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<5x2x3x4xf32>, %[[ARG1:.*]]: tensor<5x2x3x4xf32>, %[[ARG2:.*]]: tensor<5x2x3x4xf32>,
// CHECK:         %arg3: tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32> {
// CHECK:         %[[SCALE:.*]] = arith.constant 5.000000e-01 : f32
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<5x2x3x4xf32>
// CHECK:         %[[ATTN:.*]] = iree_linalg_ext.attention {indexing_maps = [#[[$MAP_Q]], #[[$MAP_K]], #[[$MAP_V]], #[[$MAP_O]]]} ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[SCALE]] : tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>, f32) outs(%[[EMPTY]] : tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32>
// CHECK:         return %[[ATTN]] : tensor<5x2x3x4xf32>

// -----
func.func @attention(%arg0: tensor<5x2x8x4xf32>, %arg1: tensor<5x2x3x4xf32>, %arg2: tensor<5x2x3x4xf32>, %arg3: tensor<5x2x8x4xf32>) -> (tensor<5x2x8x4xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<5x2x8x4xf32>, tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>) outs(%arg3: tensor<5x2x8x4xf32>) -> tensor<5x2x8x4xf32>
  return %0 : tensor<5x2x8x4xf32>
}

// CHECK-DAG: #[[$MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
// CHECK-DAG: #[[$MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
// CHECK-DAG: #[[$MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>

// CHECK-LABEL:         func.func @attention(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<5x2x8x4xf32>, %[[ARG1:.*]]: tensor<5x2x3x4xf32>, %[[ARG2:.*]]: tensor<5x2x3x4xf32>,
// CHECK:         %arg3: tensor<5x2x8x4xf32>) -> tensor<5x2x8x4xf32> {
// CHECK:         %[[SCALE:.*]] = arith.constant 5.000000e-01 : f32
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<5x2x8x4xf32>
// CHECK:         %[[ATTN:.*]] = iree_linalg_ext.attention {indexing_maps = [#[[$MAP_Q]], #[[$MAP_K]], #[[$MAP_V]], #[[$MAP_O]]]} ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[SCALE]] : tensor<5x2x8x4xf32>, tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>, f32) outs(%[[EMPTY]] : tensor<5x2x8x4xf32>) -> tensor<5x2x8x4xf32>
// CHECK:         return %[[ATTN]] : tensor<5x2x8x4xf32>

// -----
func.func @attention(%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x4xf32>, %arg2: tensor<1x3x4xf32>, %arg3: tensor<1x3x4xf32>) -> (tensor<1x3x4xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>) outs(%arg3: tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
  return %0 : tensor<1x3x4xf32>
}

// CHECK-DAG: #[[$MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG: #[[$MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG: #[[$MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

// CHECK-LABEL:         func.func @attention(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<1x3x4xf32>, %[[ARG1:.*]]: tensor<1x3x4xf32>, %[[ARG2:.*]]: tensor<1x3x4xf32>,
// CHECK:         %arg3: tensor<1x3x4xf32>) -> tensor<1x3x4xf32> {
// CHECK:         %[[SCALE:.*]] = arith.constant 5.000000e-01 : f32
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<1x3x4xf32>
// CHECK:         %[[ATTN:.*]] = iree_linalg_ext.attention {indexing_maps = [#[[$MAP_Q]], #[[$MAP_K]], #[[$MAP_V]], #[[$MAP_O]]]} ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[SCALE]] : tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>, f32) outs(%[[EMPTY]] : tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:         return %[[ATTN]] : tensor<1x3x4xf32>
