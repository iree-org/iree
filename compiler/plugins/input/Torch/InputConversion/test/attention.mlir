// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(torch-iree-tm-tensor-to-linalg-ext))" %s | FileCheck %s

func.func @attention(%arg0: tensor<5x2x3x4xf32>, %arg1: tensor<5x2x3x4xf32>, %arg2: tensor<5x2x3x4xf32>, %arg3: tensor<5x2x3x4xf32>) -> (tensor<5x2x3x4xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>) outs(%arg3: tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32>
  return %0 : tensor<5x2x3x4xf32>
}

// CHECK-LABEL:         func.func @attention(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<5x2x3x4xf32>, %[[ARG1:.*]]: tensor<5x2x3x4xf32>, %[[ARG2:.*]]: tensor<5x2x3x4xf32>,
// CHECK:         %arg3: tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32> {
// CHECK:         %[[DIM:.*]] = arith.constant 4.000000e+00 : f32
// CHECK:         %[[COL:.*]] = tensor.collapse_shape %[[ARG0]] {{.*}} : tensor<5x2x3x4xf32> into tensor<10x3x4xf32>
// CHECK:         %[[COL0:.*]] = tensor.collapse_shape %[[ARG1]] {{.*}} : tensor<5x2x3x4xf32> into tensor<10x3x4xf32>
// CHECK:         %[[COL1:.*]] = tensor.collapse_shape %[[ARG2]] {{.*}} : tensor<5x2x3x4xf32> into tensor<10x3x4xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<10x3x4xf32>
// CHECK:         %[[SCALE:.*]] = math.rsqrt %[[DIM]] : f32
// CHECK:         %[[ATTN:.*]] = iree_linalg_ext.attention ins(%[[COL]], %[[COL0]], %[[COL1]], %[[SCALE]] : tensor<10x3x4xf32>, tensor<10x3x4xf32>, tensor<10x3x4xf32>, f32) outs(%[[EMPTY]] : tensor<10x3x4xf32>) -> tensor<10x3x4xf32>
// CHECK:         %[[RET:.*]] = tensor.expand_shape %[[ATTN]] {{.*}} : tensor<10x3x4xf32> into tensor<5x2x3x4xf32>
// CHECK:         return %[[RET]] : tensor<5x2x3x4xf32>

// -----
func.func @attention(%arg0: tensor<5x2x8x4xf32>, %arg1: tensor<5x2x3x4xf32>, %arg2: tensor<5x2x3x4xf32>, %arg3: tensor<5x2x8x4xf32>) -> (tensor<5x2x8x4xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<5x2x8x4xf32>, tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>) outs(%arg3: tensor<5x2x8x4xf32>) -> tensor<5x2x8x4xf32>
  return %0 : tensor<5x2x8x4xf32>
}

// CHECK-LABEL:         func.func @attention(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<5x2x8x4xf32>, %[[ARG1:.*]]: tensor<5x2x3x4xf32>, %[[ARG2:.*]]: tensor<5x2x3x4xf32>,
// CHECK:         %arg3: tensor<5x2x8x4xf32>) -> tensor<5x2x8x4xf32> {
// CHECK:         %[[DIM:.*]] = arith.constant 4.000000e+00 : f32
// CHECK:         %[[COL:.*]] = tensor.collapse_shape %[[ARG0]] {{.*}} : tensor<5x2x8x4xf32> into tensor<10x8x4xf32>
// CHECK:         %[[COL0:.*]] = tensor.collapse_shape %[[ARG1]] {{.*}} : tensor<5x2x3x4xf32> into tensor<10x3x4xf32>
// CHECK:         %[[COL1:.*]] = tensor.collapse_shape %[[ARG2]] {{.*}} : tensor<5x2x3x4xf32> into tensor<10x3x4xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<10x8x4xf32>
// CHECK:         %[[SCALE:.*]] = math.rsqrt %[[DIM]] : f32
// CHECK:         %[[ATTN:.*]] = iree_linalg_ext.attention ins(%[[COL]], %[[COL0]], %[[COL1]], %[[SCALE]] : tensor<10x8x4xf32>, tensor<10x3x4xf32>, tensor<10x3x4xf32>, f32) outs(%[[EMPTY]] : tensor<10x8x4xf32>) -> tensor<10x8x4xf32>
// CHECK:         %[[RET:.*]] = tensor.expand_shape %[[ATTN]] {{.*}} : tensor<10x8x4xf32> into tensor<5x2x8x4xf32>
// CHECK:         return %[[RET]] : tensor<5x2x8x4xf32>

// -----
func.func @attention(%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x4xf32>, %arg2: tensor<1x3x4xf32>, %arg3: tensor<1x3x4xf32>) -> (tensor<1x3x4xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>) outs(%arg3: tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
  return %0 : tensor<1x3x4xf32>
}

// CHECK-LABEL:         func.func @attention(
// CHECK-SAME:         %[[ARG0:.*]]: tensor<1x3x4xf32>, %[[ARG1:.*]]: tensor<1x3x4xf32>, %[[ARG2:.*]]: tensor<1x3x4xf32>,
// CHECK:         %arg3: tensor<1x3x4xf32>) -> tensor<1x3x4xf32> {
// CHECK:         %[[DIM:.*]] = arith.constant 4.000000e+00 : f32
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<1x3x4xf32>
// CHECK:         %[[SCALE:.*]] = math.rsqrt %[[DIM]] : f32
// CHECK:         %[[ATTN:.*]] = iree_linalg_ext.attention ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[SCALE]] : tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>, f32) outs(%[[EMPTY]] : tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:         return %[[ATTN]] : tensor<1x3x4xf32>
