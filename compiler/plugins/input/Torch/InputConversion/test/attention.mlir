// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(torch-iree-tm-tensor-to-linalg-ext))" %s | FileCheck %s

func.func @attention(%arg0: tensor<5x2x3x4xf32>, %arg1: tensor<5x2x3x4xf32>, %arg2: tensor<5x2x3x4xf32>, %arg3: tensor<5x2x3x4xf32>) -> (tensor<5x2x3x4xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>) outs(%arg3: tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32>
  return %0 : tensor<5x2x3x4xf32>
}

// CHECK-DAG: #[[$MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
// CHECK-DAG: #[[$MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d3)>
// CHECK-DAG: #[[$MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
// CHECK-DAG: #[[$MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP_MAX:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @attention(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<5x2x3x4xf32>, %[[ARG1:.*]]: tensor<5x2x3x4xf32>, %[[ARG2:.*]]: tensor<5x2x3x4xf32>,
// CHECK-SAME:    %[[ARG3:.*]]: tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32> {
// CHECK-DAG:   %[[SCALE:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:   %[[ACC_FILL:.*]] = linalg.fill {{.*}} -> tensor<5x2x3x4xf32>
// CHECK-DAG:   %[[MAX_FILL:.*]] = linalg.fill {{.*}} -> tensor<5x2x3xf32>
// CHECK-DAG:   %[[SUM_FILL:.*]] = linalg.fill {{.*}} -> tensor<5x2x3xf32>
// CHECK:       %[[ATTN:.*]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:    indexing_maps = [#[[$MAP_Q]], #[[$MAP_K]], #[[$MAP_V]], #[[$MAP_S]], #[[$MAP_O]], #[[$MAP_MAX]], #[[$MAP_MAX]]]
// CHECK-SAME:    ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[SCALE]] :
// CHECK-SAME:    outs(%[[ACC_FILL]], %[[MAX_FILL]], %[[SUM_FILL]] :
// CHECK:       ^{{.*}}(%[[SCORE:.*]]: f32):
// CHECK:         iree_linalg_ext.yield %[[SCORE]]
// CHECK:       %[[NORM:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ATTN]]#2, %[[ATTN]]#0 :
// CHECK:         arith.divf
// CHECK:         arith.mulf
// CHECK:       return %[[NORM]] : tensor<5x2x3x4xf32>

// -----
func.func @attention(%arg0: tensor<5x2x8x4xf32>, %arg1: tensor<5x2x3x4xf32>, %arg2: tensor<5x2x3x4xf32>, %arg3: tensor<5x2x8x4xf32>) -> (tensor<5x2x8x4xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<5x2x8x4xf32>, tensor<5x2x3x4xf32>, tensor<5x2x3x4xf32>) outs(%arg3: tensor<5x2x8x4xf32>) -> tensor<5x2x8x4xf32>
  return %0 : tensor<5x2x8x4xf32>
}

// CHECK-LABEL: func.func @attention(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<5x2x8x4xf32>, %[[ARG1:.*]]: tensor<5x2x3x4xf32>, %[[ARG2:.*]]: tensor<5x2x3x4xf32>,
// CHECK-SAME:    %[[ARG3:.*]]: tensor<5x2x8x4xf32>) -> tensor<5x2x8x4xf32> {
// CHECK:       %[[ATTN:.*]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:    ins(%[[ARG0]], %[[ARG1]], %[[ARG2]],
// CHECK-SAME:    outs({{.*}}, {{.*}}, {{.*}} :
// CHECK:       %[[NORM:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ATTN]]#2, %[[ATTN]]#0 :
// CHECK:       return %[[NORM]] : tensor<5x2x8x4xf32>

// -----
func.func @attention(%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x4xf32>, %arg2: tensor<1x3x4xf32>, %arg3: tensor<1x3x4xf32>) -> (tensor<1x3x4xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>) outs(%arg3: tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
  return %0 : tensor<1x3x4xf32>
}

// CHECK-LABEL: func.func @attention(
// CHECK:       %[[ATTN:.*]]:3 = iree_linalg_ext.online_attention
// CHECK:       %[[NORM:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ATTN]]#2, %[[ATTN]]#0 :
// CHECK:       return %[[NORM]] : tensor<1x3x4xf32>

// -----
func.func @attention_dyn(%arg0: tensor<?x?x4xf32>, %arg1: tensor<?x?x4xf32>, %arg2: tensor<?x?x4xf32>, %arg3: tensor<?x?x4xf32>) -> (tensor<?x?x4xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<?x?x4xf32>, tensor<?x?x4xf32>, tensor<?x?x4xf32>) outs(%arg3: tensor<?x?x4xf32>) -> tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// CHECK-LABEL: func.func @attention_dyn(
// CHECK:       %[[ATTN:.*]]:3 = iree_linalg_ext.online_attention
// CHECK:       %[[NORM:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ATTN]]#2, %[[ATTN]]#0 :
// CHECK:       return %[[NORM]] : tensor<?x?x4xf32>

// -----
func.func @attention_dyn_head_dim(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg3: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func.func @attention_dyn_head_dim(
// CHECK:       %[[SCALE:.*]] = math.rsqrt
// CHECK:       %[[ATTN:.*]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:    ins({{.*}}, {{.*}}, {{.*}}, %[[SCALE]] :
// CHECK:       %[[NORM:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ATTN]]#2, %[[ATTN]]#0 :
// CHECK:       return %[[NORM]] : tensor<?x?x?xf32>
