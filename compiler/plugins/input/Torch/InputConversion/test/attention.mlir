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
func.func @attention_dyn(%arg0: tensor<?x?x?x4xf32>, %arg1: tensor<?x?x?x4xf32>, %arg2: tensor<?x?x?x?xf32>, %arg3: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<?x?x?x4xf32>, tensor<?x?x?x4xf32>, tensor<?x?x?x?xf32>) outs(%arg3 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func.func @attention_dyn(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<?x?x?x4xf32>, %[[ARG1:.*]]: tensor<?x?x?x4xf32>, %[[ARG2:.*]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:    %[[ARG3:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:       %[[B0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x4xf32>
// CHECK:       %[[B1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x4xf32>
// CHECK:       %[[M:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x4xf32>
// CHECK:       %[[N:.*]] = tensor.dim %[[ARG2]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:       %[[OUT:.*]] = tensor.empty(%[[B0]], %[[B1]], %[[M]], %[[N]]) : tensor<?x?x?x?xf32>
// CHECK:       %[[ACC_EMPTY:.*]] = tensor.empty({{.*}}) : tensor<?x?x?x?xf32>
// CHECK:       %[[ROW_EMPTY:.*]] = tensor.empty({{.*}}) : tensor<?x?x?xf32>
// CHECK-DAG:   %[[ACC_FILL:.*]] = linalg.fill {{.*}} outs(%[[ACC_EMPTY]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK-DAG:   %[[MAX_FILL:.*]] = linalg.fill {{.*}} outs(%[[ROW_EMPTY]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-DAG:   %[[SUM_FILL:.*]] = linalg.fill {{.*}} outs(%[[ROW_EMPTY]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:       %[[ATTN:.*]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:    ins(%[[ARG0]], %[[ARG1]], %[[ARG2]],
// CHECK-SAME:    outs(%[[ACC_FILL]], %[[MAX_FILL]], %[[SUM_FILL]] :
// CHECK-SAME:    tensor<?x?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK:       %[[NORM:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ATTN]]#2, %[[ATTN]]#0 :
// CHECK-SAME:    outs(%[[OUT]] : tensor<?x?x?x?xf32>)
// CHECK:       return %[[NORM]] : tensor<?x?x?x?xf32>

// -----
func.func @attention_dyn_head_dim(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>, %arg3: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg3: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func.func @attention_dyn_head_dim(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>, %[[ARG2:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:    %[[ARG3:.*]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[B:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:       %[[M:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:       %[[N:.*]] = tensor.dim %[[ARG2]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:       %[[OUT:.*]] = tensor.empty(%[[B]], %[[M]], %[[N]]) : tensor<?x?x?xf32>
// CHECK:       %[[HEAD_DIM:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:       %[[HEAD_DIM_I64:.*]] = arith.index_cast %[[HEAD_DIM]] : index to i64
// CHECK:       %[[HEAD_DIM_F32:.*]] = arith.sitofp %[[HEAD_DIM_I64]] : i64 to f32
// CHECK:       %[[SCALE:.*]] = math.rsqrt %[[HEAD_DIM_F32]] : f32
// CHECK:       %[[ATTN:.*]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:    ins({{.*}}, {{.*}}, {{.*}}, %[[SCALE]] :
// CHECK:       %[[NORM:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ATTN]]#2, %[[ATTN]]#0 :
// CHECK-SAME:    outs(%[[OUT]] : tensor<?x?x?xf32>)
// CHECK:       return %[[NORM]] : tensor<?x?x?xf32>

// -----
func.func @attention_with_mask(%arg0: tensor<5x2x3x4xf32>, %arg1: tensor<5x2x8x4xf32>, %arg2: tensor<5x2x8x4xf32>, %arg3: tensor<5x2x3x8xf32>, %arg4: tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32> {
  %0 = tm_tensor.attention ins(%arg0, %arg1, %arg2, %arg3 : tensor<5x2x3x4xf32>, tensor<5x2x8x4xf32>, tensor<5x2x8x4xf32>, tensor<5x2x3x8xf32>) outs(%arg4 : tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32>
  return %0 : tensor<5x2x3x4xf32>
}

// CHECK-DAG: #[[$MAP_Q_MASK:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
// CHECK-DAG: #[[$MAP_K_MASK:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
// CHECK-DAG: #[[$MAP_V_MASK:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d3)>
// CHECK-DAG: #[[$MAP_S_MASK:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
// CHECK-DAG: #[[$MAP_MASK:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
// CHECK-DAG: #[[$MAP_O_MASK:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP_ROW_MASK:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @attention_with_mask(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<5x2x3x4xf32>, %[[ARG1:.*]]: tensor<5x2x8x4xf32>, %[[ARG2:.*]]: tensor<5x2x8x4xf32>,
// CHECK-SAME:    %[[ARG3:.*]]: tensor<5x2x3x8xf32>, %[[ARG4:.*]]: tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32> {
// CHECK-DAG:   %[[SCALE:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:   %[[ACC_FILL:.*]] = linalg.fill {{.*}} -> tensor<5x2x3x4xf32>
// CHECK-DAG:   %[[MAX_FILL:.*]] = linalg.fill {{.*}} -> tensor<5x2x3xf32>
// CHECK-DAG:   %[[SUM_FILL:.*]] = linalg.fill {{.*}} -> tensor<5x2x3xf32>
// CHECK:       %[[ATTN:.*]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:    indexing_maps = [#[[$MAP_Q_MASK]], #[[$MAP_K_MASK]], #[[$MAP_V_MASK]], #[[$MAP_S_MASK]], #[[$MAP_MASK]], #[[$MAP_O_MASK]], #[[$MAP_ROW_MASK]], #[[$MAP_ROW_MASK]]]
// CHECK-SAME:    ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[SCALE]], %[[ARG3]] :
// CHECK-SAME:    tensor<5x2x3x4xf32>, tensor<5x2x8x4xf32>, tensor<5x2x8x4xf32>, f32, tensor<5x2x3x8xf32>)
// CHECK-SAME:    outs(%[[ACC_FILL]], %[[MAX_FILL]], %[[SUM_FILL]] :
// CHECK-SAME:    tensor<5x2x3x4xf32>, tensor<5x2x3xf32>, tensor<5x2x3xf32>)
// CHECK:       %[[NORM:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ATTN]]#2, %[[ATTN]]#0 :
// CHECK:       return %[[NORM]] : tensor<5x2x3x4xf32>
