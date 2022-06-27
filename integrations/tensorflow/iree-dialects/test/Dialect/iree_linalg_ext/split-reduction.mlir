// RUN: iree-dialects-opt --iree-linalg-ext-topk-split-reduction='split-ratio=3' %s | FileCheck %s --check-prefix SINGLE
// RUN: iree-dialects-opt --iree-linalg-ext-topk-split-reduction='split-ratio=4' %s | FileCheck %s --check-prefix MULTIPLE

func.func @topk_split_reduction_1d(%input_values: tensor<30xf32>, %out_values: tensor<3xf32>, %out_indices: tensor<3xi32>) -> (tensor<3xf32>, tensor<3xi32>) {
  %0:2 = iree_linalg_ext.topk
        dimension(0)
        ins(%input_values: tensor<30xf32>)
        outs(%out_values, %out_indices : tensor<3xf32>, tensor<3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<3xf32>, tensor<3xi32>
  return %0#0, %0#1 : tensor<3xf32>, tensor<3xi32>
}

// SINGLE-DAG:     #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// SINGLE-LABEL:   func.func @topk_split_reduction_1d(
// SINGLE-SAME:                                       %[[ARG0:.*]]: tensor<30xf32>,
// SINGLE-SAME:                                       %[[ARG1:.*]]: tensor<3xf32>,
// SINGLE-SAME:                                       %[[ARG2:.*]]: tensor<3xi32>) -> (tensor<3xf32>, tensor<3xi32>) {
// SINGLE-DAG:       %[[CNEG:.*]] = arith.constant 0xFF800000 : f32
// SINGLE-DAG:       %[[CPOS:.*]] = arith.constant 2147483647 : i32
// SINGLE-DAG:       %[[C10:.*]] = arith.constant 10 : i32
// SINGLE:           %[[D0:.*]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1]] : tensor<30xf32> into tensor<3x10xf32>
// SINGLE:           %[[D1:.*]] = linalg.init_tensor [3, 3] : tensor<3x3xf32>
// SINGLE:           %[[D2:.*]] = linalg.init_tensor [3, 3] : tensor<3x3xi32>
// SINGLE:           %[[D3:.*]] = linalg.fill ins(%[[CNEG]] : f32) outs(%[[D1]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// SINGLE:           %[[D4:.*]] = linalg.fill ins(%[[CPOS]] : i32) outs(%[[D2]] : tensor<3x3xi32>) -> tensor<3x3xi32>
// SINGLE:           %[[D5:.*]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "SPLIT_REDUCTION"} dimension(1) ins(%[[D0]] : tensor<3x10xf32>) outs(%[[D3]], %[[D4]] : tensor<3x3xf32>, tensor<3x3xi32>) {
// SINGLE:           ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
// SINGLE:             %[[D10:.*]] = arith.cmpf ogt, %[[ARG3]], %[[ARG4]] : f32
// SINGLE:             iree_linalg_ext.yield %[[D10]] : i1
// SINGLE:           } -> tensor<3x3xf32>, tensor<3x3xi32>
// SINGLE:           %[[ARG3:.*]] = linalg.generic {indexing_maps = [#[[MAP0]]], iterator_types = ["parallel", "parallel"]} outs(%[[D5:.*]]#1 : tensor<3x3xi32>) {
// SINGLE:           ^bb0(%[[ARG3:.*]]: i32):
// SINGLE:             %[[D10:.*]] = linalg.index 0 : index
// SINGLE:             %[[D11:.*]] = arith.index_cast %[[D10]] : index to i32
// SINGLE:             %[[D12:.*]] = arith.muli %[[D11]], %[[C10]] : i32
// SINGLE:             %[[D13:.*]] = arith.addi %[[D12]], %[[ARG3]] : i32
// SINGLE:             linalg.yield %[[D13]] : i32
// SINGLE:           } -> tensor<3x3xi32>
// SINGLE:           %[[D7:.*]] = tensor.collapse_shape %[[D5:.*]]#0 {{\[\[}}0, 1]] : tensor<3x3xf32> into tensor<9xf32>
// SINGLE:           %[[D8:.*]] = tensor.collapse_shape %[[D6:.*]] {{\[\[}}0, 1]] : tensor<3x3xi32> into tensor<9xi32>
// SINGLE:           %[[D9:.*]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "SPLIT_REDUCTION"} dimension(0) ins(%[[D7]], %[[D8]] : tensor<9xf32>, tensor<9xi32>) outs(%[[ARG1]], %[[ARG2]] : tensor<3xf32>, tensor<3xi32>) {
// SINGLE:           ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
// SINGLE:             %[[D10:.*]] = arith.cmpf ogt, %[[ARG3]], %[[ARG4]] : f32
// SINGLE:             iree_linalg_ext.yield %[[D10]] : i1
// SINGLE:           } -> tensor<3xf32>, tensor<3xi32>
// SINGLE:           return %[[D9:.*]]#0, %[[D9]]#1 : tensor<3xf32>, tensor<3xi32>
// SINGLE:         }

// -----

func.func @topk_split_reduction_nd(%input_values: tensor<3x10x40x8xf32>, %out_values: tensor<3x10x4x8xf32>, %out_indices: tensor<3x10x4x8xi32>) -> (tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>) {
  %0:2 = iree_linalg_ext.topk
        dimension(2)
        ins(%input_values : tensor<3x10x40x8xf32>)
        outs(%out_values, %out_indices : tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>
  return %0#0, %0#1 : tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>
}

// MULTIPLE-DAG:     #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// MULTIPLE-LABEL:   func.func @topk_split_reduction_nd(
// MULTIPLE-SAME:                                    %[[ARG0:.*]]: tensor<3x10x40x8xf32>,
// MULTIPLE-SAME:                                    %[[ARG1:.*]]: tensor<3x10x4x8xf32>,
// MULTIPLE-SAME:                                    %[[ARG2:.*]]: tensor<3x10x4x8xi32>) -> (tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>) {
// MULTIPLE-DAG:       %[[CNEG:.*]] = arith.constant 0xFF800000 : f32
// MULTIPLE-DAG:       %[[CPOS:.*]] = arith.constant 2147483647 : i32
// MULTIPLE-DAG:       %[[C10:.*]] = arith.constant 10 : i32
// MULTIPLE:           %[[D0:.*]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0], [1], [2, 3], [4]] : tensor<3x10x40x8xf32> into tensor<3x10x4x10x8xf32>
// MULTIPLE:           %[[D1:.*]] = linalg.init_tensor [3, 10, 4, 4, 8] : tensor<3x10x4x4x8xf32>
// MULTIPLE:           %[[D2:.*]] = linalg.init_tensor [3, 10, 4, 4, 8] : tensor<3x10x4x4x8xi32>
// MULTIPLE:           %[[D3:.*]] = linalg.fill ins(%[[CNEG]] : f32) outs(%[[D1]] : tensor<3x10x4x4x8xf32>) -> tensor<3x10x4x4x8xf32>
// MULTIPLE:           %[[D4:.*]] = linalg.fill ins(%[[CPOS]] : i32) outs(%[[D2]] : tensor<3x10x4x4x8xi32>) -> tensor<3x10x4x4x8xi32>
// MULTIPLE:           %[[D5:.*]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "SPLIT_REDUCTION"} dimension(3) ins(%[[D0]] : tensor<3x10x4x10x8xf32>) outs(%[[D3]], %[[D4]] : tensor<3x10x4x4x8xf32>, tensor<3x10x4x4x8xi32>) {
// MULTIPLE:           ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
// MULTIPLE:             %[[D10:.*]] = arith.cmpf ogt, %[[ARG3]], %[[ARG4]] : f32
// MULTIPLE:             iree_linalg_ext.yield %[[D10]] : i1
// MULTIPLE:           } -> tensor<3x10x4x4x8xf32>, tensor<3x10x4x4x8xi32>
// MULTIPLE:           %[[D6:.*]] = linalg.generic {indexing_maps = [#[[MAP0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[D5:.*]]#1 : tensor<3x10x4x4x8xi32>) {
// MULTIPLE:           ^bb0(%[[ARG3:.*]]: i32):
// MULTIPLE:             %[[D10:.*]] = linalg.index 2 : index
// MULTIPLE:             %[[D11:.*]] = arith.index_cast %[[D10]] : index to i32
// MULTIPLE:             %[[D12:.*]] = arith.muli %[[D11]], %[[C10]] : i32
// MULTIPLE:             %[[D13:.*]] = arith.addi %[[D12]], %[[ARG3]] : i32
// MULTIPLE:             linalg.yield %[[D13]] : i32
// MULTIPLE:           } -> tensor<3x10x4x4x8xi32>
// MULTIPLE:           %[[D7:.*]] = tensor.collapse_shape %[[D5:.*]]#0 {{\[\[}}0], [1], [2, 3], [4]] : tensor<3x10x4x4x8xf32> into tensor<3x10x16x8xf32>
// MULTIPLE:           %[[D8:.*]] = tensor.collapse_shape %[[D6:.*]] {{\[\[}}0], [1], [2, 3], [4]] : tensor<3x10x4x4x8xi32> into tensor<3x10x16x8xi32>
// MULTIPLE:           %[[D9:.*]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "SPLIT_REDUCTION"} dimension(2) ins(%[[D7]], %[[D8]] : tensor<3x10x16x8xf32>, tensor<3x10x16x8xi32>) outs(%[[ARG1]], %[[ARG2]] : tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>) {
// MULTIPLE:           ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
// MULTIPLE:             %[[D10:.*]] = arith.cmpf ogt, %[[ARG3]], %[[ARG4]] : f32
// MULTIPLE:             iree_linalg_ext.yield %[[D10]] : i1
// MULTIPLE:           } -> tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>
// MULTIPLE:           return %[[D9:.*]]#0, %[[D9]]#1 : tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>
// MULTIPLE:         }
