// RUN: iree-dialects-opt --test-topk-split-reduction='split-ratio=3' %s | FileCheck %s --check-prefix SINGLE
// RUN: iree-dialects-opt --test-topk-split-reduction='split-ratio=4' %s | FileCheck %s --check-prefix MULTIPLE

func.func @topk_split_reduction_1d(%input_values: tensor<30xf32>, %input_indices: tensor<30xi32>, %out_values: tensor<3xf32>, %out_indices: tensor<3xi32>) -> (tensor<3xf32>, tensor<3xi32>) {
  %0:2 = iree_linalg_ext.topk
        dimension(0)
        ins(%input_values, %input_indices : tensor<30xf32> , tensor<30xi32>)
        outs(%out_values, %out_indices : tensor<3xf32>, tensor<3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<3xf32>, tensor<3xi32>
  return %0#0, %0#1 : tensor<3xf32>, tensor<3xi32>
}

// SINGLE-LABEL:   func.func @topk_split_reduction_1d(
// SINGLE-SAME:                                    %[[VAL_0:.*]]: tensor<30xf32>,
// SINGLE-SAME:                                    %[[VAL_1:.*]]: tensor<30xi32>,
// SINGLE-SAME:                                    %[[VAL_2:.*]]: tensor<3xf32>,
// SINGLE-SAME:                                    %[[VAL_3:.*]]: tensor<3xi32>) -> (tensor<3xf32>, tensor<3xi32>) {
// SINGLE:           %[[VAL_4:.*]] = arith.constant 0xFF800000 : f32
// SINGLE:           %[[VAL_5:.*]] = arith.constant 2147483647 : i32
// SINGLE:           %[[VAL_6:.*]] = arith.constant 3 : i32
// SINGLE:           %[[VAL_7:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1]] : tensor<30xf32> into tensor<3x10xf32>
// SINGLE:           %[[VAL_8:.*]] = tensor.expand_shape %[[VAL_1]] {{\[\[}}0, 1]] : tensor<30xi32> into tensor<3x10xi32>
// SINGLE:           %[[VAL_9:.*]] = linalg.init_tensor [3, 3] : tensor<3x3xf32>
// SINGLE:           %[[VAL_10:.*]] = linalg.init_tensor [3, 3] : tensor<3x3xi32>
// SINGLE:           %[[VAL_11:.*]] = linalg.fill ins(%[[VAL_4]] : f32) outs(%[[VAL_9]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// SINGLE:           %[[VAL_12:.*]] = linalg.fill ins(%[[VAL_5]] : i32) outs(%[[VAL_10]] : tensor<3x3xi32>) -> tensor<3x3xi32>
// SINGLE:           %[[VAL_13:.*]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "SPLIT_REDUCTION"} dimension(1) ins(%[[VAL_7]], %[[VAL_8]] : tensor<3x10xf32>, tensor<3x10xi32>) outs(%[[VAL_11]], %[[VAL_12]] : tensor<3x3xf32>, tensor<3x3xi32>) {
// SINGLE:           ^bb0(%[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: f32):
// SINGLE:             %[[VAL_16:.*]] = arith.cmpf ogt, %[[VAL_14]], %[[VAL_15]] : f32
// SINGLE:             iree_linalg_ext.yield %[[VAL_16]] : i1
// SINGLE:           } -> tensor<3x3xf32>, tensor<3x3xi32>
// SINGLE:           %[[VAL_17:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[VAL_18:.*]]#1 : tensor<3x3xi32>) {
// SINGLE:           ^bb0(%[[VAL_19:.*]]: i32):
// SINGLE:             %[[VAL_20:.*]] = linalg.index 0 : index
// SINGLE:             %[[VAL_21:.*]] = arith.index_cast %[[VAL_20]] : index to i32
// SINGLE:             %[[VAL_22:.*]] = arith.muli %[[VAL_21]], %[[VAL_6]] : i32
// SINGLE:             %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_19]] : i32
// SINGLE:             linalg.yield %[[VAL_23]] : i32
// SINGLE:           } -> tensor<3x3xi32>
// SINGLE:           %[[VAL_24:.*]] = tensor.collapse_shape %[[VAL_25:.*]]#0 {{\[\[}}0, 1]] : tensor<3x3xf32> into tensor<9xf32>
// SINGLE:           %[[VAL_26:.*]] = tensor.collapse_shape %[[VAL_27:.*]] {{\[\[}}0, 1]] : tensor<3x3xi32> into tensor<9xi32>
// SINGLE:           %[[VAL_28:.*]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "SPLIT_REDUCTION"} dimension(0) ins(%[[VAL_24]], %[[VAL_26]] : tensor<9xf32>, tensor<9xi32>) outs(%[[VAL_2]], %[[VAL_3]] : tensor<3xf32>, tensor<3xi32>) {
// SINGLE:           ^bb0(%[[VAL_29:.*]]: f32, %[[VAL_30:.*]]: f32):
// SINGLE:             %[[VAL_31:.*]] = arith.cmpf ogt, %[[VAL_29]], %[[VAL_30]] : f32
// SINGLE:             iree_linalg_ext.yield %[[VAL_31]] : i1
// SINGLE:           } -> tensor<3xf32>, tensor<3xi32>
// SINGLE:           return %[[VAL_32:.*]]#0, %[[VAL_32]]#1 : tensor<3xf32>, tensor<3xi32>
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

// MULTIPLE-LABEL:   func.func @topk_split_reduction_nd(
// MULTIPLE-SAME:                                    %[[VAL_0:.*]]: tensor<3x10x40x8xf32>,
// MULTIPLE-SAME:                                    %[[VAL_1:.*]]: tensor<3x10x4x8xf32>,
// MULTIPLE-SAME:                                    %[[VAL_2:.*]]: tensor<3x10x4x8xi32>) -> (tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>) {
// MULTIPLE:           %[[VAL_3:.*]] = arith.constant 0xFF800000 : f32
// MULTIPLE:           %[[VAL_4:.*]] = arith.constant 2147483647 : i32
// MULTIPLE:           %[[VAL_5:.*]] = arith.constant 4 : i32
// MULTIPLE:           %[[VAL_6:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2, 3], [4]] : tensor<3x10x40x8xf32> into tensor<3x10x4x10x8xf32>
// MULTIPLE:           %[[VAL_7:.*]] = linalg.init_tensor [3, 10, 4, 4, 8] : tensor<3x10x4x4x8xf32>
// MULTIPLE:           %[[VAL_8:.*]] = linalg.init_tensor [3, 10, 4, 4, 8] : tensor<3x10x4x4x8xi32>
// MULTIPLE:           %[[VAL_9:.*]] = linalg.fill ins(%[[VAL_3]] : f32) outs(%[[VAL_7]] : tensor<3x10x4x4x8xf32>) -> tensor<3x10x4x4x8xf32>
// MULTIPLE:           %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_4]] : i32) outs(%[[VAL_8]] : tensor<3x10x4x4x8xi32>) -> tensor<3x10x4x4x8xi32>
// MULTIPLE:           %[[VAL_11:.*]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "SPLIT_REDUCTION"} dimension(3) ins(%[[VAL_6]] : tensor<3x10x4x10x8xf32>) outs(%[[VAL_9]], %[[VAL_10]] : tensor<3x10x4x4x8xf32>, tensor<3x10x4x4x8xi32>) {
// MULTIPLE:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// MULTIPLE:             %[[VAL_14:.*]] = arith.cmpf ogt, %[[VAL_12]], %[[VAL_13]] : f32
// MULTIPLE:             iree_linalg_ext.yield %[[VAL_14]] : i1
// MULTIPLE:           } -> tensor<3x10x4x4x8xf32>, tensor<3x10x4x4x8xi32>
// MULTIPLE:           %[[VAL_15:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL_16:.*]]#1 : tensor<3x10x4x4x8xi32>) {
// MULTIPLE:           ^bb0(%[[VAL_17:.*]]: i32):
// MULTIPLE:             %[[VAL_18:.*]] = linalg.index 2 : index
// MULTIPLE:             %[[VAL_19:.*]] = arith.index_cast %[[VAL_18]] : index to i32
// MULTIPLE:             %[[VAL_20:.*]] = arith.muli %[[VAL_19]], %[[VAL_5]] : i32
// MULTIPLE:             %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_17]] : i32
// MULTIPLE:             linalg.yield %[[VAL_21]] : i32
// MULTIPLE:           } -> tensor<3x10x4x4x8xi32>
// MULTIPLE:           %[[VAL_22:.*]] = tensor.collapse_shape %[[VAL_23:.*]]#0 {{\[\[}}0], [1], [2, 3], [4]] : tensor<3x10x4x4x8xf32> into tensor<3x10x16x8xf32>
// MULTIPLE:           %[[VAL_24:.*]] = tensor.collapse_shape %[[VAL_25:.*]] {{\[\[}}0], [1], [2, 3], [4]] : tensor<3x10x4x4x8xi32> into tensor<3x10x16x8xi32>
// MULTIPLE:           %[[VAL_26:.*]]:2 = iree_linalg_ext.topk {__internal_linalg_transform__ = "SPLIT_REDUCTION"} dimension(2) ins(%[[VAL_22]], %[[VAL_24]] : tensor<3x10x16x8xf32>, tensor<3x10x16x8xi32>) outs(%[[VAL_1]], %[[VAL_2]] : tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>) {
// MULTIPLE:           ^bb0(%[[VAL_27:.*]]: f32, %[[VAL_28:.*]]: f32):
// MULTIPLE:             %[[VAL_29:.*]] = arith.cmpf ogt, %[[VAL_27]], %[[VAL_28]] : f32
// MULTIPLE:             iree_linalg_ext.yield %[[VAL_29]] : i1
// MULTIPLE:           } -> tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>
// MULTIPLE:           return %[[VAL_30:.*]]#0, %[[VAL_30]]#1 : tensor<3x10x4x8xf32>, tensor<3x10x4x8xi32>
// MULTIPLE:         }
