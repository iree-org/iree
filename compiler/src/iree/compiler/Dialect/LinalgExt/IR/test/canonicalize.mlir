// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

func.func @pack_canonicalize(%arg0 : tensor<?x?xi32>,
    %arg1 : tensor<1x2x3x3xi32>) -> tensor<1x?x3x3xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.cast %arg1 : tensor<1x2x3x3xi32> to tensor<1x?x3x3xi32>
  %1 = iree_linalg_ext.pack %arg0 padding_value(%c0_i32 : i32)
      inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %0
      : (tensor<?x?xi32> tensor<1x?x3x3xi32>) -> tensor<1x?x3x3xi32>
  return %1 : tensor<1x?x3x3xi32>
}
// CHECK-LABEL: func.func @pack_canonicalize
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xi32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<1x2x3x3xi32>
//       CHECK:   %[[PAD_VALUE:.+]] = arith.constant 0 : i32
//       CHECK:   %[[PACK:.+]] = iree_linalg_ext.pack %[[ARG0]]
//  CHECK-SAME:       padding_value(%[[PAD_VALUE]] : i32)
//  CHECK-SAME:       into %[[ARG1]]
//       CHECK:   %[[CAST:.+]] = tensor.cast %[[PACK]]
//       CHECK:   return %[[CAST]]

// -----

func.func @sort_drop_unused_results(%arg0 : tensor<?x10xf32>,
    %arg1 : tensor<?x10xi64>) -> tensor<?x10xf32> {
  %0:2 = iree_linalg_ext.sort dimension(1) outs(%arg0, %arg1: tensor<?x10xf32>,
      tensor<?x10xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %42 = arith.cmpf oge, %arg2, %arg3 : f32
    iree_linalg_ext.yield %42 : i1
  } -> tensor<?x10xf32>, tensor<?x10xi64>
  return %0#0 : tensor<?x10xf32>
}
// CHECK-LABEL: func.func @sort_drop_unused_results
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x10xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?x10xi64>
//       CHECK:   %[[SORT:.+]] = iree_linalg_ext.sort dimension(1) outs(%[[ARG0]] : tensor<?x10xf32>)
