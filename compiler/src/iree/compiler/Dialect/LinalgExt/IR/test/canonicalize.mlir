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
