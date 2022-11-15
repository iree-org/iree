// RUN: iree-dialects-opt -iree-linalg-ext-fold-into-pack-unpack-ops %s | FileCheck %s

func.func @fold_unpack_slice(%arg0 : tensor<?x?x8x4xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %arg1
      : (tensor<?x?x8x4xf32> tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0[0, 0] [%arg2, %arg3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func @fold_unpack_slice(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x8x4xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
//      CHECK:   %[[INIT:.+]] = tensor.empty(%[[ARG2]], %[[ARG3]]) : tensor<?x?xf32>
//      CHECK:   %[[UNPACK:.+]] = iree_linalg_ext.unpack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [8, 4]
// CHECK-SAME:       into %[[INIT]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @nofold_unpack_slice_non_zero_offset(%arg0 : tensor<?x?x8x4xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index, %arg4 : index) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %arg1
      : (tensor<?x?x8x4xf32> tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0[0, %arg4] [%arg2, %arg3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @nofold_unpack_slice_non_zero_offset(
//       CHECK:   %[[UNPACK:.+]] = iree_linalg_ext.unpack
//       CHECK:   tensor.extract_slice %[[UNPACK]]

// -----

func.func @nofold_unpack_slice_non_unit_stride(%arg0 : tensor<?x?x8x4xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index, %arg4 : index) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %arg1
      : (tensor<?x?x8x4xf32> tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0[0, 0] [%arg2, %arg3] [%arg4, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @nofold_unpack_slice_non_unit_stride(
//       CHECK:   %[[UNPACK:.+]] = iree_linalg_ext.unpack
//       CHECK:   tensor.extract_slice %[[UNPACK]]
