// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-global-opt-data-layout-propagation))" --split-input-file %s | FileCheck %s

func.func @bubble_up_pack_through_collapse(%1: tensor<?x16x4xf32>, %dim : index) -> tensor<?x4x8x1xf32> {
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<?x16x4xf32> into tensor<?x4xf32>
  %2 = tensor.empty(%dim) : tensor<?x4x8x1xf32>
  %pack = tensor.pack %collapsed outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %2 : tensor<?x4xf32> -> tensor<?x4x8x1xf32>
  func.return %pack : tensor<?x4x8x1xf32>
}
// CHECK-LABEL: func.func @bubble_up_pack_through_collapse
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x16x4xf32>
// CHECK:         %[[EMPTY:.+]] = tensor.empty(%[[DIM]]) : tensor<?x2x4x8x1xf32>
// CHECK:         %[[PACK:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 1] into %[[EMPTY]] : tensor<?x16x4xf32> -> tensor<?x2x4x8x1xf32>
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[PACK]] {{\[}}[0, 1], [2], [3], [4]] : tensor<?x2x4x8x1xf32> into tensor<?x4x8x1xf32>
// CHECK:         return %[[COLLAPSED]] : tensor<?x4x8x1xf32>

// -----

func.func @push_down_unpack_through_expand(%5: tensor<?x32x8x8xf32>, %dim: index) -> tensor<?x256x256xf32> {
  %6 = tensor.empty(%dim) : tensor<?x256xf32>
  %unpack = tensor.unpack %5 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %6 : tensor<?x32x8x8xf32> -> tensor<?x256xf32>
  %expanded = tensor.expand_shape %unpack [[0, 1], [2]] : tensor<?x256xf32> into tensor<?x256x256xf32>
  func.return %expanded : tensor<?x256x256xf32>
}
// CHECK-LABEL: func.func @push_down_unpack_through_expand
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9]+]]
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3], [4]] : tensor<?x32x8x8xf32> into tensor<?x32x32x8x8xf32>
// CHECK:         %[[DIM:.+]] = tensor.dim %[[EXPANDED]], %[[C0]] : tensor<?x32x32x8x8xf32>
// CHECK:         %[[EMPTY:.+]] = tensor.empty(%[[DIM]]) : tensor<?x256x256xf32>
// CHECK:         %[[UNPACK:.+]] = tensor.unpack %[[EXPANDED:.+]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %[[EMPTY]] : tensor<?x32x32x8x8xf32> -> tensor<?x256x256xf32>
// CHECK:         return %[[UNPACK]] : tensor<?x256x256xf32>
