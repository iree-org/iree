// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-cpu-propagate-data-layout))" --split-input-file %s | FileCheck %s

func.func @collapsing_unit_dim_0(%src: tensor<1x2x1x16xi32>) -> tensor<20xi32> {
  %collapsed = tensor.collapse_shape %src [[0, 1], [2, 3]] : tensor<1x2x1x16xi32> into tensor<2x16xi32>
  %1 = tensor.empty() : tensor<20xi32>
  %unpack = linalg.unpack %collapsed inner_dims_pos = [0] inner_tiles = [16] into %1 : tensor<2x16xi32> -> tensor<20xi32>
  return %unpack : tensor<20xi32>
}
// CHECK-LABEL: func.func @collapsing_unit_dim_0(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[SRC]]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [1, 16]
// CHECK-SAME:      : tensor<1x2x1x16xi32> -> tensor<1x20xi32>
// CHECK-NEXT:    %[[COLLAPSED:.+]] = tensor.collapse_shape %[[UNPACK]]
// CHECK-SAME:      : tensor<1x20xi32> into tensor<20xi32>
// CHECK:         return %[[COLLAPSED]]

// -----

func.func @collapsing_unit_dim_1(%src: tensor<?x1x1x1x16xi32>, %batch_size: index) -> tensor<?x3xi32> {
  %collapsed = tensor.collapse_shape %src [[0], [1, 2], [3, 4]] : tensor<?x1x1x1x16xi32> into tensor<?x1x16xi32>
  %0 = tensor.empty(%batch_size) : tensor<?x3xi32>
  %unpack = linalg.unpack %collapsed inner_dims_pos = [1] inner_tiles = [16] into %0 : tensor<?x1x16xi32> -> tensor<?x3xi32>
  return %unpack : tensor<?x3xi32>
}
// CHECK-LABEL: func.func @collapsing_unit_dim_1(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[SRC]]
// CHECK-SAME:      inner_dims_pos = [1, 2] inner_tiles = [1, 16]
// CHECK-SAME:      : tensor<?x1x1x1x16xi32> -> tensor<?x1x3xi32>
// CHECK-NEXT:    %[[COLLAPSED:.+]] = tensor.collapse_shape %[[UNPACK]]
// CHECK-SAME:      : tensor<?x1x3xi32> into tensor<?x3xi32>
// CHECK:         return %[[COLLAPSED]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @collapsing_unit_dim_0_elem_unpack(%src: tensor<1x1x1x16xi32>) -> tensor<3xi32> {
  %0 = tensor.empty() : tensor<1x16xi32>
  %collapsed = tensor.collapse_shape %src [[0, 1], [2, 3]] : tensor<1x1x1x16xi32> into tensor<1x16xi32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed : tensor<1x16xi32>) outs(%0 : tensor<1x16xi32>) {
  ^bb0(%in: i32, %out: i32):
    %3 = arith.addi %in, %in : i32
    linalg.yield %3 : i32
  } -> tensor<1x16xi32>
  %2 = tensor.empty() : tensor<3xi32>
  %unpack = linalg.unpack %1 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %2 : tensor<1x16xi32> -> tensor<3xi32>
  return %unpack : tensor<3xi32>
}
// CHECK-LABEL: func.func @collapsing_unit_dim_0_elem_unpack(
// CHECK:         %[[ELEM:.+]] = linalg.generic
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[ELEM]]
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [1, 16]
// CHECK-SAME:      : tensor<1x1x1x16xi32> -> tensor<1x3xi32>
// CHECK-NEXT:    %[[COLLAPSED:.+]] = tensor.collapse_shape %[[UNPACK]]
// CHECK-SAME:      : tensor<1x3xi32> into tensor<3xi32>
// CHECK:         return %[[COLLAPSED]]

// -----

func.func @negative_unpack_with_outer_dims_perm(%src: tensor<1x1x?x1x16xi32>, %batch_size: index) -> tensor<?x3xi32> {
  %collapsed = tensor.collapse_shape %src [[0], [1, 2], [3, 4]] : tensor<1x1x?x1x16xi32> into tensor<1x?x16xi32>
  %0 = tensor.empty(%batch_size) : tensor<?x3xi32>
  %unpack = linalg.unpack %collapsed outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [16] into %0 : tensor<1x?x16xi32> -> tensor<?x3xi32>
  return %unpack : tensor<?x3xi32>
}
// CHECK-LABEL: func.func @negative_unpack_with_outer_dims_perm(
// CHECK:         tensor.collapse_shape
// CHECK:         linalg.unpack

// -----

func.func @negative_unpack_multiple_dims(%src: tensor<?x1x1x1x16x8xi32>, %d0: index, %d1: index) -> tensor<?x?xi32> {
  %collapsed = tensor.collapse_shape %src [[0], [1, 2], [3, 4], [5]] : tensor<?x1x1x1x16x8xi32> into tensor<?x1x16x8xi32>
  %0 = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %unpack = linalg.unpack %collapsed inner_dims_pos = [0, 1] inner_tiles = [16, 8] into %0 : tensor<?x1x16x8xi32> -> tensor<?x?xi32>
  return %unpack : tensor<?x?xi32>
}
// CHECK-LABEL: func.func @negative_unpack_multiple_dims(
// CHECK:         tensor.collapse_shape
// CHECK:         linalg.unpack

// -----

func.func @negative_unpack_non_collapsed_dim(%src: tensor<?x1x1x1x16xi32>, %d0: index) -> tensor<?x1xi32> {
  %collapsed = tensor.collapse_shape %src [[0], [1, 2], [3, 4]] : tensor<?x1x1x1x16xi32> into tensor<?x1x16xi32>
  %0 = tensor.empty(%d0) : tensor<?x1xi32>
  %unpack = linalg.unpack %collapsed inner_dims_pos = [0] inner_tiles = [16] into %0 : tensor<?x1x16xi32> -> tensor<?x1xi32>
  return %unpack : tensor<?x1xi32>
}
// CHECK-LABEL: func.func @negative_unpack_non_collapsed_dim(
// CHECK:         tensor.collapse_shape
// CHECK:         linalg.unpack

// -----

func.func @negative_both_m_n_non_unit_dim(%src: tensor<3x4x2x8xi32>) -> tensor<180xi32> {
  %collapsed = tensor.collapse_shape %src [[0, 1], [2, 3]] : tensor<3x4x2x8xi32> into tensor<12x16xi32>
  %1 = tensor.empty() : tensor<180xi32>
  %unpack = linalg.unpack %collapsed inner_dims_pos = [0] inner_tiles = [16] into %1 : tensor<12x16xi32> -> tensor<180xi32>
  return %unpack : tensor<180xi32>
}
// CHECK-LABEL: func.func @negative_both_m_n_non_unit_dim(
// CHECK:         tensor.collapse_shape
// CHECK:         linalg.unpack

// -----

func.func @negative_innermost_dim_is_not_collapsed(%src: tensor<1x3x1x8x16xi32>) -> tensor<48x8xi32> {
  %collapsed = tensor.collapse_shape %src [[0, 1], [2, 3], [4]] : tensor<1x3x1x8x16xi32> into tensor<3x8x16xi32>
  %1 = tensor.empty() : tensor<48x8xi32>
  %unpack = linalg.unpack %collapsed inner_dims_pos = [0] inner_tiles = [16] into %1 : tensor<3x8x16xi32> -> tensor<48x8xi32>
  return %unpack : tensor<48x8xi32>
}
// CHECK-LABEL: func.func @negative_innermost_dim_is_not_collapsed(
// CHECK:         tensor.collapse_shape
// CHECK:         linalg.unpack
