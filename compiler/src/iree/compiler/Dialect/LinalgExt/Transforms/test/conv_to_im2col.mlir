// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-linalg-ext-convert-conv-to-im2col-op))" %s | FileCheck %s

util.func public @conv_2d_nhwc_hwcf(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d3)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK:      util.func public @conv_2d_nhwc_hwcf(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<3x3x4x16xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x14x14x16xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x14x14x36xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:   m_offset = [0, 0] * [14, 1] k_offset = [0] * [1]
// CHECK-SAME:   batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<1x16x16x4xf32>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<1x14x14x36xf32>) -> tensor<1x14x14x36xf32>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1, 2], [3]] : tensor<3x3x4x16xf32> into tensor<36x16xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[COLLAPSED]] : tensor<1x14x14x36xf32>, tensor<36x16xf32>)
// CHECK-SAME:   outs(%[[ARG2]] : tensor<1x14x14x16xf32>) {
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<1x14x14x16xf32>
// CHECK:      util.return %[[MATMUL]] : tensor<1x14x14x16xf32>

// -----

util.func public @conv_2d_nchw_fchw(%arg0: tensor<1x4x16x16xf32>, %arg1: tensor<16x4x3x3xf32>, %arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32> {
  %0 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x4x16x16xf32>, tensor<16x4x3x3xf32>)
    outs(%arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
  util.return %0 : tensor<1x16x14x14xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK:      util.func public @conv_2d_nchw_fchw(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x4x16x16xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<16x4x3x3xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x16x14x14xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x14x14x36xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:   m_offset = [0, 0] * [14, 1] k_offset = [0] * [1]
// CHECK-SAME:   batch_pos = [0] m_pos = [2, 3] k_pos = [1]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<1x4x16x16xf32>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<1x14x14x36xf32>) -> tensor<1x14x14x36xf32>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1, 2, 3]] : tensor<16x4x3x3xf32> into tensor<16x36xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[COLLAPSED]], %[[IM2COL]] : tensor<16x36xf32>, tensor<1x14x14x36xf32>)
// CHECK-SAME:   outs(%[[ARG2]] : tensor<1x16x14x14xf32>) {
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<1x16x14x14xf32>
// CHECK:      util.return %[[MATMUL]] : tensor<1x16x14x14xf32>

// -----

util.func public @conv_mixed_types(%arg0: tensor<1x16x16x4xf16>, %arg1: tensor<3x3x4x16xf16>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf16>, tensor<3x3x4x16xf16>)
    outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d3)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK:      util.func public @conv_mixed_types(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf16>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<3x3x4x16xf16>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x14x14x16xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x14x14x36xf16>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:   m_offset = [0, 0] * [14, 1] k_offset = [0] * [1]
// CHECK-SAME:   batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<1x16x16x4xf16>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<1x14x14x36xf16>) -> tensor<1x14x14x36xf16>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1, 2], [3]] : tensor<3x3x4x16xf16> into tensor<36x16xf16>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[COLLAPSED]] : tensor<1x14x14x36xf16>, tensor<36x16xf16>)
// CHECK-SAME:   outs(%[[ARG2]] : tensor<1x14x14x16xf32>) {
// CHECK:          arith.extf
// CHECK:          arith.extf
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<1x14x14x16xf32>
// CHECK:      util.return %[[MATMUL]] : tensor<1x14x14x16xf32>

// -----

util.func public @conv_strided(%arg0: tensor<1x16x16x4xf16>, %arg1: tensor<3x3x4x16xf16>, %arg2: tensor<1x7x7x16xf32>) -> tensor<1x7x7x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf16>, tensor<3x3x4x16xf16>)
    outs(%arg2: tensor<1x7x7x16xf32>) -> tensor<1x7x7x16xf32>
  util.return %0 : tensor<1x7x7x16xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d3)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK:      util.func public @conv_strided(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf16>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<3x3x4x16xf16>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x7x7x16xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x7x7x36xf16>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [2, 2] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:   m_offset = [0, 0] * [7, 1] k_offset = [0] * [1]
// CHECK-SAME:   batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<1x16x16x4xf16>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<1x7x7x36xf16>) -> tensor<1x7x7x36xf16>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1, 2], [3]] : tensor<3x3x4x16xf16> into tensor<36x16xf16>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[COLLAPSED]] : tensor<1x7x7x36xf16>, tensor<36x16xf16>)
// CHECK-SAME:   outs(%[[ARG2]] : tensor<1x7x7x16xf32>) {
// CHECK:          arith.extf
// CHECK:          arith.extf
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<1x7x7x16xf32>
// CHECK:      util.return %[[MATMUL]] : tensor<1x7x7x16xf32>

// -----
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2 + d5, d3 + d6, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d1, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d1)>
util.func public @conv_nhwc_hwfc(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x16x4xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<3x3x16x4xf32>) outs(%arg2 : tensor<1x14x14x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d5)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d3, d5)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK:      util.func public @conv_nhwc_hwfc(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<3x3x16x4xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x14x14x16xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x14x14x9x4xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   m_offset = [0, 0] * [14, 1] k_offset = [0, 0] * [4, 1]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<1x16x16x4xf32>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<1x14x14x9x4xf32>) -> tensor<1x14x14x9x4xf32>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]] : tensor<3x3x16x4xf32> into tensor<9x16x4xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[COLLAPSED]] : tensor<1x14x14x9x4xf32>, tensor<9x16x4xf32>)
// CHECK:      util.return %[[MATMUL]] : tensor<1x14x14x16xf32>

// -----
util.func public @conv_2d_nhwc_fhwc(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<16x3x3x4xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<16x3x3x4xf32>)
    outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK:      util.func public @conv_2d_nhwc_fhwc(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<16x3x3x4xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x14x14x16xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x14x14x36xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1, 2, 3]] : tensor<16x3x3x4xf32> into tensor<16x36xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[COLLAPSED]] : tensor<1x14x14x36xf32>, tensor<16x36xf32>)
// CHECK:      util.return %[[MATMUL]] : tensor<1x14x14x16xf32>

// -----

util.func public @conv_1d_ncw_fcw_transpose_maps(%arg0: tensor<1x8x130xf32>, %arg1: tensor<16x8x3xf32>) -> tensor<1x16x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1x16x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x16x128xf32>) -> tensor<1x16x128xf32>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1 + d3)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d2, d4, d3)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1)>],
      iterator_types =
      ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1 : tensor<1x8x130xf32>,tensor<16x8x3xf32>)
      outs(%fill : tensor<1x16x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.mulf %in, %in_0 : f32
      %9 = arith.addf %out, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<1x16x128xf32>
  util.return %0 : tensor<1x16x128xf32>
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:      util.func public @conv_1d_ncw_fcw_transpose_maps(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x8x130xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<16x8x3xf32>
// CHECK:      %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x16x128xf32>
// CHECK:      %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<1x16x128xf32>) -> tensor<1x16x128xf32>
// CHECK:      %[[EMPTY2:.+]] = tensor.empty() : tensor<1x128x24xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1] dilations = [1] kernel_size = [3]
// CHECK-SAME:   m_offset = [0] * [1] k_offset = [0] * [1]
// CHECK-SAME:   batch_pos = [0] m_pos = [2] k_pos = [1]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<1x8x130xf32>)
// CHECK-SAME:   outs(%[[EMPTY2]] : tensor<1x128x24xf32>) -> tensor<1x128x24xf32>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1, 2]] : tensor<16x8x3xf32> into tensor<16x24xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[COLLAPSED]], %[[IM2COL]] : tensor<16x24xf32>, tensor<1x128x24xf32>)
// CHECK-SAME:   outs(%[[FILL]] : tensor<1x16x128xf32>) {
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<1x16x128xf32>
// CHECK:      util.return %[[MATMUL]] : tensor<1x16x128xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5, d2 + d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @conv_2d_chwn_chwf(%arg0: tensor<16x26x18x288xf32>, %arg1: tensor<16x24x16x288xf32>, %arg2: tensor<288x3x3x288xf32>) -> tensor<288x3x3x288xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x26x18x288xf32>, tensor<16x24x16x288xf32>) outs(%arg2 : tensor<288x3x3x288xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %12 = arith.mulf %in, %in_3 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<288x3x3x288xf32>
  util.return %0 : tensor<288x3x3x288xf32>
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d1, d2, d4)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK:      util.func public @conv_2d_chwn_chwf(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<16x26x18x288xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<16x24x16x288xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<288x3x3x288xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<288x3x3x6144xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
// CHECK-SAME:   m_offset = [0, 0] * [3, 1] k_offset = [0] * [1]
// CHECK-SAME:   batch_pos = [3] m_pos = [1, 2] k_pos = [0]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<16x26x18x288xf32>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<288x3x3x6144xf32>) -> tensor<288x3x3x6144xf32>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1, 2], [3]] : tensor<16x24x16x288xf32> into tensor<6144x288xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[COLLAPSED]], %[[IM2COL]] : tensor<6144x288xf32>, tensor<288x3x3x6144xf32>)
// CHECK-SAME:   outs(%[[ARG2]] : tensor<288x3x3x288xf32>) {
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<288x3x3x288xf32>
// CHECK:      util.return %[[MATMUL]] : tensor<288x3x3x288xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0 + d4, d1 + d5, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @conv_2d_hwcn_hwcf(%arg0: tensor<26x18x16x288xf32>, %arg1: tensor<24x16x16x288xf32>, %arg2: tensor<3x3x288x288xf32>) -> tensor<3x3x288x288xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<26x18x16x288xf32>, tensor<24x16x16x288xf32>) outs(%arg2 : tensor<3x3x288x288xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %12 = arith.mulf %in, %in_3 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<3x3x288x288xf32>
  util.return %0 : tensor<3x3x288x288xf32>
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d1, d4)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK:      util.func public @conv_2d_hwcn_hwcf(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<26x18x16x288xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<24x16x16x288xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<3x3x288x288xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<288x3x3x6144xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
// CHECK-SAME:   m_offset = [0, 0] * [3, 1] k_offset = [0] * [1]
// CHECK-SAME:   batch_pos = [3] m_pos = [0, 1] k_pos = [2]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<26x18x16x288xf32>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<288x3x3x6144xf32>) -> tensor<288x3x3x6144xf32>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1, 2], [3]] : tensor<24x16x16x288xf32> into tensor<6144x288xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[COLLAPSED]] : tensor<288x3x3x6144xf32>,  tensor<6144x288xf32>)
// CHECK-SAME:   outs(%[[ARG2]] : tensor<3x3x288x288xf32>) {
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<3x3x288x288xf32>
// CHECK:      util.return %[[MATMUL]] : tensor<3x3x288x288xf32>
