// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-linalg-ext-convert-conv2d-to-im2col-op))" %s | FileCheck %s

util.func public @conv_2d_nhwc_hwcf(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:      util.func public @conv_2d_nhwc_hwcf(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<3x3x4x16xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x14x14x16xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x196x36xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:   m_offset = [0] k_offset = [0]
// CHECK-SAME:   batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<1x16x16x4xf32>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<1x196x36xf32>) -> tensor<1x196x36xf32>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1, 2], [3]] : tensor<3x3x4x16xf32> into tensor<36x16xf32>
// CHECK-DAG:  %[[COLLAPSED_0:.+]] = tensor.collapse_shape %[[ARG2]] {{\[}}[0], [1, 2], [3]] : tensor<1x14x14x16xf32> into tensor<1x196x16xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[COLLAPSED]] : tensor<1x196x36xf32>, tensor<36x16xf32>)
// CHECK-SAME:   outs(%[[COLLAPSED_0]] : tensor<1x196x16xf32>) {
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<1x196x16xf32>
// CHECK:      %[[EXPANDED:.+]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0], [1, 2], [3]] output_shape [1, 14, 14, 16] : tensor<1x196x16xf32> into tensor<1x14x14x16xf32>
// CHECK:      util.return %[[EXPANDED]] : tensor<1x14x14x16xf32>

// -----

util.func public @conv_2d_nchw_fchw(%arg0: tensor<1x4x16x16xf32>, %arg1: tensor<16x4x3x3xf32>, %arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32> {
  %0 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x4x16x16xf32>, tensor<16x4x3x3xf32>)
    outs(%arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
  util.return %0 : tensor<1x16x14x14xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:      util.func public @conv_2d_nchw_fchw(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x4x16x16xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<16x4x3x3xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x16x14x14xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x196x36xf32>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:   m_offset = [0] k_offset = [0]
// CHECK-SAME:   batch_pos = [0] m_pos = [2, 3] k_pos = [1]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<1x4x16x16xf32>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<1x196x36xf32>) -> tensor<1x196x36xf32>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1, 2, 3]] : tensor<16x4x3x3xf32> into tensor<16x36xf32>
// CHECK-DAG:  %[[COLLAPSED_0:.+]] = tensor.collapse_shape %[[ARG2]] {{\[}}[0], [1], [2, 3]] : tensor<1x16x14x14xf32> into tensor<1x16x196xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[COLLAPSED]], %[[IM2COL]] : tensor<16x36xf32>, tensor<1x196x36xf32>)
// CHECK-SAME:   outs(%[[COLLAPSED_0]] : tensor<1x16x196xf32>) {
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<1x16x196xf32>
// CHECK:      %[[EXPANDED:.+]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0], [1], [2, 3]] output_shape [1, 16, 14, 14] : tensor<1x16x196xf32> into tensor<1x16x14x14xf32>
// CHECK:      util.return %[[EXPANDED]] : tensor<1x16x14x14xf32>

// -----

util.func public @conv_mixed_types(%arg0: tensor<1x16x16x4xf16>, %arg1: tensor<3x3x4x16xf16>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf16>, tensor<3x3x4x16xf16>)
    outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:      util.func public @conv_mixed_types(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf16>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<3x3x4x16xf16>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x14x14x16xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x196x36xf16>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:   m_offset = [0] k_offset = [0]
// CHECK-SAME:   batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<1x16x16x4xf16>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<1x196x36xf16>) -> tensor<1x196x36xf16>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1, 2], [3]] : tensor<3x3x4x16xf16> into tensor<36x16xf16>
// CHECK-DAG:  %[[COLLAPSED_0:.+]] = tensor.collapse_shape %[[ARG2]] {{\[}}[0], [1, 2], [3]] : tensor<1x14x14x16xf32> into tensor<1x196x16xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[COLLAPSED]] : tensor<1x196x36xf16>, tensor<36x16xf16>)
// CHECK-SAME:   outs(%[[COLLAPSED_0]] : tensor<1x196x16xf32>) {
// CHECK:          arith.extf
// CHECK:          arith.extf
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<1x196x16xf32>
// CHECK:      %[[EXPANDED:.+]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0], [1, 2], [3]] output_shape [1, 14, 14, 16] : tensor<1x196x16xf32> into tensor<1x14x14x16xf32>
// CHECK:      util.return %[[EXPANDED]] : tensor<1x14x14x16xf32>

// -----

util.func public @conv_strided(%arg0: tensor<1x16x16x4xf16>, %arg1: tensor<3x3x4x16xf16>, %arg2: tensor<1x7x7x16xf32>) -> tensor<1x7x7x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf16>, tensor<3x3x4x16xf16>)
    outs(%arg2: tensor<1x7x7x16xf32>) -> tensor<1x7x7x16xf32>
  util.return %0 : tensor<1x7x7x16xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:      util.func public @conv_strided(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf16>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<3x3x4x16xf16>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x7x7x16xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x49x36xf16>
// CHECK:      %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-SAME:   strides = [2, 2] dilations = [1, 1] kernel_size = [3, 3]
// CHECK-SAME:   m_offset = [0] k_offset = [0]
// CHECK-SAME:   batch_pos = [0] m_pos = [1, 2] k_pos = [3]
// CHECK-SAME:   ins(%[[ARG0]] : tensor<1x16x16x4xf16>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<1x49x36xf16>) -> tensor<1x49x36xf16>
// CHECK-DAG:  %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1, 2], [3]] : tensor<3x3x4x16xf16> into tensor<36x16xf16>
// CHECK-DAG:  %[[COLLAPSED_0:.+]] = tensor.collapse_shape %[[ARG2]] {{\[}}[0], [1, 2], [3]] : tensor<1x7x7x16xf32> into tensor<1x49x16xf32>
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[COLLAPSED]] : tensor<1x49x36xf16>, tensor<36x16xf16>)
// CHECK-SAME:   outs(%[[COLLAPSED_0]] : tensor<1x49x16xf32>) {
// CHECK:          arith.extf
// CHECK:          arith.extf
// CHECK:          arith.mulf
// CHECK:          arith.addf
// CHECK:      } -> tensor<1x49x16xf32>
// CHECK:      %[[EXPANDED:.+]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0], [1, 2], [3]] output_shape [1, 7, 7, 16] : tensor<1x49x16xf32> into tensor<1x7x7x16xf32>
// CHECK:      util.return %[[EXPANDED]] : tensor<1x7x7x16xf32>
