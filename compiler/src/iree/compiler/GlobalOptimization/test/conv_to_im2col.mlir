// RUN: iree-opt --split-input-file --iree-global-opt-convert-conv-to-im2col %s | FileCheck %s

// Channels-last: the batch and output image loops stay contiguous in the
// gathered input and the output, so the convolution becomes a plain matmul.
func.func @conv_nhwc_hwcf(%input: tensor<1x16x16x4xf32>, %filter: tensor<3x3x4x16xf32>,
                          %init: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
      outs(%init : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  return %0 : tensor<1x14x14x16xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>
// CHECK-DAG:  #[[$ID6:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @conv_nhwc_hwcf(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x16x16x4xf32>
//  CHECK-SAME:     %[[FILTER:[a-zA-Z0-9_]+]]: tensor<3x3x4x16xf32>
//  CHECK-SAME:     %[[INIT:[a-zA-Z0-9_]+]]: tensor<1x14x14x16xf32>
//       CHECK:   %[[GATHER_INIT:.+]] = tensor.empty() : tensor<1x14x14x3x3x4xf32>
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]], #[[$ID6]]]
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<1x16x16x4xf32>) outs(%[[GATHER_INIT]] : tensor<1x14x14x3x3x4xf32>)
//       CHECK:   %[[LHS_COLLAPSED:.+]] = tensor.collapse_shape %[[GATHER]] {{\[}}[0, 1, 2], [3, 4, 5]] : tensor<1x14x14x3x3x4xf32> into tensor<196x36xf32>
//       CHECK:   %[[RHS_COLLAPSED:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0, 1, 2], [3]] : tensor<3x3x4x16xf32> into tensor<36x16xf32>
//       CHECK:   %[[OUT_COLLAPSED:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1, 2], [3]] : tensor<1x14x14x16xf32> into tensor<196x16xf32>
//       CHECK:   %[[MATMUL:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[LHS_COLLAPSED]], %[[RHS_COLLAPSED]] : tensor<196x36xf32>, tensor<36x16xf32>)
//  CHECK-SAME:       outs(%[[OUT_COLLAPSED]] : tensor<196x16xf32>)
//       CHECK:     arith.mulf
//       CHECK:     arith.addf
//       CHECK:   %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1, 2], [3]] output_shape [1, 14, 14, 16]
//       CHECK:   return %[[RESULT]]

// -----

// Channels-first: the output channel separates the batch from the output image
// loops, so the batch stays a separate loop. The K loops follow the filter
// order, which keeps the filter collapsible without a transpose.
func.func @conv_nchw_fchw_batch(%input: tensor<8x4x16x16xf32>, %filter: tensor<16x4x3x3xf32>,
                                %init: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32> {
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<8x4x16x16xf32>, tensor<16x4x3x3xf32>)
      outs(%init : tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32>
  return %0 : tensor<8x16x14x14xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1 + d4, d2 + d5)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @conv_nchw_fchw_batch(
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<8x14x14x4x3x3xf32>)
//   CHECK-DAG:   tensor.collapse_shape %[[GATHER]] {{\[}}[0], [1, 2], [3, 4, 5]] : tensor<8x14x14x4x3x3xf32> into tensor<8x196x36xf32>
//   CHECK-DAG:   tensor.collapse_shape %{{.+}} {{\[}}[0], [1, 2, 3]] : tensor<16x4x3x3xf32> into tensor<16x36xf32>
//   CHECK-DAG:   tensor.collapse_shape %{{.+}} {{\[}}[0], [1], [2, 3]] : tensor<8x16x14x14xf32> into tensor<8x16x196xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//       CHECK:   tensor.expand_shape %{{.+}} {{\[}}[0], [1], [2, 3]] output_shape [8, 16, 14, 14]

// -----

// Strides and dilations only appear in the gather's read map.
func.func @conv_nchw_strided_dilated(%input: tensor<1x4x16x16xf32>, %filter: tensor<16x4x3x3xf32>,
                                     %init: tensor<1x16x6x6xf32>) -> tensor<1x16x6x6xf32> {
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x4x16x16xf32>, tensor<16x4x3x3xf32>)
      outs(%init : tensor<1x16x6x6xf32>) -> tensor<1x16x6x6xf32>
  return %0 : tensor<1x16x6x6xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1 * 2 + d4 * 2, d2 * 2 + d5 * 2)>
// CHECK-LABEL: func.func @conv_nchw_strided_dilated(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x6x6x4x3x3xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<1x36x36xf32>, tensor<16x36xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x16x36xf32>)

// -----

// Depthwise channels are batch loops of the contraction and come first.
func.func @depthwise_conv_nhwc_hwc(%input: tensor<1x114x114x16xf32>, %filter: tensor<3x3x16xf32>,
                                   %init: tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x114x114x16xf32>, tensor<3x3x16xf32>)
      outs(%init : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  return %0 : tensor<1x112x112x16xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 + d4, d3 + d5, d0)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @depthwise_conv_nhwc_hwc(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<16x1x112x112x3x3xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<16x12544x9xf32>, tensor<9x16xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<12544x16xf32>)

// -----

// Groups are batch loops as well.
func.func @grouped_conv_ngchw_gfchw(%input: tensor<1x2x4x10x10xf32>, %filter: tensor<2x8x4x3x3xf32>,
                                    %init: tensor<1x2x8x8x8xf32>) -> tensor<1x2x8x8x8xf32> {
  %0 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x2x4x10x10xf32>, tensor<2x8x4x3x3xf32>)
      outs(%init : tensor<1x2x8x8x8xf32>) -> tensor<1x2x8x8x8xf32>
  return %0 : tensor<1x2x8x8x8xf32>
}
// CHECK-LABEL: func.func @grouped_conv_ngchw_gfchw(
//       CHECK:   linalg.generic
//  CHECK-SAME:       outs(%{{.+}} : tensor<2x1x8x8x4x3x3xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<2x1x64x36xf32>, tensor<2x8x36xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x2x8x64xf32>)

// -----

// An unstrided 1x1 convolution reads its input in place, without a gather.
func.func @pointwise_conv_nchw(%input: tensor<1x32x8x8xf32>, %filter: tensor<16x32x1x1xf32>,
                               %init: tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32> {
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x32x8x8xf32>, tensor<16x32x1x1xf32>)
      outs(%init : tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
  return %0 : tensor<1x16x8x8xf32>
}
// CHECK-LABEL: func.func @pointwise_conv_nchw(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x32x8x8xf32>
//   CHECK-NOT:   tensor.empty
//       CHECK:   %[[LHS:.+]] = tensor.collapse_shape %[[INPUT]] {{\[}}[0], [1], [2, 3]] : tensor<1x32x8x8xf32> into tensor<1x32x64xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[LHS]], %{{.+}} : tensor<1x32x64xf32>, tensor<16x32x1xf32>)

// -----

// A strided 1x1 convolution still needs the gather to subsample its input.
func.func @strided_pointwise_conv_nchw(%input: tensor<1x32x8x8xf32>, %filter: tensor<16x32x1x1xf32>,
                                       %init: tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32> {
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x32x8x8xf32>, tensor<16x32x1x1xf32>)
      outs(%init : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  return %0 : tensor<1x16x4x4xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1 * 2, d2 * 2)>
// CHECK-LABEL: func.func @strided_pointwise_conv_nchw(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x4x4x32xf32>)

// -----

// Generic convolutions are rewritten as well, with their payload unchanged.
#input_map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 + d4, d2 + d5)>
#filter_map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>
#output_map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
func.func @generic_integer_conv_chw(%input: tensor<16x162x162xi8>, %filter: tensor<16x16x3x3xi8>,
                                    %init: tensor<16x160x160xi32>) -> tensor<16x160x160xi32> {
  %0 = linalg.generic {indexing_maps = [#input_map, #filter_map, #output_map],
                       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
      ins(%input, %filter : tensor<16x162x162xi8>, tensor<16x16x3x3xi8>) outs(%init : tensor<16x160x160xi32>) {
  ^bb0(%in: i8, %weight: i8, %acc: i32):
    %in_i32 = arith.extsi %in : i8 to i32
    %weight_i32 = arith.extsi %weight : i8 to i32
    %product = arith.muli %in_i32, %weight_i32 : i32
    %sum = arith.addi %acc, %product : i32
    linalg.yield %sum : i32
  } -> tensor<16x160x160xi32>
  return %0 : tensor<16x160x160xi32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0 + d3, d1 + d4)>
// CHECK-LABEL: func.func @generic_integer_conv_chw(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<160x160x16x3x3xi8>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<25600x144xi8>, tensor<16x144xi8>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<16x25600xi32>)
//       CHECK:   ^bb0(%[[IN:.+]]: i8, %[[WEIGHT:.+]]: i8, %[[ACC:.+]]: i32):
//       CHECK:     %[[IN_I32:.+]] = arith.extsi %[[IN]] : i8 to i32
//       CHECK:     %[[WEIGHT_I32:.+]] = arith.extsi %[[WEIGHT]] : i8 to i32
//       CHECK:     %[[PRODUCT:.+]] = arith.muli %[[IN_I32]], %[[WEIGHT_I32]] : i32
//       CHECK:     arith.addi %[[ACC]], %[[PRODUCT]] : i32

// -----

// Dynamic loop extents size the gather from the operands.
func.func @conv_1d_dynamic(%input: tensor<?x?x4xf32>, %filter: tensor<3x4x8xf32>,
                           %init: tensor<?x?x8xf32>) -> tensor<?x?x8xf32> {
  %0 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
      ins(%input, %filter : tensor<?x?x4xf32>, tensor<3x4x8xf32>)
      outs(%init : tensor<?x?x8xf32>) -> tensor<?x?x8xf32>
  return %0 : tensor<?x?x8xf32>
}
// CHECK-LABEL: func.func @conv_1d_dynamic(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<?x?x4xf32>
//  CHECK-SAME:     %[[FILTER:[a-zA-Z0-9_]+]]: tensor<3x4x8xf32>
//  CHECK-SAME:     %[[INIT:[a-zA-Z0-9_]+]]: tensor<?x?x8xf32>
//   CHECK-DAG:   %[[BATCH:.+]] = tensor.dim %[[INPUT]], %c0
//   CHECK-DAG:   %[[WIDTH:.+]] = tensor.dim %[[INIT]], %c1
//       CHECK:   %[[GATHER_INIT:.+]] = tensor.empty(%[[BATCH]], %[[WIDTH]]) : tensor<?x?x3x4xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:       outs(%[[GATHER_INIT]] : tensor<?x?x3x4xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<?x12xf32>, tensor<12x8xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<?x8xf32>)

// -----

// Pooling ops only use the filter for its shape and are left alone.
func.func @pooling_nhwc_sum(%input: tensor<1x16x16x4xf32>, %window: tensor<3x3xf32>,
                            %init: tensor<1x14x14x4xf32>) -> tensor<1x14x14x4xf32> {
  %0 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %window : tensor<1x16x16x4xf32>, tensor<3x3xf32>)
      outs(%init : tensor<1x14x14x4xf32>) -> tensor<1x14x14x4xf32>
  return %0 : tensor<1x14x14x4xf32>
}
// CHECK-LABEL: func.func @pooling_nhwc_sum(
//       CHECK:   linalg.pooling_nhwc_sum
//   CHECK-NOT:   linalg.generic
