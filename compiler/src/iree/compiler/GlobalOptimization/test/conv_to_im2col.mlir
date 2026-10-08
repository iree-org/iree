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

// With an output-channel-first filter, the filter becomes the transposed
// matmul operand.
func.func @conv_nhwc_fhwc(%input: tensor<1x16x16x4xf32>, %filter: tensor<16x3x3x4xf32>,
                          %init: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x16x16x4xf32>, tensor<16x3x3x4xf32>)
      outs(%init : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  return %0 : tensor<1x14x14x16xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @conv_nhwc_fhwc(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x14x14x3x3x4xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<196x36xf32>, tensor<16x36xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<196x16xf32>)

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
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @conv_nchw_strided_dilated(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x6x6x4x3x3xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<1x36x36xf32>, tensor<16x36xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x16x36xf32>)

// -----

// An unstrided 1x1 convolution reads its input in place, without a gather.
func.func @pointwise_conv_nchw(%input: tensor<1x32x8x8xf32>, %filter: tensor<16x32x1x1xf32>,
                               %init: tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32> {
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x32x8x8xf32>, tensor<16x32x1x1xf32>)
      outs(%init : tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
  return %0 : tensor<1x16x8x8xf32>
}
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @pointwise_conv_nchw(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x32x8x8xf32>
//   CHECK-NOT:   tensor.empty
//       CHECK:   %[[LHS:.+]] = tensor.collapse_shape %[[INPUT]] {{\[}}[0], [1], [2, 3]] : tensor<1x32x8x8xf32> into tensor<1x32x64xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
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
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @strided_pointwise_conv_nchw(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x4x4x32xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<1x16x32xf32>, tensor<16x32x1xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x16x16xf32>)

// -----

// Groups are batch loops of the contraction and come first.
func.func @grouped_conv_ngchw_gfchw(%input: tensor<1x2x4x10x10xf32>, %filter: tensor<2x8x4x3x3xf32>,
                                    %init: tensor<1x2x8x8x8xf32>) -> tensor<1x2x8x8x8xf32> {
  %0 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x2x4x10x10xf32>, tensor<2x8x4x3x3xf32>)
      outs(%init : tensor<1x2x8x8x8xf32>) -> tensor<1x2x8x8x8xf32>
  return %0 : tensor<1x2x8x8x8xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d0, d4, d2 + d5, d3 + d6)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0, d3, d4)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d4)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @grouped_conv_ngchw_gfchw(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<2x1x8x8x4x3x3xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<2x1x64x36xf32>, tensor<2x8x36xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x2x8x64xf32>)

// -----

// Channels-last groups also become the leading batch loop of the gather.
func.func @grouped_conv_nhwgc_gfhwc(%input: tensor<1x10x10x2x4xf32>, %filter: tensor<2x8x3x3x4xf32>,
                                    %init: tensor<1x8x8x2x8xf32>) -> tensor<1x8x8x2x8xf32> {
  %0 = linalg.conv_2d_nhwgc_gfhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x10x10x2x4xf32>, tensor<2x8x3x3x4xf32>)
      outs(%init : tensor<1x8x8x2x8xf32>) -> tensor<1x8x8x2x8xf32>
  return %0 : tensor<1x8x8x2x8xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2 + d4, d3 + d5, d0, d6)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @grouped_conv_nhwgc_gfhwc(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<2x1x8x8x3x3x4xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<2x64x36xf32>, tensor<2x8x36xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<64x2x8xf32>)

// -----

// A channel multiplier is an output channel that reuses the gathered input, so
// depthwise convolutions with a multiplier are rewritten with the channel as a
// batch loop.
func.func @depthwise_conv_multiplier(%input: tensor<1x10x10x4xf32>, %filter: tensor<3x3x4x2xf32>,
                                     %init: tensor<1x8x8x4x2xf32>) -> tensor<1x8x8x4x2xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x10x10x4xf32>, tensor<3x3x4x2xf32>)
      outs(%init : tensor<1x8x8x4x2xf32>) -> tensor<1x8x8x4x2xf32>
  return %0 : tensor<1x8x8x4x2xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 + d4, d3 + d5, d0)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @depthwise_conv_multiplier(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<4x1x8x8x3x3xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<4x64x9xf32>, tensor<9x4x2xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<64x4x2xf32>)

// -----

// The gather copies the narrow input type; the extensions stay in the
// contraction payload.
func.func @conv_mixed_precision(%input: tensor<1x4x10x10xbf16>, %filter: tensor<16x4x3x3xbf16>,
                                %init: tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32> {
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x4x10x10xbf16>, tensor<16x4x3x3xbf16>)
      outs(%init : tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
  return %0 : tensor<1x16x8x8xf32>
}
// CHECK-LABEL: func.func @conv_mixed_precision(
//       CHECK:   %[[GATHER_INIT:.+]] = tensor.empty() : tensor<1x8x8x4x3x3xbf16>
//       CHECK:   linalg.generic
//  CHECK-SAME:       outs(%[[GATHER_INIT]] : tensor<1x8x8x4x3x3xbf16>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<1x64x36xbf16>, tensor<16x36xbf16>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x16x64xf32>)
//       CHECK:   ^bb0(%[[IN:.+]]: bf16, %[[WEIGHT:.+]]: bf16, %[[ACC:.+]]: f32):
//       CHECK:     %[[IN_F32:.+]] = arith.extf %[[IN]] : bf16 to f32
//       CHECK:     %[[WEIGHT_F32:.+]] = arith.extf %[[WEIGHT]] : bf16 to f32
//       CHECK:     %[[PRODUCT:.+]] = arith.mulf %[[IN_F32]], %[[WEIGHT_F32]] : f32
//       CHECK:     arith.addf %[[ACC]], %[[PRODUCT]] : f32

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
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @generic_integer_conv_chw(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<160x160x16x3x3xi8>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<25600x144xi8>, tensor<16x144xi8>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<16x25600xi32>)
//       CHECK:   ^bb0(%[[IN:.+]]: i8, %[[WEIGHT:.+]]: i8, %[[ACC:.+]]: i32):
//       CHECK:     %[[IN_I32:.+]] = arith.extsi %[[IN]] : i8 to i32
//       CHECK:     %[[WEIGHT_I32:.+]] = arith.extsi %[[WEIGHT]] : i8 to i32
//       CHECK:     %[[PRODUCT:.+]] = arith.muli %[[IN_I32]], %[[WEIGHT_I32]] : i32
//       CHECK:     arith.addi %[[ACC]], %[[PRODUCT]] : i32

// -----

func.func @conv_1d_ncw_fcw(%input: tensor<2x4x18xf32>, %filter: tensor<8x4x3xf32>,
                           %init: tensor<2x8x8xf32>) -> tensor<2x8x8xf32> {
  %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
      ins(%input, %filter : tensor<2x4x18xf32>, tensor<8x4x3xf32>)
      outs(%init : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  return %0 : tensor<2x8x8xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1 * 2 + d3)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @conv_1d_ncw_fcw(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x4x18xf32>
//  CHECK-SAME:     %[[FILTER:[a-zA-Z0-9_]+]]: tensor<8x4x3xf32>
//  CHECK-SAME:     %[[INIT:[a-zA-Z0-9_]+]]: tensor<2x8x8xf32>
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<2x4x18xf32>) outs(%{{.+}} : tensor<2x8x4x3xf32>)
//       CHECK:   %[[LHS_COLLAPSED:.+]] = tensor.collapse_shape %[[GATHER]] {{\[}}[0], [1], [2, 3]] : tensor<2x8x4x3xf32> into tensor<2x8x12xf32>
//       CHECK:   %[[RHS_COLLAPSED:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0], [1, 2]] : tensor<8x4x3xf32> into tensor<8x12xf32>
//       CHECK:   %[[RESULT:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[LHS_COLLAPSED]], %[[RHS_COLLAPSED]] : tensor<2x8x12xf32>, tensor<8x12xf32>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<2x8x8xf32>)
//       CHECK:   return %[[RESULT]]

// -----

// Dynamic loop extents size the gather from the operands.
func.func @conv_1d_dynamic(%input: tensor<?x?x4xf32>, %filter: tensor<3x4x8xf32>,
                           %init: tensor<?x?x8xf32>) -> tensor<?x?x8xf32> {
  %0 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
      ins(%input, %filter : tensor<?x?x4xf32>, tensor<3x4x8xf32>)
      outs(%init : tensor<?x?x8xf32>) -> tensor<?x?x8xf32>
  return %0 : tensor<?x?x8xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 + d2, d3)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @conv_1d_dynamic(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<?x?x4xf32>
//  CHECK-SAME:     %[[FILTER:[a-zA-Z0-9_]+]]: tensor<3x4x8xf32>
//  CHECK-SAME:     %[[INIT:[a-zA-Z0-9_]+]]: tensor<?x?x8xf32>
//   CHECK-DAG:   %[[BATCH:.+]] = tensor.dim %[[INPUT]], %c0
//   CHECK-DAG:   %[[WIDTH:.+]] = tensor.dim %[[INIT]], %c1
//       CHECK:   %[[GATHER_INIT:.+]] = tensor.empty(%[[BATCH]], %[[WIDTH]]) : tensor<?x?x3x4xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%[[GATHER_INIT]] : tensor<?x?x3x4xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<?x12xf32>, tensor<12x8xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<?x8xf32>)

// -----

// A channels-last 3x3x3 convolution gathers all three window loops and becomes
// a plain matmul.
func.func @conv_3d_ndhwc_dhwcf(%input: tensor<1x6x10x10x4xf32>, %filter: tensor<3x3x3x4x16xf32>,
                               %init: tensor<1x4x8x8x16xf32>) -> tensor<1x4x8x8x16xf32> {
  %0 = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
      ins(%input, %filter : tensor<1x6x10x10x4xf32>, tensor<3x3x3x4x16xf32>)
      outs(%init : tensor<1x4x8x8x16xf32>) -> tensor<1x4x8x8x16xf32>
  return %0 : tensor<1x4x8x8x16xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d4, d2 + d5, d3 + d6, d7)>
// CHECK-DAG:  #[[$ID8:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @conv_3d_ndhwc_dhwcf(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x6x10x10x4xf32>
//  CHECK-SAME:     %[[FILTER:[a-zA-Z0-9_]+]]: tensor<3x3x3x4x16xf32>
//  CHECK-SAME:     %[[INIT:[a-zA-Z0-9_]+]]: tensor<1x4x8x8x16xf32>
//       CHECK:   %[[GATHER_INIT:.+]] = tensor.empty() : tensor<1x4x8x8x3x3x3x4xf32>
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]], #[[$ID8]]]
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<1x6x10x10x4xf32>) outs(%[[GATHER_INIT]] : tensor<1x4x8x8x3x3x3x4xf32>)
//       CHECK:   %[[LHS_COLLAPSED:.+]] = tensor.collapse_shape %[[GATHER]] {{\[}}[0, 1, 2, 3], [4, 5, 6, 7]] : tensor<1x4x8x8x3x3x3x4xf32> into tensor<256x108xf32>
//       CHECK:   %[[RHS_COLLAPSED:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0, 1, 2, 3], [4]] : tensor<3x3x3x4x16xf32> into tensor<108x16xf32>
//       CHECK:   %[[OUT_COLLAPSED:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1, 2, 3], [4]] : tensor<1x4x8x8x16xf32> into tensor<256x16xf32>
//       CHECK:   %[[MATMUL:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[LHS_COLLAPSED]], %[[RHS_COLLAPSED]] : tensor<256x108xf32>, tensor<108x16xf32>)
//  CHECK-SAME:       outs(%[[OUT_COLLAPSED]] : tensor<256x16xf32>)
//       CHECK:   %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1, 2, 3], [4]] output_shape [1, 4, 8, 8, 16]
//       CHECK:   return %[[RESULT]]

// -----

// Channels-first 3x3x3: the K loops follow the filter order, so the filter
// collapses without a transpose and the batch stays a separate loop.
func.func @conv_3d_ncdhw_fcdhw(%input: tensor<1x4x6x10x10xf32>, %filter: tensor<16x4x3x3x3xf32>,
                               %init: tensor<1x16x4x8x8xf32>) -> tensor<1x16x4x8x8xf32> {
  %0 = linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
      ins(%input, %filter : tensor<1x4x6x10x10xf32>, tensor<16x4x3x3x3xf32>)
      outs(%init : tensor<1x16x4x8x8xf32>) -> tensor<1x16x4x8x8xf32>
  return %0 : tensor<1x16x4x8x8xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d4, d1 + d5, d2 + d6, d3 + d7)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @conv_3d_ncdhw_fcdhw(
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x4x8x8x4x3x3x3xf32>)
//   CHECK-DAG:   tensor.collapse_shape %[[GATHER]] {{\[}}[0], [1, 2, 3], [4, 5, 6, 7]] : tensor<1x4x8x8x4x3x3x3xf32> into tensor<1x256x108xf32>
//   CHECK-DAG:   tensor.collapse_shape %{{.+}} {{\[}}[0], [1, 2, 3, 4]] : tensor<16x4x3x3x3xf32> into tensor<16x108xf32>
//   CHECK-DAG:   tensor.collapse_shape %{{.+}} {{\[}}[0], [1], [2, 3, 4]] : tensor<1x16x4x8x8xf32> into tensor<1x16x256xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//       CHECK:   tensor.expand_shape %{{.+}} {{\[}}[0], [1], [2, 3, 4]] output_shape [1, 16, 4, 8, 8]

// -----

// Only the depth window loop has a non-unit extent, so it is the only window
// loop gathered. The unit window loops remain as a unit reduction loop of the
// contraction.
func.func @conv_3d_unit_spatial_window(%input: tensor<1x4x6x10x10xf32>, %filter: tensor<16x4x3x1x1xf32>,
                                       %init: tensor<1x16x4x10x10xf32>) -> tensor<1x16x4x10x10xf32> {
  %0 = linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
      ins(%input, %filter : tensor<1x4x6x10x10xf32>, tensor<16x4x3x1x1xf32>)
      outs(%init : tensor<1x16x4x10x10xf32>) -> tensor<1x16x4x10x10xf32>
  return %0 : tensor<1x16x4x10x10xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1 + d5, d2, d3)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK-LABEL: func.func @conv_3d_unit_spatial_window(
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x4x10x10x4x3xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<1x400x12xf32>, tensor<16x12x1xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<1x16x400xf32>)

// -----

// A depthwise convolution has no output channel to reuse the gathered input
// across, so it is left alone.
func.func @depthwise_conv_nhwc_hwc(%input: tensor<1x114x114x16xf32>, %filter: tensor<3x3x16xf32>,
                                   %init: tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x114x114x16xf32>, tensor<3x3x16xf32>)
      outs(%init : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  return %0 : tensor<1x112x112x16xf32>
}
// CHECK-LABEL: func.func @depthwise_conv_nhwc_hwc(
//       CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
//   CHECK-NOT:   linalg.generic

// -----

// Pooling ops have no output channel either and are left alone.
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

// -----

// Quantized convolutions carry zero-point operands and are expected to be
// lowered before im2col runs.
func.func @quantized_conv_nhwc_hwcf(%input: tensor<1x10x10x4xi8>, %filter: tensor<3x3x4x16xi8>,
                                    %input_zp: i32, %filter_zp: i32,
                                    %init: tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32> {
  %0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter, %input_zp, %filter_zp : tensor<1x10x10x4xi8>, tensor<3x3x4x16xi8>, i32, i32)
      outs(%init : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
  return %0 : tensor<1x8x8x16xi32>
}
// CHECK-LABEL: func.func @quantized_conv_nhwc_hwcf(
//       CHECK:   linalg.conv_2d_nhwc_hwcf_q
//   CHECK-NOT:   linalg.generic
