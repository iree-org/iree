// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline="use-im2col-for-convs=true" %s | FileCheck %s

// Exercise im2col at its position in the production pipeline, after named
// convolutions are generalized and unit dimensions are folded. A filter window
// of extent one along a strided axis is folded away there, which leaves an
// input access such as `(c, oh * 2, ow * 2)` that is no longer a convolution.
// With im2col enabled, no contraction may keep such an access: each one reads
// a gathered input through a projected permutation and has one loop per
// dimension kind.

// -----

util.func public @strided_pointwise_conv(
    %input: tensor<1x32x8x8xf32>,
    %filter: tensor<16x32x1x1xf32>) -> tensor<1x16x4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x16x4x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x32x8x8xf32>, tensor<16x32x1x1xf32>)
      outs(%init : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  util.return %result : tensor<1x16x4x4xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2) -> (d2, d0 * 2, d1 * 2)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: util.func public @strided_pointwise_conv(
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       outs(%{{.+}} : tensor<4x4x32xf32>)
//       CHECK:   %[[LHS:.+]] = tensor.collapse_shape %[[GATHER]] {{\[}}[0, 1], [2]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[LHS]], %{{.+}} : tensor<16x32xf32>, tensor<16x32xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<16x16xf32>)
//   CHECK-NOT:   tensor.extract_slice

// -----

// Only the height axis has a unit window. The remaining window access along
// the width is gathered together with the bare strided height loop.
util.func public @strided_conv_unit_window_height(
    %input: tensor<1x32x8x9xf32>,
    %filter: tensor<16x32x1x3xf32>) -> tensor<1x16x4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x16x4x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x32x8x9xf32>, tensor<16x32x1x3xf32>)
      outs(%init : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  util.return %result : tensor<1x16x4x4xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0 * 2, d1 * 2 + d3)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: util.func public @strided_conv_unit_window_height(
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<4x4x32x3xf32>)
//       CHECK:   %[[LHS:.+]] = tensor.collapse_shape %[[GATHER]] {{\[}}[0, 1], [2, 3]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[LHS]], %{{.+}} : tensor<16x96xf32>, tensor<16x96xf32>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<16x16xf32>)
//   CHECK-NOT:   tensor.extract_slice

// -----

// The QDQ form converts to integer math first; the integer contraction is then
// gathered like the floating-point one.
util.func public @strided_pointwise_conv_qdq(
    %aq: tensor<1x32x8x8xi8>,
    %bq: tensor<16x32x1x1xi8>,
    %sb: tensor<16xf32>,
    %sa: f32,
    %za: i8) -> tensor<1x16x4x4xf32> {
  %zb = arith.constant dense<0> : tensor<16xi8>
  %ai = tensor.empty() : tensor<1x32x8x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>,
                        affine_map<(n, c, h, w) -> ()>,
                        affine_map<(n, c, h, w) -> ()>,
                        affine_map<(n, c, h, w) -> (n, c, h, w)>]>
      ins(%aq, %sa, %za : tensor<1x32x8x8xi8>, f32, i8)
      outs(%ai : tensor<1x32x8x8xf32>) -> tensor<1x32x8x8xf32>
  %bi = tensor.empty() : tensor<16x32x1x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(f, c, h, w) -> (f, c, h, w)>,
                        affine_map<(f, c, h, w) -> (f)>,
                        affine_map<(f, c, h, w) -> (f)>,
                        affine_map<(f, c, h, w) -> (f, c, h, w)>]>
      ins(%bq, %sb, %zb : tensor<16x32x1x1xi8>, tensor<16xf32>, tensor<16xi8>)
      outs(%bi : tensor<16x32x1x1xf32>) -> tensor<16x32x1x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x16x4x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%a, %b : tensor<1x32x8x8xf32>, tensor<16x32x1x1xf32>)
      outs(%init : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  util.return %result : tensor<1x16x4x4xf32>
}
// CHECK-DAG:  #[[$GATHER_READ:.+]] = affine_map<(d0, d1, d2) -> (d2, d0 * 2, d1 * 2)>
// CHECK-DAG:  #[[$LHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[$RHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[$OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: util.func public @strided_pointwise_conv_qdq(
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$GATHER_READ]],
//  CHECK-SAME:       outs(%{{.+}} : tensor<4x4x32xi8>)
//       CHECK:   %[[LHS:.+]] = tensor.collapse_shape %[[GATHER]] {{\[}}[0, 1], [2]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[LHS]], %{{.+}} : tensor<16x32xi8>, tensor<16x32xi8>)
//  CHECK-SAME:       outs(%{{.+}} : tensor<16x16xi32>)
//       CHECK:     arith.muli %{{.+}}, %{{.+}} : i32
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   arith.sitofp
