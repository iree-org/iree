// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline --iree-global-opt-enable-qdq-to-integer-math=true %s | FileCheck %s --implicit-check-not=iree_linalg_ext.dequantize_affine
// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline --iree-global-opt-enable-qdq-to-integer-math=false %s | FileCheck %s --check-prefix=DISABLED
// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline="use-im2col-for-convs=true" --iree-global-opt-enable-qdq-to-integer-math=true %s | FileCheck %s --implicit-check-not=iree_linalg_ext.dequantize_affine

// Exercise the production pipeline, including unit-dimension folding and
// transpose propagation, rather than invoking the QDQ pass in isolation.

// -----

// Unit batch dimensions must not hide dequantize producers before the rewrite.
util.func public @convolution_zero_point_corrections(
    %aq: tensor<1x5x5x1xi8>,
    %bq: tensor<2x2x1x2xi8>,
    %sa: f32,
    %sb: tensor<2xf32>,
    %za: i8,
    %zb: tensor<2xi8>) -> tensor<1x2x2x2xf32> {
  %ai = tensor.empty() : tensor<1x5x5x1xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>,
                        affine_map<(n, h, w, c) -> ()>,
                        affine_map<(n, h, w, c) -> ()>,
                        affine_map<(n, h, w, c) -> (n, h, w, c)>]}
      ins(%aq, %sa, %za : tensor<1x5x5x1xi8>, f32, i8)
      outs(%ai : tensor<1x5x5x1xf32>) -> tensor<1x5x5x1xf32>
  %bi = tensor.empty() : tensor<2x2x1x2xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(h, w, c, f) -> (h, w, c, f)>,
                        affine_map<(h, w, c, f) -> (f)>,
                        affine_map<(h, w, c, f) -> (f)>,
                        affine_map<(h, w, c, f) -> (h, w, c, f)>]}
      ins(%bq, %sb, %zb : tensor<2x2x1x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%bi : tensor<2x2x1x2xf32>) -> tensor<2x2x1x2xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x2x2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%a, %b : tensor<1x5x5x1xf32>, tensor<2x2x1x2xf32>)
      outs(%init : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  util.return %result : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: util.func public @convolution_zero_point_corrections(
// CHECK: arith.muli %{{.+}}, %{{.+}} : i32
// CHECK: arith.sitofp
// DISABLED-LABEL: util.func public @convolution_zero_point_corrections(
// DISABLED: iree_linalg_ext.dequantize_affine
// DISABLED: arith.mulf %{{.+}}, %{{.+}} : f32

// -----

// Unit batch dimensions must not hide dequantize producers before the rewrite.
util.func public @convolution_nchw_asymmetric(
    %aq: tensor<1x2x3x4xi8>,
    %bq: tensor<2x2x2x2xi8>,
    %sb: tensor<2xf32>,
    %zb: tensor<2xi8>,
    %sa: f32,
    %za: i8) -> tensor<1x2x2x3xf32> {
  %ai = tensor.empty() : tensor<1x2x3x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>,
                        affine_map<(n, c, h, w) -> ()>,
                        affine_map<(n, c, h, w) -> ()>,
                        affine_map<(n, c, h, w) -> (n, c, h, w)>]}
      ins(%aq, %sa, %za : tensor<1x2x3x4xi8>, f32, i8)
      outs(%ai : tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %bi = tensor.empty() : tensor<2x2x2x2xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(f, c, h, w) -> (f, c, h, w)>,
                        affine_map<(f, c, h, w) -> (f)>,
                        affine_map<(f, c, h, w) -> (f)>,
                        affine_map<(f, c, h, w) -> (f, c, h, w)>]}
      ins(%bq, %sb, %zb : tensor<2x2x2x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%bi : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x2x3xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%a, %b : tensor<1x2x3x4xf32>, tensor<2x2x2x2xf32>)
      outs(%init : tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
  util.return %result : tensor<1x2x2x3xf32>
}
// CHECK-LABEL: util.func public @convolution_nchw_asymmetric(
// CHECK: arith.muli %{{.+}}, %{{.+}} : i32
// CHECK: arith.sitofp
// DISABLED-LABEL: util.func public @convolution_nchw_asymmetric(
// DISABLED: iree_linalg_ext.dequantize_affine
// DISABLED: arith.mulf %{{.+}}, %{{.+}} : f32

// -----

// Unit batch dimensions must not hide dequantize producers before the rewrite.
util.func public @depthwise_asymmetric_strided_dilated(
    %aq: tensor<1x4x5x2xi8>,
    %bq: tensor<2x2x2xi8>,
    %sa: tensor<2xf32>,
    %za: tensor<2xi8>,
    %sb: tensor<2xf32>,
    %zb: tensor<2xi8>) -> tensor<1x2x2x2xf32> {
  %ai = tensor.empty() : tensor<1x4x5x2xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned, indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>,
                        affine_map<(n, h, w, c) -> (c)>,
                        affine_map<(n, h, w, c) -> (c)>,
                        affine_map<(n, h, w, c) -> (n, h, w, c)>]}
      ins(%aq, %sa, %za : tensor<1x4x5x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%ai : tensor<1x4x5x2xf32>) -> tensor<1x4x5x2xf32>
  %bi = tensor.empty() : tensor<2x2x2xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(h, w, c) -> (h, w, c)>,
                        affine_map<(h, w, c) -> (c)>,
                        affine_map<(h, w, c) -> (c)>,
                        affine_map<(h, w, c) -> (h, w, c)>]}
      ins(%bq, %sb, %zb : tensor<2x2x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%bi : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x2x2x2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %result = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>}
      ins(%a, %b : tensor<1x4x5x2xf32>, tensor<2x2x2xf32>)
      outs(%init : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  util.return %result : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: util.func public @depthwise_asymmetric_strided_dilated(
// CHECK: arith.muli %{{.+}}, %{{.+}} : i32
// CHECK: arith.sitofp
// DISABLED-LABEL: util.func public @depthwise_asymmetric_strided_dilated(
// DISABLED: iree_linalg_ext.dequantize_affine
// DISABLED: arith.mulf %{{.+}}, %{{.+}} : f32

// -----

// The early rewrite cannot see through the RHS transpose. Keep a later
// opportunity after transpose propagation exposes the dequantize producer.
util.func public @transposed_dequantized_rhs(
    %aq: tensor<4x8xi8>, %bq: tensor<3x8xi8>, %sa: f32, %sb: f32,
    %za: i8, %zb: i8) -> tensor<4x3xf32> {
  %ai = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %sa, %za : tensor<4x8xi8>, f32, i8)
      outs(%ai : tensor<4x8xf32>) -> tensor<4x8xf32>
  %bi = tensor.empty() : tensor<3x8xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(n, k) -> (n, k)>,
                        affine_map<(n, k) -> ()>,
                        affine_map<(n, k) -> ()>,
                        affine_map<(n, k) -> (n, k)>]}
      ins(%bq, %sb, %zb : tensor<3x8xi8>, f32, i8)
      outs(%bi : tensor<3x8xf32>) -> tensor<3x8xf32>
  %ti = tensor.empty() : tensor<8x3xf32>
  %bt = linalg.transpose ins(%b : tensor<3x8xf32>)
      outs(%ti : tensor<8x3xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x3xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x3xf32>) -> tensor<4x3xf32>
  %result = linalg.matmul ins(%a, %bt : tensor<4x8xf32>, tensor<8x3xf32>)
      outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  util.return %result : tensor<4x3xf32>
}
// CHECK-LABEL: util.func public @transposed_dequantized_rhs(
// CHECK: arith.muli %{{.+}}, %{{.+}} : i32
// CHECK: arith.sitofp
// DISABLED-LABEL: util.func public @transposed_dequantized_rhs(
// DISABLED: iree_linalg_ext.dequantize_affine
// DISABLED: linalg.matmul
// DISABLED-SAME: outs(%{{.+}} : tensor<4x3xf32>)
