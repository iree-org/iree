// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline %s | FileCheck %s --implicit-check-not=iree_linalg_ext.dequantize_affine --implicit-check-not=arith.truncf
// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline="use-im2col-for-convs=true" %s | FileCheck %s --check-prefixes=CHECK,IM2COL --implicit-check-not=iree_linalg_ext.dequantize_affine --implicit-check-not=arith.truncf
// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline="use-im2col-for-convs=true" --iree-global-opt-demote-contraction-inputs-type=f16 --iree-global-opt-demote-contraction-inputs-operations=all %s | FileCheck %s --check-prefixes=CHECK,IM2COL --implicit-check-not=iree_linalg_ext.dequantize_affine --implicit-check-not=arith.truncf
// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline="use-im2col-for-convs=true" --iree-global-opt-demote-contraction-inputs-type=bf16 --iree-global-opt-demote-contraction-inputs-operations=all %s | FileCheck %s --check-prefixes=CHECK,IM2COL --implicit-check-not=iree_linalg_ext.dequantize_affine --implicit-check-not=arith.truncf
// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline="use-im2col-for-convs=true" --iree-global-opt-enable-qdq-to-integer-math=false %s | FileCheck %s --check-prefix=DISABLED --implicit-check-not=arith.muli

// Biased QDQ contractions must convert after bias detachment and before im2col
// or floating-point demotion hides the dequantize producers. The IM2COL checks
// require an integer gather and a three-loop integer matrix contraction.

// -----

util.func public @biased_conv(
    %aq: tensor<1x3x3x1xi8>,
    %bq: tensor<2x2x1x2xi8>,
    %sa: f32,
    %sb: tensor<2xf32>,
    %za: i8,
    %zb: tensor<2xi8>,
    %bias: tensor<2xf32>) -> tensor<1x2x2x2xf32> {
  %ai = tensor.empty() : tensor<1x3x3x1xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>,
                        affine_map<(n, h, w, c) -> ()>,
                        affine_map<(n, h, w, c) -> ()>,
                        affine_map<(n, h, w, c) -> (n, h, w, c)>]>
      ins(%aq, %sa, %za : tensor<1x3x3x1xi8>, f32, i8)
      outs(%ai : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  %bi = tensor.empty() : tensor<2x2x1x2xf32>
  %b = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(h, w, c, f) -> (h, w, c, f)>,
                        affine_map<(h, w, c, f) -> (f)>,
                        affine_map<(h, w, c, f) -> (f)>,
                        affine_map<(h, w, c, f) -> (h, w, c, f)>]>
      ins(%bq, %sb, %zb : tensor<2x2x1x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%bi : tensor<2x2x1x2xf32>) -> tensor<2x2x1x2xf32>
  %empty = tensor.empty() : tensor<1x2x2x2xf32>
  %init = linalg.broadcast ins(%bias : tensor<2xf32>) outs(%empty : tensor<1x2x2x2xf32>) dimensions = [0, 1, 2]
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%a, %b : tensor<1x3x3x1xf32>, tensor<2x2x1x2xf32>)
      outs(%init : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  util.return %result : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: util.func public @biased_conv(
// IM2COL: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "parallel", "parallel"]{{.*}}ins(%{{.+}} : tensor<3x3xi8>) outs(%{{.+}} : tensor<2x2x2x2xi8>)
// IM2COL: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction"]{{.*}}ins(%{{.+}}, %{{.+}} : tensor<4x4xi8>, tensor<4x2xi8>) outs(%{{.+}} : tensor<4x2xi32>)
// CHECK: arith.muli %{{.+}}, %{{.+}} : i32
// CHECK: arith.sitofp
// CHECK: arith.addf
// DISABLED-LABEL: util.func public @biased_conv(
// DISABLED: iree_linalg_ext.dequantize_affine

// -----

util.func public @biased_nchw(
    %aq: tensor<1x2x3x4xi8>,
    %bq: tensor<2x2x2x2xi8>,
    %sb: tensor<2xf32>,
    %zb: tensor<2xi8>,
    %sa: f32,
    %za: i8,
    %bias: tensor<2xf32>) -> tensor<1x2x2x3xf32> {
  %ai = tensor.empty() : tensor<1x2x3x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>,
                        affine_map<(n, c, h, w) -> ()>,
                        affine_map<(n, c, h, w) -> ()>,
                        affine_map<(n, c, h, w) -> (n, c, h, w)>]>
      ins(%aq, %sa, %za : tensor<1x2x3x4xi8>, f32, i8)
      outs(%ai : tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %bi = tensor.empty() : tensor<2x2x2x2xf32>
  %b = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(f, c, h, w) -> (f, c, h, w)>,
                        affine_map<(f, c, h, w) -> (f)>,
                        affine_map<(f, c, h, w) -> (f)>,
                        affine_map<(f, c, h, w) -> (f, c, h, w)>]>
      ins(%bq, %sb, %zb : tensor<2x2x2x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%bi : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  %empty = tensor.empty() : tensor<1x2x2x3xf32>
  %init = linalg.broadcast ins(%bias : tensor<2xf32>) outs(%empty : tensor<1x2x2x3xf32>) dimensions = [0, 2, 3]
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%a, %b : tensor<1x2x3x4xf32>, tensor<2x2x2x2xf32>)
      outs(%init : tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
  util.return %result : tensor<1x2x2x3xf32>
}
// CHECK-LABEL: util.func public @biased_nchw(
// IM2COL: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]{{.*}}ins(%{{.+}} : tensor<2x3x4xi8>) outs(%{{.+}} : tensor<2x3x2x2x2xi8>)
// IM2COL: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction"]{{.*}}ins(%{{.+}}, %{{.+}} : tensor<6x8xi8>, tensor<2x8xi8>) outs(%{{.+}} : tensor<2x6xi32>)
// CHECK: arith.muli %{{.+}}, %{{.+}} : i32
// CHECK: arith.sitofp
// CHECK: arith.addf
// DISABLED-LABEL: util.func public @biased_nchw(
// DISABLED: iree_linalg_ext.dequantize_affine

// -----

util.func public @biased_matmul(
    %aq: tensor<4x8xi8>, %bq: tensor<3x8xi8>, %sa: f32, %sb: f32,
    %za: i8, %zb: i8, %bias: tensor<3xf32>) -> tensor<4x3xf32> {
  %ai = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]>
      ins(%aq, %sa, %za : tensor<4x8xi8>, f32, i8)
      outs(%ai : tensor<4x8xf32>) -> tensor<4x8xf32>
  %bi = tensor.empty() : tensor<3x8xf32>
  %b = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, k) -> (n, k)>,
                        affine_map<(n, k) -> ()>,
                        affine_map<(n, k) -> ()>,
                        affine_map<(n, k) -> (n, k)>]>
      ins(%bq, %sb, %zb : tensor<3x8xi8>, f32, i8)
      outs(%bi : tensor<3x8xf32>) -> tensor<3x8xf32>
  %ti = tensor.empty() : tensor<8x3xf32>
  %bt = linalg.transpose ins(%b : tensor<3x8xf32>)
      outs(%ti : tensor<8x3xf32>) permutation = [1, 0]
  %empty = tensor.empty() : tensor<4x3xf32>
  %init = linalg.broadcast ins(%bias : tensor<3xf32>) outs(%empty : tensor<4x3xf32>) dimensions = [0]
  %result = linalg.matmul ins(%a, %bt : tensor<4x8xf32>, tensor<8x3xf32>)
      outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  util.return %result : tensor<4x3xf32>
}
// CHECK-LABEL: util.func public @biased_matmul(
// CHECK: arith.muli %{{.+}}, %{{.+}} : i32
// CHECK: arith.sitofp
// CHECK: arith.addf
// DISABLED-LABEL: util.func public @biased_matmul(
// DISABLED: iree_linalg_ext.dequantize_affine
