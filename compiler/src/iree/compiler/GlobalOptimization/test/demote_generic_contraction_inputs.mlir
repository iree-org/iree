// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-opt --split-input-file --iree-global-opt-demote-contraction-inputs="type=f16 operation=matmul" %s | FileCheck %s --check-prefixes=MATMUL,NOCONV,NEGATIVE -D'$TYPE=f16'
// RUN: iree-opt --split-input-file --iree-global-opt-demote-contraction-inputs="type=bf16 operation=matmul" %s | FileCheck %s --check-prefixes=MATMUL,NOCONV,NEGATIVE -D'$TYPE=bf16'
// RUN: iree-opt --split-input-file --iree-global-opt-demote-contraction-inputs="type=f16 operation=conv" %s | FileCheck %s --check-prefixes=NOMATMUL,CONV,NEGATIVE -D'$TYPE=f16'
// RUN: iree-opt --split-input-file --iree-global-opt-demote-contraction-inputs="type=bf16 operation=conv" %s | FileCheck %s --check-prefixes=NOMATMUL,CONV,NEGATIVE -D'$TYPE=bf16'
// RUN: iree-opt --split-input-file --iree-global-opt-demote-contraction-inputs="type=f16 operation=all" %s | FileCheck %s --check-prefixes=MATMUL,CONV,NEGATIVE -D'$TYPE=f16'
// RUN: iree-opt --split-input-file --iree-global-opt-demote-contraction-inputs="type=bf16 operation=all" %s | FileCheck %s --check-prefixes=MATMUL,CONV,NEGATIVE -D'$TYPE=bf16'
// RUN: iree-opt --split-input-file --iree-global-opt-demote-contraction-inputs="type=f16 operation=none" %s | FileCheck %s --check-prefix=NONE --implicit-check-not=arith.truncf

// Transposed output maps and payload attributes must survive demotion. Dynamic
// shapes also exercise allocation of the demoted input tensors.
util.func public @generic_matmul(%a: tensor<?x8xf32>, %b: tensor<8x3xf32>, %init: tensor<3x?xf32>) -> tensor<3x?xf32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (n, m)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<?x8xf32>, tensor<8x3xf32>)
      outs(%init : tensor<3x?xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %prod = arith.mulf %lhs, %rhs fastmath<contract> : f32
      %sum = arith.addf %acc, %prod fastmath<contract> : f32
      linalg.yield %sum : f32
  } -> tensor<3x?xf32>
  util.return %result : tensor<3x?xf32>
}
// MATMUL-DAG: #[[$OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// MATMUL-LABEL: @generic_matmul(
// MATMUL: tensor.dim
// MATMUL: tensor.empty(%{{.*}})
// MATMUL: arith.truncf {{.*}} : f32 to [[$TYPE]]
// MATMUL: arith.truncf {{.*}} : f32 to [[$TYPE]]
// MATMUL: linalg.generic
// MATMUL-SAME: indexing_maps = [{{.*}}, #[[$OUTPUT_MAP]]]
// MATMUL-SAME: ins(%{{.*}}, %{{.*}} : tensor<?x8x[[$TYPE]]>, tensor<8x3x[[$TYPE]]>)
// MATMUL-SAME: outs(%{{.*}} : tensor<3x?xf32>)
// MATMUL: arith.extf
// MATMUL: arith.extf
// MATMUL: arith.mulf {{.*}} fastmath<contract> : f32
// MATMUL: arith.addf {{.*}} fastmath<contract> : f32
// MATMUL: util.return
// NOMATMUL-LABEL: @generic_matmul(
// NOMATMUL-NOT: arith.truncf
// NOMATMUL: util.return
// NONE-LABEL: @generic_matmul(
// NONE: util.return

// -----

util.func public @generic_conv(%a: tensor<1x8x2xf32>, %b: tensor<3x2x4xf32>, %init: tensor<1x6x4xf32>) -> tensor<1x6x4xf32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(n, w, f, kw, c) -> (n, w + kw, c)>,
                       affine_map<(n, w, f, kw, c) -> (kw, c, f)>,
                       affine_map<(n, w, f, kw, c) -> (n, w, f)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%a, %b : tensor<1x8x2xf32>, tensor<3x2x4xf32>)
      outs(%init : tensor<1x6x4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %prod = arith.mulf %rhs, %lhs : f32
      %sum = arith.addf %prod, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<1x6x4xf32>
  util.return %result : tensor<1x6x4xf32>
}
// CONV-LABEL: @generic_conv(
// CONV: arith.truncf {{.*}} : f32 to [[$TYPE]]
// CONV: arith.truncf {{.*}} : f32 to [[$TYPE]]
// CONV: linalg.generic
// CONV-SAME: ins(%{{.*}}, %{{.*}} : tensor<1x8x2x[[$TYPE]]>, tensor<3x2x4x[[$TYPE]]>)
// CONV-SAME: outs(%{{.*}} : tensor<1x6x4xf32>)
// CONV: arith.extf
// CONV: arith.extf
// CONV: arith.mulf {{.*}} : f32
// CONV: arith.addf {{.*}} : f32
// CONV: util.return
// NOCONV-LABEL: @generic_conv(
// NOCONV-NOT: arith.truncf
// NOCONV: util.return
// NONE-LABEL: @generic_conv(
// NONE: util.return

// -----

// Convolution-shaped indexing alone is not enough: this body is not a
// multiply-accumulate.
util.func public @non_contraction(%a: tensor<1x8x2xf32>, %b: tensor<3x2x4xf32>, %init: tensor<1x6x4xf32>) -> tensor<1x6x4xf32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(n, w, f, kw, c) -> (n, w + kw, c)>,
                       affine_map<(n, w, f, kw, c) -> (kw, c, f)>,
                       affine_map<(n, w, f, kw, c) -> (n, w, f)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%a, %b : tensor<1x8x2xf32>, tensor<3x2x4xf32>)
      outs(%init : tensor<1x6x4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %prod = arith.mulf %lhs, %rhs : f32
      %max = arith.maximumf %prod, %acc : f32
      linalg.yield %max : f32
  } -> tensor<1x6x4xf32>
  util.return %result : tensor<1x6x4xf32>
}
// NEGATIVE-LABEL: @non_contraction(
// NEGATIVE-NOT: arith.truncf
// NEGATIVE: arith.maximumf
// NEGATIVE: util.return
// NONE-LABEL: @non_contraction(
// NONE: util.return

// -----

// The shared contraction-body matcher accepts intervening unary operations,
// but convolution input demotion requires a plain multiply-accumulate body.
util.func public @conv_with_unary_op(%a: tensor<1x8x2xf32>, %b: tensor<3x2x4xf32>, %init: tensor<1x6x4xf32>) -> tensor<1x6x4xf32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(n, w, f, kw, c) -> (n, w + kw, c)>,
                       affine_map<(n, w, f, kw, c) -> (kw, c, f)>,
                       affine_map<(n, w, f, kw, c) -> (n, w, f)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%a, %b : tensor<1x8x2xf32>, tensor<3x2x4xf32>)
      outs(%init : tensor<1x6x4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %neg = arith.negf %lhs : f32
      %prod = arith.mulf %neg, %rhs : f32
      %sum = arith.addf %prod, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<1x6x4xf32>
  util.return %result : tensor<1x6x4xf32>
}
// NEGATIVE-LABEL: @conv_with_unary_op(
// NEGATIVE-NOT: arith.truncf
// NEGATIVE: arith.negf
// NEGATIVE: arith.mulf
// NEGATIVE: arith.addf
// NEGATIVE: util.return
// NONE-LABEL: @conv_with_unary_op(
// NONE: util.return

// -----

util.func public @integer_contraction(%a: tensor<4x8xi8>, %b: tensor<8x3xi8>, %init: tensor<4x3xi32>) -> tensor<4x3xi32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<4x8xi8>, tensor<8x3xi8>)
      outs(%init : tensor<4x3xi32>) {
    ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
      %a32 = arith.extsi %lhs : i8 to i32
      %b32 = arith.extsi %rhs : i8 to i32
      %prod = arith.muli %a32, %b32 : i32
      %sum = arith.addi %prod, %acc : i32
      linalg.yield %sum : i32
  } -> tensor<4x3xi32>
  util.return %result : tensor<4x3xi32>
}
// NEGATIVE-LABEL: @integer_contraction(
// NEGATIVE-NOT: arith.truncf
// NEGATIVE: arith.muli
// NEGATIVE: util.return
// NONE-LABEL: @integer_contraction(
// NONE: util.return

// -----

// Already mixed-precision contractions must not get another input conversion.
util.func public @mixed_precision(%a: tensor<4x8xf16>, %b: tensor<8x3xf16>, %init: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<4x8xf16>, tensor<8x3xf16>)
      outs(%init : tensor<4x3xf32>) {
    ^bb0(%lhs: f16, %rhs: f16, %acc: f32):
      %a32 = arith.extf %lhs : f16 to f32
      %b32 = arith.extf %rhs : f16 to f32
      %prod = arith.mulf %a32, %b32 : f32
      %sum = arith.addf %prod, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<4x3xf32>
  util.return %result : tensor<4x3xf32>
}
// NEGATIVE-LABEL: @mixed_precision(
// NEGATIVE-NOT: arith.truncf
// NEGATIVE: arith.mulf
// NEGATIVE: util.return
// NONE-LABEL: @mixed_precision(
// NONE: util.return
