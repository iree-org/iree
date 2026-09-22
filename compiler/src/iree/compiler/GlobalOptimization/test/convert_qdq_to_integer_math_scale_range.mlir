// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-convert-qdq-to-integer-math,linalg-inline-scalar-operands,canonicalize))" %s | FileCheck %s

// Numerical corner cases in integer contractions and their scaling epilogues.
// Inline constant scalar scales and canonicalize to expose the arithmetic
// retained by the rewrite. FIXME checks record current behavior, not a promise
// that these numerical differences should be preserved.

// This contraction returns zero: (1 * 1e20) * (0 * 1e20) = 0.
// Apply each scale to the accumulator in sequence; combining the scales first
// would overflow to infinity and produce NaN from 0 * infinity.
func.func @scale_product_overflow() -> tensor<1x1xf32> {
  %aq = arith.constant dense<1> : tensor<1x1xi8>
  %bq = arith.constant dense<0> : tensor<1x1xi8>
  %scale = arith.constant 1.0e20 : f32
  %ai = tensor.empty() : tensor<1x1xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %scale : tensor<1x1xi8>, f32)
      outs(%ai : tensor<1x1xf32>) -> tensor<1x1xf32>
  %bi = tensor.empty() : tensor<1x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %scale : tensor<1x1xi8>, f32)
      outs(%bi : tensor<1x1xf32>) -> tensor<1x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @scale_product_overflow(
// CHECK: %[[SCALE:.+]] = arith.constant 1.000000e+20 : f32
// CHECK: linalg.generic {{.*}} outs(%{{.+}} : tensor<1x1xi32>)
// CHECK: %[[REAL:.+]] = arith.sitofp %{{.+}} : i32 to f32
// CHECK-NEXT: %[[PARTIAL:.+]] = arith.mulf %[[REAL]], %[[SCALE]] : f32
// CHECK-NEXT: %[[RESULT:.+]] = arith.mulf %[[PARTIAL]], %[[SCALE]] : f32
// CHECK-NEXT: linalg.yield %[[RESULT]] : f32

// -----

// FIXME: Preserve dequantization rounding, or explicitly document that removing
// it is permitted by this pass's numerical contract. With scales of 1, f16
// dequantization rounds 2049 to 2048, so the original dot product is zero.
// The rewrite instead computes -2048 + 2049 = 1 in i32 and rounds only the
// final result to f16. The checks below record the bypassed input rounding.
func.func @f16_dequantization_rounding() -> tensor<1x1xf16> {
  %aq = arith.constant dense<[[-1, 1]]> : tensor<1x2xi8>
  %bq = arith.constant dense<[[2048], [2049]]> : tensor<2x1xi16>
  %scale = arith.constant 1.0 : f32
  %ai = tensor.empty() : tensor<1x2xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %scale : tensor<1x2xi8>, f32)
      outs(%ai : tensor<1x2xf16>) -> tensor<1x2xf16>
  %bi = tensor.empty() : tensor<2x1xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %scale : tensor<2x1xi16>, f32)
      outs(%bi : tensor<2x1xf16>) -> tensor<2x1xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<1x1xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<1x1xf16>) -> tensor<1x1xf16>
  %result = linalg.matmul ins(%a, %b : tensor<1x2xf16>, tensor<2x1xf16>)
      outs(%init : tensor<1x1xf16>) -> tensor<1x1xf16>
  return %result : tensor<1x1xf16>
}
// CHECK-LABEL: func.func @f16_dequantization_rounding(
// CHECK: linalg.generic {{.*}} outs(%{{.+}} : tensor<1x1xi32>)
// CHECK: arith.extsi %{{.+}} : i16 to i32
// CHECK: linalg.generic {{.*}} outs(%{{.+}} : tensor<1x1xf16>)
// CHECK: %[[REAL:.+]] = arith.sitofp %{{.+}} : i32 to f16
// CHECK-NEXT: linalg.yield %[[REAL]] : f16

// -----

// FIXME: Preserve signed-zero semantics or reject a negative-zero init. The
// original computation is -0 + (0 * -1) = -0, with both scales equal to 1.
// The rewrite accepts the init but replaces it with integer zero, so its
// result is +0. The checks record the zero init and direct integer-to-float
// conversion that loses the original sign.
func.func @negative_zero_init() -> tensor<1x1xf32> {
  %aq = arith.constant dense<0> : tensor<1x1xi8>
  %bq = arith.constant dense<-1> : tensor<1x1xi8>
  %scale = arith.constant 1.0 : f32
  %ai = tensor.empty() : tensor<1x1xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %scale : tensor<1x1xi8>, f32)
      outs(%ai : tensor<1x1xf32>) -> tensor<1x1xf32>
  %bi = tensor.empty() : tensor<1x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %scale : tensor<1x1xi8>, f32)
      outs(%bi : tensor<1x1xf32>) -> tensor<1x1xf32>
  %zero = arith.constant -0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @negative_zero_init(
// CHECK: %[[ZERO:.+]] = arith.constant 0 : i32
// CHECK: linalg.fill ins(%[[ZERO]] : i32)
// CHECK: linalg.generic {{.*}} outs(%{{.+}} : tensor<1x1xi32>)
// CHECK: linalg.generic {{.*}} outs(%{{.+}} : tensor<1x1xf32>)
// CHECK: %[[REAL:.+]] = arith.sitofp %{{.+}} : i32 to f32
// CHECK-NEXT: linalg.yield %[[REAL]] : f32

// -----

// FIXME: Preserve NaN propagation for infinite scales, or explicitly exclude
// non-finite scales from the supported input contract. The original lhs is
// [0 * infinity, 1 * infinity] = [NaN, infinity], so its dot product with [1, 1]
// is NaN. The rewrite computes an integer dot product of 1, then multiplies by
// infinity and returns infinity. These checks record the lost NaN propagation.
func.func @infinite_scale_nan_propagation() -> tensor<1x1xf32> {
  %aq = arith.constant dense<[[0, 1]]> : tensor<1x2xi8>
  %bq = arith.constant dense<1> : tensor<2x1xi8>
  %sa = arith.constant 0x7F800000 : f32
  %sb = arith.constant 1.0 : f32
  %ai = tensor.empty() : tensor<1x2xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%aq, %sa : tensor<1x2xi8>, f32)
      outs(%ai : tensor<1x2xf32>) -> tensor<1x2xf32>
  %bi = tensor.empty() : tensor<2x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%bq, %sb : tensor<2x1xi8>, f32)
      outs(%bi : tensor<2x1xf32>) -> tensor<2x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x2xf32>, tensor<2x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @infinite_scale_nan_propagation(
// CHECK: %[[INF:.+]] = arith.constant 0x7F800000 : f32
// CHECK: linalg.generic {{.*}} outs(%{{.+}} : tensor<1x1xi32>)
// CHECK: %[[REAL:.+]] = arith.sitofp %{{.+}} : i32 to f32
// CHECK-NEXT: %[[RESULT:.+]] = arith.mulf %[[REAL]], %[[INF]] : f32
// CHECK-NEXT: linalg.yield %[[RESULT]] : f32

// -----

// This contraction returns approximately 1.07368e-37, a normal f32 value.
// Apply each 1e-23 scale to the accumulator in sequence; combining the scales
// first would underflow to zero and erase the result.
func.func @scale_product_underflow() -> tensor<1x1xf32> {
  %q = arith.constant dense<32767> : tensor<1x1xi16>
  %scale = arith.constant 1.0e-23 : f32
  %ai = tensor.empty() : tensor<1x1xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(m, k) -> (m, k)>,
                        affine_map<(m, k) -> ()>,
                        affine_map<(m, k) -> (m, k)>]}
      ins(%q, %scale : tensor<1x1xi16>, f32)
      outs(%ai : tensor<1x1xf32>) -> tensor<1x1xf32>
  %bi = tensor.empty() : tensor<1x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(k, n) -> (k, n)>,
                        affine_map<(k, n) -> ()>,
                        affine_map<(k, n) -> (k, n)>]}
      ins(%q, %scale : tensor<1x1xi16>, f32)
      outs(%bi : tensor<1x1xf32>) -> tensor<1x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @scale_product_underflow(
// CHECK: %[[SCALE:.+]] = arith.constant 9.99999999E-24 : f32
// CHECK: linalg.generic {{.*}} outs(%{{.+}} : tensor<1x1xi32>)
// CHECK: %[[REAL:.+]] = arith.sitofp %{{.+}} : i32 to f32
// CHECK-NEXT: %[[PARTIAL:.+]] = arith.mulf %[[REAL]], %[[SCALE]] : f32
// CHECK-NEXT: %[[RESULT:.+]] = arith.mulf %[[PARTIAL]], %[[SCALE]] : f32
// CHECK-NEXT: linalg.yield %[[RESULT]] : f32

// -----

// Both symmetric operands produce a scalar zero correction in the rewrite.
// Scalar inlining exposes it to canonicalization, which folds D - 0 to D and
// removes the correction operand. Runtime scales keep the epilogue observable.
func.func @symmetric_zero_correction_folded(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f32)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @symmetric_zero_correction_folded(
// CHECK-SAME: %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[SA:[a-zA-Z0-9_]+]]: f32,
// CHECK-SAME: %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[SB:[a-zA-Z0-9_]+]]: f32
// CHECK: %[[D:.+]] = linalg.generic {{.*}} ins(%[[AQ]], %[[BQ]] : tensor<4x8xi8>, tensor<8x16xi8>)
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xi32>)
// CHECK: %[[RESULT:.+]] = linalg.generic {{.*}} ins(%[[D]] : tensor<4x16xi32>)
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xf32>)
// CHECK-NEXT: ^bb0(%[[ED:[a-zA-Z0-9_]+]]: i32, %{{.+}}: f32):
// CHECK-NEXT: %[[REAL:.+]] = arith.sitofp %[[ED]] : i32 to f32
// CHECK-NEXT: %[[PARTIAL:.+]] = arith.mulf %[[REAL]], %[[SA]] : f32
// CHECK-NEXT: %[[SCALED:.+]] = arith.mulf %[[PARTIAL]], %[[SB]] : f32
// CHECK-NEXT: linalg.yield %[[SCALED]] : f32
// CHECK: return %[[RESULT]] : tensor<4x16xf32>

// -----

// Cleanup also folds the zero correction for symmetric blockwise quantization
// while preserving scale widening, f32 accumulation, and final narrowing.
func.func @symmetric_half_partial_reduction(%aq: tensor<4x2x4xi8>, %a_s: tensor<4x2xf16>,
    %bq: tensor<2x4x16xi8>, %b_s: f32) -> tensor<4x16xf16> {
  %a_i = tensor.empty() : tensor<4x2x4xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%aq, %a_s : tensor<4x2x4xi8>, tensor<4x2xf16>)
      outs(%a_i : tensor<4x2x4xf16>) -> tensor<4x2x4xf16>
  %b_i = tensor.empty() : tensor<2x4x16xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%bq, %b_s : tensor<2x4x16xi8>, f32)
      outs(%b_i : tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
  %cst = arith.constant 0.000000e+00 : f16
  %e = tensor.empty() : tensor<4x16xf16>
  %f = linalg.fill ins(%cst : f16) outs(%e : tensor<4x16xf16>) -> tensor<4x16xf16>
  %c = linalg.contract
      indexing_maps = [affine_map<(m, n, g, l) -> (m, g, l)>,
                       affine_map<(m, n, g, l) -> (g, l, n)>,
                       affine_map<(m, n, g, l) -> (m, n)>]
      ins(%a, %b : tensor<4x2x4xf16>, tensor<2x4x16xf16>)
      outs(%f : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %c : tensor<4x16xf16>
}
// CHECK-LABEL: func.func @symmetric_half_partial_reduction(
// CHECK-SAME: %{{.+}}: tensor<4x2x4xi8>, %[[SA_INPUT:[a-zA-Z0-9_]+]]: tensor<4x2xf16>,
// CHECK-SAME: %{{.+}}: tensor<2x4x16xi8>, %[[SB:[a-zA-Z0-9_]+]]: f32
// CHECK: %[[D:.+]] = linalg.generic {{.*}} outs(%{{.+}} : tensor<4x16x2xi32>)
// CHECK: %[[PARTIALS:.+]] = linalg.generic {{.*}} ins(%[[D]], %[[SA_INPUT]] : tensor<4x16x2xi32>, tensor<4x2xf16>)
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xf32>)
// CHECK-NEXT: ^bb0(%[[ED:[a-zA-Z0-9_]+]]: i32, %[[SA:[a-zA-Z0-9_]+]]: f16, %[[ACC:[a-zA-Z0-9_]+]]: f32):
// CHECK-NEXT: %[[REAL:.+]] = arith.sitofp %[[ED]] : i32 to f32
// CHECK-NEXT: %[[WIDE_SA:.+]] = arith.extf %[[SA]] : f16 to f32
// CHECK-NEXT: %[[PARTIAL:.+]] = arith.mulf %[[REAL]], %[[WIDE_SA]] : f32
// CHECK-NEXT: %[[SCALED:.+]] = arith.mulf %[[PARTIAL]], %[[SB]] : f32
// CHECK-NEXT: %[[TOTAL:.+]] = arith.addf %[[ACC]], %[[SCALED]] : f32
// CHECK-NEXT: linalg.yield %[[TOTAL]] : f32
// CHECK: linalg.generic {{.*}} ins(%[[PARTIALS]] : tensor<4x16xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xf16>)
// CHECK-NEXT: ^bb0(%[[V:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f16):
// CHECK-NEXT: %[[NARROW:.+]] = arith.truncf %[[V]] : f32 to f16
// CHECK-NEXT: linalg.yield %[[NARROW]] : f16
