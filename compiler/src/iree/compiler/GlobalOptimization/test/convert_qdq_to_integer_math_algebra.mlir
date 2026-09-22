// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-convert-qdq-to-integer-math))" %s | FileCheck %s

// Correction terms, widening, and the arithmetic of the scaling epilogue.
// Indexing lives in convert_qdq_to_integer_math_maps.mlir and the contractions
// the rewrite declines in convert_qdq_to_integer_math_negative.mlir.

// The quantized grid bounds a zero point independently of its carrier type.
// PT2E represents scalar zero points as i64 even for i8 quantization.
func.func @wide_zero_point_carrier(%aq: tensor<4x8xi8>, %a_s: f32,
    %a_z: i64, %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i64)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x16xf32>) -> tensor<4x16xf32>
  %result = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %result : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @wide_zero_point_carrier(
// CHECK-NOT: iree_linalg_ext.dequantize_affine
// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xi32>)

//===----------------------------------------------------------------------===//
// Zero-point corrections
//
// sum_k (Aq - zA)(Bq - zB) = D - zB*RA - zA*RB + N*zA*zB. Which terms exist
// depends on which side carries a zero point, so these four cases add one term
// at a time. The first spells out the shared epilogue arithmetic the rest
// inherit.
//
// D = sum_k Aq[k] * Bq[k]
// RA = sum_k Aq[k]
// RB = sum_k Bq[k]
// N = number of terms in the sum (reduction extent)
//
//===----------------------------------------------------------------------===//

// Both symmetric: nothing to correct, so the epilogue only applies the scales.
// C = sA*sB*D = sA*sB*sum_k Aq[k] * Bq[k]
func.func @zp_neither(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
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
// CHECK-LABEL: func.func @zp_neither(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32,
//  CHECK-SAME:   %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32
// CHECK: %[[ZERO:.+]] = arith.constant 0 : i32
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   %[[D:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     ins(%[[AQ]], %[[BQ]] : tensor<4x8xi8>, tensor<8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
//       CHECK:   ^bb0(%[[L:[a-zA-Z0-9_]+]]: i8, %[[R:[a-zA-Z0-9_]+]]: i8, %[[ACC:[a-zA-Z0-9_]+]]: i32):
//       CHECK:     %[[LE:.+]] = arith.extsi %[[L]] : i8 to i32
//       CHECK:     %[[RE:.+]] = arith.extsi %[[R]] : i8 to i32
//       CHECK:     %[[P:.+]] = arith.muli %[[LE]], %[[RE]]
//       CHECK:     %[[SUM:.+]] = arith.addi %[[ACC]], %[[P]] : i32
//  CHECK-NEXT:     linalg.yield %[[SUM]] : i32

// The correction input is scalar i32 zero. This is the scaling epilogue: it
// yields (float(D - 0) * sA) * sB. Subsequent scalar inlining and
// canonicalization fold away the subtraction.
// CHECK: %[[RESULT:.+]] = linalg.generic {{.*}} ins(%[[D]], %[[ZERO]], %[[AS]], %[[BS]] : tensor<4x16xi32>, i32, f32, f32)
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xf32>)
// CHECK: ^bb0(%[[ED:[a-zA-Z0-9_]+]]: i32, %[[EC:[a-zA-Z0-9_]+]]: i32, %[[SA:[a-zA-Z0-9_]+]]: f32, %[[SB:[a-zA-Z0-9_]+]]: f32, %[[ACC:[a-zA-Z0-9_]+]]: f32):
// CHECK: %[[CORRECTED:.+]] = arith.subi %[[ED]], %[[EC]] : i32
// CHECK: %[[REAL:.+]] = arith.sitofp %[[CORRECTED]] : i32 to f32
// CHECK: %[[PARTIAL:.+]] = arith.mulf %[[REAL]], %[[SA]] : f32
// CHECK: %[[SCALED:.+]] = arith.mulf %[[PARTIAL]], %[[SB]] : f32
// CHECK-NEXT: linalg.yield %[[SCALED]] : f32
// CHECK: return %[[RESULT]] : tensor<4x16xf32>

// -----

// A zero point on the lhs produces the *rhs* sum and the `- zA*RB` term.
func.func @zp_lhs_only(%aq: tensor<4x8xi8>, %a_s: f32, %a_z: i8, %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i8)
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
// CHECK-LABEL: func.func @zp_lhs_only(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32, %[[AZ:[a-zA-Z0-9_]+]]: i8,
//  CHECK-SAME:   %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   %[[D:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     ins(%[[AQ]], %[[BQ]] :
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)

// One sum, and it reduces the *rhs*, which is what pairs with the lhs zero
// point.
//       CHECK:   %[[RB:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     ins(%[[BQ]] : tensor<8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)

// It is a plain accumulation of the widened operand, nothing more.
//       CHECK:   ^bb0(%[[SV:[a-zA-Z0-9_]+]]: i8, %[[SACC:[a-zA-Z0-9_]+]]: i32):
//       CHECK:     %[[SVE:.+]] = arith.extsi %[[SV]] : i8 to i32
//       CHECK:     %[[SUM:.+]] = arith.addi %[[SACC]], %[[SVE]]
//       CHECK:     linalg.yield %[[SUM]]

// The zero-point product is independently evaluable and retains the sum shape.
// CHECK: %[[TERM:.+]] = linalg.generic {{.*}} ins(%[[AZ]], %[[RB]] : i8, tensor<16xi32>)
// CHECK-SAME: outs(%{{.+}} : tensor<16xi32>)
// CHECK: ^bb0(%[[Z:[a-zA-Z0-9_]+]]: i8, %[[SUM:[a-zA-Z0-9_]+]]: i32, %{{.+}}: i32):
// CHECK: %[[WIDE:.+]] = arith.extsi %[[Z]] : i8 to i32
// CHECK: %[[PRODUCT:.+]] = arith.muli %[[WIDE]], %[[SUM]]
// CHECK: linalg.yield %[[PRODUCT]]
// CHECK: linalg.generic {{.*}} ins(%[[D]], %[[TERM]], %[[AS]], %[[BS]] :
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xf32>)
// CHECK: ^bb0(%[[ED:[a-zA-Z0-9_]+]]: i32, %[[EC:[a-zA-Z0-9_]+]]: i32, %[[SA:[a-zA-Z0-9_]+]]: f32, %[[SB:[a-zA-Z0-9_]+]]: f32, %[[ACC:[a-zA-Z0-9_]+]]: f32):
// CHECK: %[[CORRECTED:.+]] = arith.subi %[[ED]], %[[EC]] : i32
// CHECK: %[[REAL:.+]] = arith.sitofp %[[CORRECTED]] : i32 to f32

// -----

// The mirror image: a zero point on the rhs produces the lhs sum.
func.func @zp_rhs_only(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
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
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<8x16xi8>, f32, i8)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @zp_rhs_only(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32,
//  CHECK-SAME:   %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32, %[[BZ:[a-zA-Z0-9_]+]]: i8
//       CHECK:   %[[D:.+]] = linalg.generic {{.*}} ins(%[[AQ]], %[[BQ]] :
// The mirror pairing uses the lhs sum and rhs zero point.
//       CHECK:   %[[RA:.+]] = linalg.generic {{.*}} ins(%[[AQ]] : tensor<4x8xi8>)
//       CHECK:   %[[TERM:.+]] = linalg.generic {{.*}} ins(%[[BZ]], %[[RA]] : i8, tensor<4xi32>)
//       CHECK:   linalg.generic {{.*}} ins(%[[D]], %[[TERM]], %[[AS]], %[[BS]] :

// -----

// Both asymmetric: combine the two products as TA + TB - N*zA*zB.
// TA = zB * RA, TB = zA * RB
// The cross term is needed only when both zero points are present; N is 8.
func.func @zp_both(%aq: tensor<4x8xi8>, %a_s: f32, %a_z: i8, %bq: tensor<8x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i8)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<8x16xi8>, f32, i8)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @zp_both(
// CHECK-SAME: %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32, %[[ZA:[a-zA-Z0-9_]+]]: i8,
// CHECK-SAME: %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32, %[[ZB:[a-zA-Z0-9_]+]]: i8
// CHECK: %[[N:.+]] = arith.constant 8 : i32
// CHECK: %[[D:.+]] = linalg.generic {{.*}} ins(%[[AQ]], %[[BQ]] :
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xi32>)
// CHECK: %[[RA:.+]] = linalg.generic {{.*}} ins(%[[AQ]] : tensor<4x8xi8>)
// CHECK-SAME: outs(%{{.+}} : tensor<4xi32>)
// CHECK: %[[TA:.+]] = linalg.generic {{.*}} ins(%[[ZB]], %[[RA]] : i8, tensor<4xi32>)
// CHECK-SAME: outs(%{{.+}} : tensor<4xi32>)
// CHECK: %[[RB:.+]] = linalg.generic {{.*}} ins(%[[BQ]] : tensor<8x16xi8>)
// CHECK-SAME: outs(%{{.+}} : tensor<16xi32>)
// CHECK: %[[TB:.+]] = linalg.generic {{.*}} ins(%[[ZA]], %[[RB]] : i8, tensor<16xi32>)
// CHECK-SAME: outs(%{{.+}} : tensor<16xi32>)

// The cross term stays scalar; only combining row/column terms needs M x N.
// CHECK: %[[WA:.+]] = arith.extsi %[[ZA]] : i8 to i32
// CHECK: %[[WB:.+]] = arith.extsi %[[ZB]] : i8 to i32
// CHECK: %[[ZZ:.+]] = arith.muli %[[WA]], %[[WB]]
// CHECK: %[[TZ:.+]] = arith.muli %[[ZZ]], %[[N]]
// CHECK: %[[CORRECTION:.+]] = linalg.generic {{.*}} ins(%[[TA]], %[[TB]], %[[TZ]] :
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xi32>)
// CHECK: ^bb0(%[[A:[a-zA-Z0-9_]+]]: i32, %[[B:[a-zA-Z0-9_]+]]: i32, %[[Z:[a-zA-Z0-9_]+]]: i32, %{{.+}}: i32):
// CHECK: %[[SUM:.+]] = arith.addi %[[A]], %[[B]]
// CHECK: %[[OFFSET:.+]] = arith.subi %[[SUM]], %[[Z]]
// CHECK: linalg.yield %[[OFFSET]]
// CHECK: linalg.generic {{.*}} ins(%[[D]], %[[CORRECTION]], %[[AS]], %[[BS]] :
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xf32>)

//===----------------------------------------------------------------------===//
// Widening
//
// Storage signedness and zero-point signedness are independent per operand. The
// accumulator is signed by construction whatever the inputs were, so the final
// conversion to float is always signed.
//===----------------------------------------------------------------------===//

// -----

// Unsigned storage and unsigned zero points: every widening is a zero extension.
// The final conversion stays signed, because the accumulator is signed by
// construction whatever the inputs were.
func.func @both_unsigned(%aq: tensor<4x8xi8>, %a_s: f32, %a_z: i8, %bq: tensor<8x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i8)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<8x16xi8>, f32, i8)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @both_unsigned(
// CHECK-SAME: %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32, %[[AZ:[a-zA-Z0-9_]+]]: i8,
// CHECK-SAME: %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32, %[[BZ:[a-zA-Z0-9_]+]]: i8
// CHECK: linalg.generic {{.*}} ins(%[[AQ]], %[[BQ]] :
// CHECK-NEXT: ^bb0(%[[L:[a-zA-Z0-9_]+]]: i8, %[[R:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32):
// CHECK-DAG: arith.extui %[[L]] : i8 to i32
// CHECK-DAG: arith.extui %[[R]] : i8 to i32
// CHECK: %[[RA:.+]] = linalg.generic {{.*}} ins(%[[AQ]] :
// CHECK-NEXT: ^bb0(%[[A:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32):
// CHECK: arith.extui %[[A]] : i8 to i32
// CHECK: linalg.generic {{.*}} ins(%[[BZ]], %[[RA]] :
// CHECK-NEXT: ^bb0(%[[ZB:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32, %{{.+}}: i32):
// CHECK: arith.extui %[[ZB]] : i8 to i32
// CHECK: %[[RB:.+]] = linalg.generic {{.*}} ins(%[[BQ]] :
// CHECK-NEXT: ^bb0(%[[B:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32):
// CHECK: arith.extui %[[B]] : i8 to i32
// CHECK: linalg.generic {{.*}} ins(%[[AZ]], %[[RB]] :
// CHECK-NEXT: ^bb0(%[[ZA:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32, %{{.+}}: i32):
// CHECK: arith.extui %[[ZA]] : i8 to i32
// CHECK-DAG: arith.extui %[[AZ]] : i8 to i32
// CHECK-DAG: arith.extui %[[BZ]] : i8 to i32
// CHECK: arith.sitofp

// -----

// The combination PT2E emits on x86: unsigned activations against signed
// weights. The two operands must widen differently within the same op.
func.func @lhs_unsigned_rhs_signed(%aq: tensor<4x8xi8>, %a_s: f32, %a_z: i8, %bq: tensor<8x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i8)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<8x16xi8>, f32, i8)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @lhs_unsigned_rhs_signed(
// CHECK-SAME: %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32, %[[AZ:[a-zA-Z0-9_]+]]: i8,
// CHECK-SAME: %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32, %[[BZ:[a-zA-Z0-9_]+]]: i8
// CHECK: linalg.generic {{.*}} ins(%[[AQ]], %[[BQ]] :
// CHECK-NEXT: ^bb0(%[[L:[a-zA-Z0-9_]+]]: i8, %[[R:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32):
// CHECK-DAG: arith.extui %[[L]] : i8 to i32
// CHECK-DAG: arith.extsi %[[R]] : i8 to i32
// CHECK: %[[RA:.+]] = linalg.generic {{.*}} ins(%[[AQ]] :
// CHECK-NEXT: ^bb0(%[[A:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32):
// CHECK: arith.extui %[[A]] : i8 to i32
// CHECK: linalg.generic {{.*}} ins(%[[BZ]], %[[RA]] :
// CHECK-NEXT: ^bb0(%[[ZB:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32, %{{.+}}: i32):
// CHECK: arith.extsi %[[ZB]] : i8 to i32
// CHECK: %[[RB:.+]] = linalg.generic {{.*}} ins(%[[BQ]] :
// CHECK-NEXT: ^bb0(%[[B:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32):
// CHECK: arith.extsi %[[B]] : i8 to i32
// CHECK: linalg.generic {{.*}} ins(%[[AZ]], %[[RB]] :
// CHECK-NEXT: ^bb0(%[[ZA:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32, %{{.+}}: i32):
// CHECK: arith.extui %[[ZA]] : i8 to i32
// CHECK-DAG: arith.extui %[[AZ]] : i8 to i32
// CHECK-DAG: arith.extsi %[[BZ]] : i8 to i32

// -----

// `input_unsigned` and `zp_unsigned` are independent: an unsigned tensor may
// carry a signed zero point of a wider type, which is what ONNX and TFLite
// importers produce. The data zero extends while the zero point sign extends.
func.func @zp_signedness_differs_from_storage(%aq: tensor<4x8xi8>, %a_s: f32, %a_z: i16, %bq: tensor<8x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i16)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<8x16xi8>, f32, i8)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @zp_signedness_differs_from_storage(
// CHECK-SAME: %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32, %[[AZ:[a-zA-Z0-9_]+]]: i16,
// CHECK-SAME: %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32, %[[BZ:[a-zA-Z0-9_]+]]: i8
// CHECK: linalg.generic {{.*}} ins(%[[AQ]], %[[BQ]] :
// CHECK-NEXT: ^bb0(%[[L:[a-zA-Z0-9_]+]]: i8, %[[R:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32):
// CHECK-DAG: arith.extui %[[L]] : i8 to i32
// CHECK-DAG: arith.extsi %[[R]] : i8 to i32
// CHECK: %[[RA:.+]] = linalg.generic {{.*}} ins(%[[AQ]] :
// CHECK-NEXT: ^bb0(%[[A:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32):
// CHECK: arith.extui %[[A]] : i8 to i32
// CHECK: linalg.generic {{.*}} ins(%[[BZ]], %[[RA]] :
// CHECK-NEXT: ^bb0(%[[ZB:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32, %{{.+}}: i32):
// CHECK: arith.extsi %[[ZB]] : i8 to i32
// CHECK: %[[RB:.+]] = linalg.generic {{.*}} ins(%[[BQ]] :
// CHECK-NEXT: ^bb0(%[[B:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32):
// CHECK: arith.extsi %[[B]] : i8 to i32
// CHECK: linalg.generic {{.*}} ins(%[[AZ]], %[[RB]] :
// CHECK-NEXT: ^bb0(%[[ZA:[a-zA-Z0-9_]+]]: i16, %{{.+}}: i32, %{{.+}}: i32):
// CHECK: arith.extsi %[[ZA]] : i16 to i32
// CHECK-DAG: arith.extsi %[[AZ]] : i16 to i32
// CHECK-DAG: arith.extsi %[[BZ]] : i8 to i32

//===----------------------------------------------------------------------===//
// Contraction body
//
// Matching accepts either operand order in the multiply and in the add.
// Bodies that do anything else are in
// convert_qdq_to_integer_math_negative.mlir.
//===----------------------------------------------------------------------===//

// -----

// Both the multiply and the add have their operands the other way round. The
// rebuilt body is in canonical order regardless of the order matched.
func.func @commuted_mul_add(%aq: tensor<1x8xi8>, %sa: f32,
    %bq: tensor<8x1xi8>, %sb: f32) -> tensor<1x1xf32> {
  %ai = tensor.empty() : tensor<1x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%aq, %sa : tensor<1x8xi8>, f32)
      outs(%ai : tensor<1x8xf32>) -> tensor<1x8xf32>
  %bi = tensor.empty() : tensor<8x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%bq, %sb : tensor<8x1xi8>, f32)
      outs(%bi : tensor<8x1xf32>) -> tensor<8x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<1x8xf32>, tensor<8x1xf32>) outs(%init : tensor<1x1xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %rhs, %lhs : f32
      %sum = arith.addf %product, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @commuted_mul_add(
//       CHECK:   ^bb0(%[[L:[a-zA-Z0-9_]+]]: i8, %[[R:[a-zA-Z0-9_]+]]: i8, %[[ACC:[a-zA-Z0-9_]+]]: i32):
//   CHECK-DAG:     %[[LE:.+]] = arith.extsi %[[L]]
//   CHECK-DAG:     %[[RE:.+]] = arith.extsi %[[R]]
//       CHECK:     %[[P:.+]] = arith.muli %[[LE]], %[[RE]]
//       CHECK:     arith.addi %[[ACC]], %[[P]]

//===----------------------------------------------------------------------===//
// Reduction form
//
// Reduction dims split by whether a quantization parameter varies along them.
// Only the arithmetic consequence is checked here; the resulting maps and
// iterator types belong to the maps file.
//===----------------------------------------------------------------------===//

// -----

// Blockwise: K is expanded to (G, L) and the lhs parameters vary along G, so the
// contraction is integer within a block and the residual sum across blocks
// happens in floating point. The observable is the trailing addf in the
// epilogue, which no other form produces.
func.func @form_partial_reduces_in_float(%aq: tensor<4x2x4xi8>, %a_s: tensor<4x2xf32>, %a_z: tensor<4x2xi8>,
    %bq: tensor<2x4x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x2x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%aq, %a_s, %a_z : tensor<4x2x4xi8>, tensor<4x2xf32>, tensor<4x2xi8>)
      outs(%a_i : tensor<4x2x4xf32>) -> tensor<4x2x4xf32>
  %b_i = tensor.empty() : tensor<2x4x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%bq, %b_s : tensor<2x4x16xi8>, f32)
      outs(%b_i : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.contract
      indexing_maps = [affine_map<(m, n, g, l) -> (m, g, l)>,
                       affine_map<(m, n, g, l) -> (g, l, n)>,
                       affine_map<(m, n, g, l) -> (m, n)>]
      ins(%a, %b : tensor<4x2x4xf32>, tensor<2x4x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @form_partial_reduces_in_float(
// CHECK: %[[D:.+]] = linalg.generic {{.*}} outs(%{{.+}} : tensor<4x16x2xi32>)
// CHECK: %[[CORRECTION:.+]] = linalg.generic {{.*}} outs(%{{.+}} : tensor<4x16x2xi32>)
// CHECK: linalg.generic {{.*}} ins(%[[D]], %[[CORRECTION]], %{{[^ ,)]+}}, %{{[^ ,)]+}} :
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xf32>)
// CHECK-NEXT: ^bb0(%{{.+}}: i32, %{{.+}}: i32, %{{.+}}: f32, %{{.+}}: f32, %[[ACC:[a-zA-Z0-9_]+]]: f32):
// CHECK: %[[PARTIAL:.+]] = arith.mulf
// CHECK: %[[SCALED:.+]] = arith.mulf %[[PARTIAL]], %{{.+}}
// CHECK: %[[TOTAL:.+]] = arith.addf %[[ACC]], %[[SCALED]] : f32
// CHECK: linalg.yield %[[TOTAL]] : f32

//===----------------------------------------------------------------------===//
// Scale precision
//
// Scaling uses at least f32, preserving wider scale and output types. The result
// is narrowed once, after every partial has accumulated, so a low-precision
// result never rounds a partial sum.
//===----------------------------------------------------------------------===//

// -----

// An f16 contraction with f32 scales, as a half precision PT2E model produces.
// Scaling stays in f32 until the final result is narrowed.
func.func @wider_scales(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>,
    %b_s: f32) -> tensor<4x16xf16> {
  %a_i = tensor.empty() : tensor<4x8xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f32)
      outs(%a_i : tensor<4x8xf16>) -> tensor<4x8xf16>
  %b_i = tensor.empty() : tensor<8x16xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf16>) -> tensor<8x16xf16>
  %cst = arith.constant 0.000000e+00 : f16
  %e = tensor.empty() : tensor<4x16xf16>
  %f = linalg.fill ins(%cst : f16) outs(%e : tensor<4x16xf16>) -> tensor<4x16xf16>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf16>, tensor<8x16xf16>)
      outs(%f : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %c : tensor<4x16xf16>
}
// CHECK-LABEL: func.func @wider_scales(
// CHECK: linalg.generic {{.*}} outs(%{{.+}} : tensor<4x16xf16>)
// CHECK: arith.sitofp %{{.+}} : i32 to f32
// CHECK: %[[PARTIAL:.+]] = arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK: %[[SCALED:.+]] = arith.mulf %[[PARTIAL]], %{{.+}} : f32
// CHECK: %[[NARROW:.+]] = arith.truncf %[[SCALED]] : f32 to f16
// CHECK: linalg.yield %[[NARROW]] : f16

// -----

// The same rule with the scale wider than the result rather than between the
// result and f32: the epilogue computes in f64 and truncates once.
func.func @double_scales(%aq: tensor<4x8xi8>, %a_s: f64, %bq: tensor<8x16xi8>,
    %b_s: f64) -> tensor<4x16xf16> {
  %a_i = tensor.empty() : tensor<4x8xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f64)
      outs(%a_i : tensor<4x8xf16>) -> tensor<4x8xf16>
  %b_i = tensor.empty() : tensor<8x16xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f64)
      outs(%b_i : tensor<8x16xf16>) -> tensor<8x16xf16>
  %cst = arith.constant 0.000000e+00 : f16
  %e = tensor.empty() : tensor<4x16xf16>
  %f = linalg.fill ins(%cst : f16) outs(%e : tensor<4x16xf16>) -> tensor<4x16xf16>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf16>, tensor<8x16xf16>)
      outs(%f : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %c : tensor<4x16xf16>
}
// CHECK-LABEL: func.func @double_scales(
// CHECK: arith.sitofp %{{.+}} : i32 to f64
// CHECK: %[[PARTIAL:.+]] = arith.mulf %{{.+}}, %{{.+}} : f64
// CHECK: %[[SCALED:.+]] = arith.mulf %[[PARTIAL]], %{{.+}} : f64
// CHECK: %[[NARROW:.+]] = arith.truncf %[[SCALED]] : f64 to f16
// CHECK: linalg.yield %[[NARROW]] : f16

// -----

// Widen the half scale before multiplication and narrow only after all blocks
// have accumulated in f32.
func.func @half_partial_reduction(%aq: tensor<4x2x4xi8>, %a_s: tensor<4x2xf16>, %a_z: tensor<4x2xi8>,
    %bq: tensor<2x4x16xi8>, %b_s: f32) -> tensor<4x16xf16> {
  %a_i = tensor.empty() : tensor<4x2x4xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%aq, %a_s, %a_z : tensor<4x2x4xi8>, tensor<4x2xf16>, tensor<4x2xi8>)
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
// CHECK-LABEL: func.func @half_partial_reduction(
// This epilogue corrects and scales each integer block result, then sums all
// blocks in f32. ACCUMULATED is the complete result, with no block dimension.
// CHECK: %[[ACCUMULATED:.+]] = linalg.generic {{.*}} outs(%{{.+}} : tensor<4x16xf32>)
// CHECK-NEXT: ^bb0(%{{.+}}: i32, %{{.+}}: i32, %[[SA:[a-zA-Z0-9_]+]]: f16, %[[SB:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f32):
// CHECK: %[[WIDE_SA:.+]] = arith.extf %[[SA]] : f16 to f32
// CHECK: %[[PARTIAL:.+]] = arith.mulf %{{.+}}, %[[WIDE_SA]] : f32
// CHECK-NEXT: arith.mulf %[[PARTIAL]], %[[SB]] : f32
// CHECK-NOT: arith.truncf
// CHECK: %[[TOTAL:.+]] = arith.addf %{{.+}}, %{{.+}} : f32
// CHECK-NOT: arith.truncf
// CHECK: linalg.yield %[[TOTAL]] : f32
// A separate elementwise generic narrows each completed result to f16 once.
// CHECK: linalg.generic {{.*}} ins(%[[ACCUMULATED]] : tensor<4x16xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xf16>)
// CHECK-NEXT: ^bb0(%[[V:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f16):
// CHECK: %[[NARROW:.+]] = arith.truncf %[[V]] : f32 to f16
// CHECK: linalg.yield %[[NARROW]] : f16

//===----------------------------------------------------------------------===//
// Accumulator bound
//
// The bound accounts for the input and zero-point ranges, so the same shape
// can be admissible symmetric and inadmissible asymmetric. The rejected half
// of each pair below is in
// convert_qdq_to_integer_math_negative.mlir.
//===----------------------------------------------------------------------===//

// -----

// i8 by i8 with K = 65536. Symmetric, so a single term, and 7+7+16+0+1 = 31 bits
// fit. Making both sides asymmetric costs two more bits and no longer fits: see
// @acc_asymmetric_too_deep.
func.func @acc_symmetric_fits_deep(%aq: tensor<4x65536xi8>, %a_s: f32, %bq: tensor<65536x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x65536xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x65536xi8>, f32)
      outs(%a_i : tensor<4x65536xf32>) -> tensor<4x65536xf32>
  %b_i = tensor.empty() : tensor<65536x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<65536x16xi8>, f32)
      outs(%b_i : tensor<65536x16xf32>) -> tensor<65536x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x65536xf32>, tensor<65536x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @acc_symmetric_fits_deep(
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.generic

// -----

// Mixed storage widths are bounded per operand rather than by the wider of the
// two: 15+7 magnitude bits plus a shallow reduction still fits.
func.func @acc_mixed_width_fits(%aq: tensor<4x8xi16>, %a_s: f32, %a_z: i16, %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi16>, f32, i16)
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
// CHECK-LABEL: func.func @acc_mixed_width_fits(
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<4x8xi16>, tensor<8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)

// -----

// The deepest signed-i8 symmetric reduction below the positive i32 endpoint.
// One element more overflows it: see @acc_positive_endpoint.
func.func @acc_last_safe_depth(%aq: tensor<1x131071xi8>, %sa: f32,
    %bq: tensor<131071x1xi8>, %sb: f32) -> tensor<1x1xf32> {
  %ai = tensor.empty() : tensor<1x131071xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%aq, %sa : tensor<1x131071xi8>, f32)
      outs(%ai : tensor<1x131071xf32>) -> tensor<1x131071xf32>
  %bi = tensor.empty() : tensor<131071x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%bq, %sb : tensor<131071x1xi8>, f32)
      outs(%bi : tensor<131071x1xf32>) -> tensor<131071x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x131071xf32>, tensor<131071x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @acc_last_safe_depth(
// CHECK: tensor<1x1xi32>
// CHECK-NOT: linalg.matmul

// -----

// P+Q is bounded by 2*N*255^2. Accept floor(INT32_MAX/(2*255^2)),
// beyond the old four-term magnitude bound of 8256.
func.func @unsigned_reduction_16512(%aq: tensor<1x16512xi8>, %a_s: f32, %a_z: i8, %bq: tensor<16512x1xi8>, %b_s: f32, %b_z: i8) -> tensor<1x1xf32> {
  %a_i = tensor.empty() : tensor<1x16512xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<1x16512xi8>, f32, i8)
      outs(%a_i : tensor<1x16512xf32>) -> tensor<1x16512xf32>
  %b_i = tensor.empty() : tensor<16512x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<16512x1xi8>, f32, i8)
      outs(%b_i : tensor<16512x1xf32>) -> tensor<16512x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<1x1xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<1x1xf32>) -> tensor<1x1xf32>
  %c = linalg.matmul ins(%a, %b : tensor<1x16512xf32>, tensor<16512x1xf32>)
      outs(%f : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %c : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @unsigned_reduction_16512(
// CHECK-NOT: iree_linalg_ext.dequantize_affine
// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.+}} : tensor<1x1xi32>)

// -----

// The centered product is in [-255^2, 255^2] and bounds all intermediates.
// Accept the last safe depth.
func.func @signed_reduction_33025(%aq: tensor<1x33025xi8>, %a_s: f32, %a_z: i8, %bq: tensor<33025x1xi8>, %b_s: f32, %b_z: i8) -> tensor<1x1xf32> {
  %a_i = tensor.empty() : tensor<1x33025xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<1x33025xi8>, f32, i8)
      outs(%a_i : tensor<1x33025xf32>) -> tensor<1x33025xf32>
  %b_i = tensor.empty() : tensor<33025x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<33025x1xi8>, f32, i8)
      outs(%b_i : tensor<33025x1xf32>) -> tensor<33025x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<1x1xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<1x1xf32>) -> tensor<1x1xf32>
  %c = linalg.matmul ins(%a, %b : tensor<1x33025xf32>, tensor<33025x1xf32>)
      outs(%f : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %c : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @signed_reduction_33025(
// CHECK-NOT: iree_linalg_ext.dequantize_affine
// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.+}} : tensor<1x1xi32>)

// -----

// The correction interval is [-97665, 97410], larger than the centered product.
// Accept the last safe depth.
func.func @mixed_reduction_21988(%aq: tensor<1x21988xi8>, %a_s: f32, %a_z: i8, %bq: tensor<21988x1xi8>, %b_s: f32, %b_z: i8) -> tensor<1x1xf32> {
  %a_i = tensor.empty() : tensor<1x21988xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<1x21988xi8>, f32, i8)
      outs(%a_i : tensor<1x21988xf32>) -> tensor<1x21988xf32>
  %b_i = tensor.empty() : tensor<21988x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<21988x1xi8>, f32, i8)
      outs(%b_i : tensor<21988x1xf32>) -> tensor<21988x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<1x1xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<1x1xf32>) -> tensor<1x1xf32>
  %c = linalg.matmul ins(%a, %b : tensor<1x21988xf32>, tensor<21988x1xf32>)
      outs(%f : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %c : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @mixed_reduction_21988(
// CHECK-NOT: iree_linalg_ext.dequantize_affine
// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.+}} : tensor<1x1xi32>)
