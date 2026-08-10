// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-convert-qdq-to-integer-math))" %s | FileCheck %s

// Contractions the rewrite declines. Each keeps its floating point form, which
// stays correct. The witness is the surviving dequantize together with the
// original contraction, or with the body operation that blocked the match.

//===----------------------------------------------------------------------===//
// Structural preconditions
//
// The rewrite needs two dequantized operands feeding a zero-initialised
// contraction, and it has to be able to read the contraction out of the body.
//===----------------------------------------------------------------------===//

// Only one operand is quantized, so there is no integer contraction to form:
// the other operand is real valued.
func.func @single_quantized_operand(%aq: tensor<4x8xi8>, %a_s: f32, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f32)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @single_quantized_operand(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// The integer contraction starts from an integer zero, so the original
// accumulator has to be a zero fill for the epilogue not to have to carry it.
// A bias initialised accumulator is not handled.
func.func @nonzero_init(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>, %b_s: f32, %bias: tensor<4x16xf32>) -> tensor<4x16xf32> {
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
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%bias : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @nonzero_init(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// The dequantize has to be the contraction's immediate producer. Anything in
// between hides it, which is why the pass runs after the passes that move
// reshapes and transposes out from between the two.
func.func @op_between_dequantize_and_contraction(%aq: tensor<4x6xi8>, %a_s: f32,
    %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x6xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x6xi8>, f32)
      outs(%a_i : tensor<4x6xf32>) -> tensor<4x6xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %pad_cst = arith.constant 0.000000e+00 : f32
  %a_pad = tensor.pad %a low[0, 0] high[0, 2] {
  ^bb0(%i: index, %j: index):
    tensor.yield %pad_cst : f32
  } : tensor<4x6xf32> to tensor<4x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a_pad, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @op_between_dequantize_and_contraction(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// The remaining cases are bodies. A body qualifies only if it is exactly three
// operations, a multiply and an add and the yield, and those three multiply the
// two inputs and add the accumulator. The five cases here fail on the count:
// each computes a contraction, or something close to one, with extra operations
// around it. This one extends f16 inputs into an f32 accumulator, which is what
// a half precision model produces.
func.func @element_type_mismatch(%aq: tensor<4x8xi8>, %a_s: f16, %bq: tensor<8x16xi8>, %b_s: f16) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f16)
      outs(%a_i : tensor<4x8xf16>) -> tensor<4x8xf16>
  %b_i = tensor.empty() : tensor<8x16xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f16)
      outs(%b_i : tensor<8x16xf16>) -> tensor<8x16xf16>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d2, d1)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<4x8xf16>, tensor<8x16xf16>)
      outs(%f : tensor<4x16xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %le = arith.extf %in : f16 to f32
    %re = arith.extf %in_0 : f16 to f32
    %m = arith.mulf %le, %re : f32
    %s = arith.addf %out, %m : f32
    linalg.yield %s : f32
  } -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @element_type_mismatch(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   arith.extf

// -----

// A round trip through f16 rounds the lhs before multiplying it.
func.func @narrowed_input(%aq: tensor<1x8xi8>, %sa: f32,
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
      %half = arith.truncf %lhs : f32 to f16
      %rounded = arith.extf %half : f16 to f32
      %product = arith.mulf %rounded, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @narrowed_input(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   arith.truncf

// -----

// A negation on the way into the multiply.
func.func @negated_input(%aq: tensor<1x8xi8>, %sa: f32,
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
      %negative = arith.negf %lhs : f32
      %product = arith.mulf %negative, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
  } -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @negated_input(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   arith.negf

// -----

// A negation on the accumulator the add reads.
func.func @negated_accumulator(%aq: tensor<1x8xi8>, %sa: f32,
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
      %negative = arith.negf %acc : f32
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %negative, %product : f32
      linalg.yield %sum : f32
  } -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @negated_accumulator(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   arith.negf

// -----

// A negation between the add and the yield.
func.func @negated_result(%aq: tensor<1x8xi8>, %sa: f32,
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
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      %negative = arith.negf %sum : f32
      linalg.yield %negative : f32
  } -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @negated_result(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   arith.negf

// -----

// The four cases below are three operations wide and still do not qualify. The
// maps and iterator types alone do not make an op a contraction: this one
// maximises instead of accumulating, so the expansion the rewrite performs
// would be meaningless.
func.func @non_mul_add_body(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
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
  %c = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d2, d1)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %m = arith.mulf %in, %in_0 : f32
    %s = arith.maximumf %out, %m : f32
    linalg.yield %s : f32
  } -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @non_mul_add_body(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   arith.maximumf

// -----

// The accumulated value is a difference rather than a product.
func.func @non_mul_product(%aq: tensor<1x8xi8>, %sa: f32,
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
      %difference = arith.subf %lhs, %rhs : f32
      %sum = arith.addf %acc, %difference : f32
      linalg.yield %sum : f32
  } -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @non_mul_product(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   arith.subf

// -----

// A product of the lhs with itself. The rhs reaches the op but not the
// multiply, so substituting the quantized operands would change what is
// computed.
func.func @multiply_ignores_rhs(%aq: tensor<1x8xi8>, %sa: f32,
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
      %square = arith.mulf %lhs, %lhs : f32
      %sum = arith.addf %acc, %square : f32
      linalg.yield %sum : f32
  } -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @multiply_ignores_rhs(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   ^bb0(%[[L:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f32, %{{.+}}: f32):
//       CHECK:     arith.mulf %[[L]], %[[L]]

// -----

// The add doubles the product instead of accumulating it, so the op is not a
// reduction over the multiply however its iterator types are marked.
func.func @add_drops_accumulator(%aq: tensor<1x8xi8>, %sa: f32,
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
      %product = arith.mulf %lhs, %rhs : f32
      %doubled = arith.addf %product, %product : f32
      linalg.yield %doubled : f32
  } -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @add_drops_accumulator(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   %[[P:.+]] = arith.mulf
//       CHECK:   arith.addf %[[P]], %[[P]]

//===----------------------------------------------------------------------===//
// Reduction legality
//
// A reduction dim can become integer only when both inputs index it, the output
// does not, and every quantization parameter is invariant along it. Its extent
// has to be statically known and nonzero, and at least one such dim has to
// remain.
//===----------------------------------------------------------------------===//

// -----

// Per-channel along K, the only reduction dim, so no sub-reduction has constant
// parameters and there is no integer contraction to form.
func.func @form_none(%aq: tensor<4x8xi8>, %a_s: tensor<8xf32>,
    %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, tensor<8xf32>)
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
// CHECK-LABEL: func.func @form_none(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// An unknown reduction extent cannot be bounded against the i32 accumulator.
func.func @dynamic_reduction(%aq: tensor<4x?xi8>, %a_s: f32, %a_z: i8,
    %bq: tensor<?x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %c1 = arith.constant 1 : index
  %k = tensor.dim %aq, %c1 : tensor<4x?xi8>
  %a_i = tensor.empty(%k) : tensor<4x?xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x?xi8>, f32, i8)
      outs(%a_i : tensor<4x?xf32>) -> tensor<4x?xf32>
  %b_i = tensor.empty(%k) : tensor<?x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<?x16xi8>, f32, i8)
      outs(%b_i : tensor<?x16xf32>) -> tensor<?x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x?xf32>, tensor<?x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @dynamic_reduction(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// The same with the reduction split over two dims: the product of the extents
// is what has to be bounded, so one unknown factor is enough to decline.
func.func @dynamic_multi_dim_reduction(%aq: tensor<4x?x?xi8>, %a_s: f32, %a_z: i8,
    %bq: tensor<?x?x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %k0 = tensor.dim %aq, %c1 : tensor<4x?x?xi8>
  %k1 = tensor.dim %aq, %c2 : tensor<4x?x?xi8>
  %a_i = tensor.empty(%k0, %k1) : tensor<4x?x?xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%aq, %a_s, %a_z : tensor<4x?x?xi8>, f32, i8)
      outs(%a_i : tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  %b_i = tensor.empty(%k0, %k1) : tensor<?x?x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%bq, %b_s, %b_z : tensor<?x?x16xi8>, f32, i8)
      outs(%b_i : tensor<?x?x16xf32>) -> tensor<?x?x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.contract
      indexing_maps = [affine_map<(m, n, k0, k1) -> (m, k0, k1)>,
                       affine_map<(m, n, k0, k1) -> (k0, k1, n)>,
                       affine_map<(m, n, k0, k1) -> (m, n)>]
      ins(%a, %b : tensor<4x?x?xf32>, tensor<?x?x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @dynamic_multi_dim_reduction(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.contract

// -----

// An empty reduction retains its original zero result rather than acquiring one
// from a correction term the contraction never produced.
func.func @empty_reduction(%aq: tensor<1x0xi8>, %sa: f32,
    %bq: tensor<0x1xi8>, %sb: f32) -> tensor<1x1xf32> {
  %ai = tensor.empty() : tensor<1x0xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%aq, %sa : tensor<1x0xi8>, f32)
      outs(%ai : tensor<1x0xf32>) -> tensor<1x0xf32>
  %bi = tensor.empty() : tensor<0x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%bq, %sb : tensor<0x1xi8>, f32)
      outs(%bi : tensor<0x1xf32>) -> tensor<0x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x0xf32>, tensor<0x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @empty_reduction(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// Linalg permits a reduction-marked dimension to index the output, but it is
// not a contraction reduction. Reject it whether its scale is invariant or
// varying: it must neither be summed away nor appended as a duplicate partial
// result axis.

func.func @reduction_dim_in_output(%aq: tensor<2x4xi8>, %sa: f32,
    %varying_scale: tensor<2xf32>, %bq: tensor<2x4xi8>, %sb: f32)
    -> (tensor<2xf32>, tensor<2xf32>) {
  %ai = tensor.empty() : tensor<2x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %sa : tensor<2x4xi8>, f32)
      outs(%ai : tensor<2x4xf32>) -> tensor<2x4xf32>
  %a_varying = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %varying_scale : tensor<2x4xi8>, tensor<2xf32>)
      outs(%ai : tensor<2x4xf32>) -> tensor<2x4xf32>
  %bi = tensor.empty() : tensor<2x4xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %sb : tensor<2x4xi8>, f32)
      outs(%bi : tensor<2x4xf32>) -> tensor<2x4xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2xf32>) -> tensor<2xf32>
  %constant = linalg.generic {
      indexing_maps = [affine_map<(m, k) -> (m, k)>,
                       affine_map<(m, k) -> (m, k)>,
                       affine_map<(m, k) -> (m)>],
      iterator_types = ["reduction", "reduction"]}
      ins(%a, %b : tensor<2x4xf32>, tensor<2x4xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %product, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<2xf32>
  %varying = linalg.generic {
      indexing_maps = [affine_map<(m, k) -> (m, k)>,
                       affine_map<(m, k) -> (m, k)>,
                       affine_map<(m, k) -> (m)>],
      iterator_types = ["reduction", "reduction"]}
      ins(%a_varying, %b : tensor<2x4xf32>, tensor<2x4xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %product, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<2xf32>
  return %constant, %varying : tensor<2xf32>, tensor<2xf32>
}
// CHECK-LABEL: func.func @reduction_dim_in_output(
// CHECK-COUNT-3: iree_linalg_ext.dequantize_affine
// CHECK: linalg.generic {{.*}} ins(%{{.+}}, %{{.+}} : tensor<2x4xf32>, tensor<2x4xf32>)
// CHECK: linalg.generic {{.*}} ins(%{{.+}}, %{{.+}} : tensor<2x4xf32>, tensor<2x4xf32>)

//===----------------------------------------------------------------------===//
// Accumulator bound
//
// Every integer intermediate is i32, bounded by
// N * (|Aq|max + |zA|max) * (|Bq|max + |zB|max). Quantization ranges bound the
// inputs when present; otherwise the quantized storage types do. Zero points
// are points on those storage grids, regardless of their SSA carrier types.
// The admissible counterpart of each pair below is in
// convert_qdq_to_integer_math_algebra.mlir.
//===----------------------------------------------------------------------===//

// -----

// The shape and types of @acc_symmetric_fits_deep with both sides asymmetric.
// That is four terms rather than one, costing two more bits: 7+7+16+2+1 = 33,
// so it no longer fits. This pair is what pins the term counting.
func.func @acc_asymmetric_too_deep(%aq: tensor<4x65536xi8>, %a_s: f32, %a_z: i8, %bq: tensor<65536x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x65536xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x65536xi8>, f32, i8)
      outs(%a_i : tensor<4x65536xf32>) -> tensor<4x65536xf32>
  %b_i = tensor.empty() : tensor<65536x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<65536x16xi8>, f32, i8)
      outs(%b_i : tensor<65536x16xf32>) -> tensor<65536x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x65536xf32>, tensor<65536x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @acc_asymmetric_too_deep(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// One element deeper than @acc_last_safe_depth: all -128 inputs sum to +2^31,
// outside the signed i32 range.
func.func @acc_positive_endpoint(%aq: tensor<1x131072xi8>, %sa: f32,
    %bq: tensor<131072x1xi8>, %sb: f32) -> tensor<1x1xf32> {
  %ai = tensor.empty() : tensor<1x131072xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%aq, %sa : tensor<1x131072xi8>, f32)
      outs(%ai : tensor<1x131072xf32>) -> tensor<1x131072xf32>
  %bi = tensor.empty() : tensor<131072x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%bq, %sb : tensor<131072x1xi8>, f32)
      outs(%bi : tensor<131072x1xf32>) -> tensor<131072x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x131072xf32>, tensor<131072x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @acc_positive_endpoint(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// Both operands i16: 15+15 magnitude bits leaves no room for even a shallow
// reduction.
func.func @acc_wide_storage(%aq: tensor<4x8xi16>, %a_s: f32, %bq: tensor<8x16xi16>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi16>, f32)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi16>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @acc_wide_storage(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// i64 storage is at the accumulator width, so the operand is rejected on width
// before any extent is consulted and the unknown depth never matters.
func.func @acc_dynamic_wide_storage(%aq: tensor<1x?xi64>, %sa: f32,
    %bq: tensor<?x1xi64>, %sb: f32) -> tensor<1x1xf32> {
  %dim = arith.constant 1 : index
  %k = tensor.dim %aq, %dim : tensor<1x?xi64>
  %ai = tensor.empty(%k) : tensor<1x?xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%aq, %sa : tensor<1x?xi64>, f32)
      outs(%ai : tensor<1x?xf32>) -> tensor<1x?xf32>
  %bi = tensor.empty(%k) : tensor<?x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%bq, %sb : tensor<?x1xi64>, f32)
      outs(%bi : tensor<?x1xf32>) -> tensor<?x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x?xf32>, tensor<?x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @acc_dynamic_wide_storage(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul
