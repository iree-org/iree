// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-optimize-comparison-ops, canonicalize, cse))" %s \
// RUN:   | FileCheck %s

//===----------------------------------------------------------------------===//
// Ordered predicates: normal + swapped operands
//
// All tests use offset vec [1..4] with sdiv=8 so that both standard
// (slt/sge) and adjusted (sle/sgt) bucket checks pass, including after
// normalization flips the predicate for swapped operands.
//   Standard check: floor(1/8)=0 == floor(4/8)=0, threshold=5
//   Adjusted check: floor(0/8)=0 == floor(3/8)=0, threshold=4
//===----------------------------------------------------------------------===//

// slt normal → sge(scalar, 5). Swapped slt(bcast, vec) → sgt(vec, bcast)
// which uses adjusted check → slt(scalar, 4).

func.func @simplify_slt(%arg0 : index) -> (vector<4xi1>, vector<4xi1>) {
  %bound = util.assume.int %arg0<udiv = 8> : index
  %c1 = arith.constant 1 : index
  %step = vector.step : vector<4xindex>
  %offset = vector.broadcast %c1 : index to vector<4xindex>
  %vec = arith.addi %step, %offset : vector<4xindex>
  %bcast = vector.broadcast %bound : index to vector<4xindex>
  %normal = arith.cmpi slt, %vec, %bcast : vector<4xindex>
  %swapped = arith.cmpi slt, %bcast, %vec : vector<4xindex>
  return %normal, %swapped : vector<4xi1>, vector<4xi1>
}

// CHECK-LABEL: @simplify_slt
//   CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
//       CHECK: %[[BOUND:.+]] = util.assume.int
//       CHECK: %[[CMP1:.+]] = arith.cmpi sge, %[[BOUND]], %[[C5]] : index
//       CHECK: %[[B1:.+]] = vector.broadcast %[[CMP1]] : i1 to vector<4xi1>
//       CHECK: %[[CMP2:.+]] = arith.cmpi slt, %[[BOUND]], %[[C4]] : index
//       CHECK: %[[B2:.+]] = vector.broadcast %[[CMP2]] : i1 to vector<4xi1>
//       CHECK: return %[[B1]], %[[B2]]

// -----

// sge normal → slt(scalar, 5). Swapped sge(bcast, vec) → sle(vec, bcast)
// which uses adjusted check → sge(scalar, 4).

func.func @simplify_sge(%arg0 : index) -> (vector<4xi1>, vector<4xi1>) {
  %bound = util.assume.int %arg0<udiv = 8> : index
  %c1 = arith.constant 1 : index
  %step = vector.step : vector<4xindex>
  %offset = vector.broadcast %c1 : index to vector<4xindex>
  %vec = arith.addi %step, %offset : vector<4xindex>
  %bcast = vector.broadcast %bound : index to vector<4xindex>
  %normal = arith.cmpi sge, %vec, %bcast : vector<4xindex>
  %swapped = arith.cmpi sge, %bcast, %vec : vector<4xindex>
  return %normal, %swapped : vector<4xi1>, vector<4xi1>
}

// CHECK-LABEL: @simplify_sge
//   CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
//       CHECK: %[[BOUND:.+]] = util.assume.int
//       CHECK: %[[CMP1:.+]] = arith.cmpi slt, %[[BOUND]], %[[C5]] : index
//       CHECK: %[[B1:.+]] = vector.broadcast %[[CMP1]] : i1 to vector<4xi1>
//       CHECK: %[[CMP2:.+]] = arith.cmpi sge, %[[BOUND]], %[[C4]] : index
//       CHECK: %[[B2:.+]] = vector.broadcast %[[CMP2]] : i1 to vector<4xi1>
//       CHECK: return %[[B1]], %[[B2]]

// -----

// sle normal → sge(scalar, 4). Swapped sle(bcast, vec) → sge(vec, bcast)
// which uses standard check → slt(scalar, 5).

func.func @simplify_sle(%arg0 : index) -> (vector<4xi1>, vector<4xi1>) {
  %bound = util.assume.int %arg0<udiv = 8> : index
  %c1 = arith.constant 1 : index
  %step = vector.step : vector<4xindex>
  %offset = vector.broadcast %c1 : index to vector<4xindex>
  %vec = arith.addi %step, %offset : vector<4xindex>
  %bcast = vector.broadcast %bound : index to vector<4xindex>
  %normal = arith.cmpi sle, %vec, %bcast : vector<4xindex>
  %swapped = arith.cmpi sle, %bcast, %vec : vector<4xindex>
  return %normal, %swapped : vector<4xi1>, vector<4xi1>
}

// CHECK-LABEL: @simplify_sle
//   CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
//       CHECK: %[[BOUND:.+]] = util.assume.int
//       CHECK: %[[CMP1:.+]] = arith.cmpi sge, %[[BOUND]], %[[C4]] : index
//       CHECK: %[[B1:.+]] = vector.broadcast %[[CMP1]] : i1 to vector<4xi1>
//       CHECK: %[[CMP2:.+]] = arith.cmpi slt, %[[BOUND]], %[[C5]] : index
//       CHECK: %[[B2:.+]] = vector.broadcast %[[CMP2]] : i1 to vector<4xi1>
//       CHECK: return %[[B1]], %[[B2]]

// -----

// sgt normal → slt(scalar, 4). Swapped sgt(bcast, vec) → slt(vec, bcast)
// which uses standard check → sge(scalar, 5).

func.func @simplify_sgt(%arg0 : index) -> (vector<4xi1>, vector<4xi1>) {
  %bound = util.assume.int %arg0<udiv = 8> : index
  %c1 = arith.constant 1 : index
  %step = vector.step : vector<4xindex>
  %offset = vector.broadcast %c1 : index to vector<4xindex>
  %vec = arith.addi %step, %offset : vector<4xindex>
  %bcast = vector.broadcast %bound : index to vector<4xindex>
  %normal = arith.cmpi sgt, %vec, %bcast : vector<4xindex>
  %swapped = arith.cmpi sgt, %bcast, %vec : vector<4xindex>
  return %normal, %swapped : vector<4xi1>, vector<4xi1>
}

// CHECK-LABEL: @simplify_sgt
//   CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
//       CHECK: %[[BOUND:.+]] = util.assume.int
//       CHECK: %[[CMP1:.+]] = arith.cmpi slt, %[[BOUND]], %[[C4]] : index
//       CHECK: %[[B1:.+]] = vector.broadcast %[[CMP1]] : i1 to vector<4xi1>
//       CHECK: %[[CMP2:.+]] = arith.cmpi sge, %[[BOUND]], %[[C5]] : index
//       CHECK: %[[B2:.+]] = vector.broadcast %[[CMP2]] : i1 to vector<4xi1>
//       CHECK: return %[[B1]], %[[B2]]

// -----

//===----------------------------------------------------------------------===//
// eq/ne predicates
//===----------------------------------------------------------------------===//

// eq/ne with offset vec [1..4], sdiv=8. No multiple of 8 in [1,4], so
// eq → false, ne → true. Swapped operands produce the same result (symmetric).

func.func @fold_eq_ne(%arg0 : index)
    -> (vector<4xi1>, vector<4xi1>, vector<4xi1>, vector<4xi1>) {
  %bound = util.assume.int %arg0<udiv = 8> : index
  %c1 = arith.constant 1 : index
  %step = vector.step : vector<4xindex>
  %offset = vector.broadcast %c1 : index to vector<4xindex>
  %vec = arith.addi %step, %offset : vector<4xindex>
  %bcast = vector.broadcast %bound : index to vector<4xindex>
  %eq_normal = arith.cmpi eq, %vec, %bcast : vector<4xindex>
  %eq_swapped = arith.cmpi eq, %bcast, %vec : vector<4xindex>
  %ne_normal = arith.cmpi ne, %vec, %bcast : vector<4xindex>
  %ne_swapped = arith.cmpi ne, %bcast, %vec : vector<4xindex>
  return %eq_normal, %eq_swapped, %ne_normal, %ne_swapped
      : vector<4xi1>, vector<4xi1>, vector<4xi1>, vector<4xi1>
}

// CHECK-LABEL: @fold_eq_ne
//   CHECK-DAG: %[[FALSE:.+]] = arith.constant dense<false> : vector<4xi1>
//   CHECK-DAG: %[[TRUE:.+]] = arith.constant dense<true> : vector<4xi1>
//       CHECK: return %[[FALSE]], %[[FALSE]], %[[TRUE]], %[[TRUE]]

// -----

//===----------------------------------------------------------------------===//
// Negative ranges (tests floorDiv rounding)
//===----------------------------------------------------------------------===//

// slt with negative vec range [-7..-5], sdiv=4. Tests floorDiv rounding:
// floor(-7/4) = -2 == floor(-5/4) = -2. Threshold = -5 + 1 = -4.

func.func @simplify_slt_negative_range(%arg0 : index) -> vector<3xi1> {
  %bound = util.assume.int %arg0<udiv = 4> : index
  %cn7 = arith.constant -7 : index
  %step = vector.step : vector<3xindex>
  %offset = vector.broadcast %cn7 : index to vector<3xindex>
  %vec = arith.addi %step, %offset : vector<3xindex>
  %bcast = vector.broadcast %bound : index to vector<3xindex>
  %mask = arith.cmpi slt, %vec, %bcast : vector<3xindex>
  return %mask : vector<3xi1>
}

// CHECK-LABEL: @simplify_slt_negative_range
//   CHECK-NOT: arith.cmpi slt
//   CHECK-DAG: %[[CN4:.+]] = arith.constant -4 : index
//       CHECK: arith.cmpi sge, %{{.*}}, %[[CN4]] : index
//       CHECK: vector.broadcast
//       CHECK: return

// -----

//===----------------------------------------------------------------------===//
// Negative tests
//===----------------------------------------------------------------------===//

// Cross-bucket: vec [4..11] spans floor(4/8)=0 and floor(11/8)=1.

func.func @no_simplify_cross_bucket(%arg0 : index) -> vector<8xi1> {
  %bound = util.assume.int %arg0<udiv = 8> : index
  %c4 = arith.constant 4 : index
  %step = vector.step : vector<8xindex>
  %offset = vector.broadcast %c4 : index to vector<8xindex>
  %vec = arith.addi %step, %offset : vector<8xindex>
  %bcast = vector.broadcast %bound : index to vector<8xindex>
  %mask = arith.cmpi slt, %vec, %bcast : vector<8xindex>
  return %mask : vector<8xi1>
}

// CHECK-LABEL: @no_simplify_cross_bucket
//       CHECK: vector.broadcast %{{.*}} : index to vector<8xindex>
//       CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : vector<8xindex>
//       CHECK: return

// -----

// sle at boundary: vec [0..7] with sdiv=8. The adjusted bucket check
// floor(-1/8)=-1 != floor(6/8)=0. Fails because scalar=0 would give
// 0<=0=T but 1<=0=F.

func.func @no_simplify_sle_at_boundary(%arg0 : index) -> vector<8xi1> {
  %bound = util.assume.int %arg0<udiv = 8> : index
  %step = vector.step : vector<8xindex>
  %bcast = vector.broadcast %bound : index to vector<8xindex>
  %mask = arith.cmpi sle, %step, %bcast : vector<8xindex>
  return %mask : vector<8xi1>
}

// CHECK-LABEL: @no_simplify_sle_at_boundary
//       CHECK: vector.broadcast %{{.*}} : index to vector<8xindex>
//       CHECK: arith.cmpi sle, %{{.*}}, %{{.*}} : vector<8xindex>
//       CHECK: return

// -----

// sgt at boundary: same adjusted check as sle — vec [0..7] fails.

func.func @no_simplify_sgt_at_boundary(%arg0 : index) -> vector<8xi1> {
  %bound = util.assume.int %arg0<udiv = 8> : index
  %step = vector.step : vector<8xindex>
  %bcast = vector.broadcast %bound : index to vector<8xindex>
  %mask = arith.cmpi sgt, %step, %bcast : vector<8xindex>
  return %mask : vector<8xi1>
}

// CHECK-LABEL: @no_simplify_sgt_at_boundary
//       CHECK: vector.broadcast %{{.*}} : index to vector<8xindex>
//       CHECK: arith.cmpi sgt, %{{.*}}, %{{.*}} : vector<8xindex>
//       CHECK: return

// -----

// Insufficient divisibility: udiv=4 with vec width 8 spans multiple buckets.

func.func @no_simplify_insufficient_divisibility(%arg0 : index) -> vector<8xi1> {
  %bound = util.assume.int %arg0<udiv = 4> : index
  %step = vector.step : vector<8xindex>
  %bcast = vector.broadcast %bound : index to vector<8xindex>
  %mask = arith.cmpi slt, %step, %bcast : vector<8xindex>
  return %mask : vector<8xi1>
}

// CHECK-LABEL: @no_simplify_insufficient_divisibility
//       CHECK: vector.broadcast %{{.*}} : index to vector<8xindex>
//       CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : vector<8xindex>
//       CHECK: return

// -----

// No broadcast operand — neither operand is a vector.broadcast.

func.func @no_simplify_no_broadcast(
    %v1 : vector<8xindex>, %v2 : vector<8xindex>) -> vector<8xi1> {
  %mask = arith.cmpi slt, %v1, %v2 : vector<8xindex>
  return %mask : vector<8xi1>
}

// CHECK-LABEL: @no_simplify_no_broadcast
//       CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : vector<8xindex>
//       CHECK: return

// -----

// Unsigned predicates: divisibility rewrite is not applied.

func.func @no_simplify_unsigned(%arg0 : index) -> (vector<8xi1>, vector<8xi1>) {
  %bound = util.assume.int %arg0<udiv = 8> : index
  %step = vector.step : vector<8xindex>
  %bcast = vector.broadcast %bound : index to vector<8xindex>
  %ult = arith.cmpi ult, %step, %bcast : vector<8xindex>
  %uge = arith.cmpi uge, %step, %bcast : vector<8xindex>
  return %ult, %uge : vector<8xi1>, vector<8xi1>
}

// CHECK-LABEL: @no_simplify_unsigned
//       CHECK: arith.cmpi ult, %{{.*}}, %{{.*}} : vector<8xindex>
//       CHECK: arith.cmpi uge, %{{.*}}, %{{.*}} : vector<8xindex>
//       CHECK: return

// -----

// eq/ne with vec [0..7] and sdiv=8: 0 is a multiple of 8 so the range
// contains a multiple. Cannot fold.

func.func @no_fold_eq_ne_has_multiple(%arg0 : index) -> (vector<8xi1>, vector<8xi1>) {
  %bound = util.assume.int %arg0<udiv = 8> : index
  %step = vector.step : vector<8xindex>
  %bcast = vector.broadcast %bound : index to vector<8xindex>
  %eq = arith.cmpi eq, %step, %bcast : vector<8xindex>
  %ne = arith.cmpi ne, %step, %bcast : vector<8xindex>
  return %eq, %ne : vector<8xi1>, vector<8xi1>
}

// CHECK-LABEL: @no_fold_eq_ne_has_multiple
//       CHECK: vector.broadcast %{{.*}} : index to vector<8xindex>
//       CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : vector<8xindex>
//       CHECK: arith.cmpi ne, %{{.*}}, %{{.*}} : vector<8xindex>
//       CHECK: return
