// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-propagate-constant-offsets))" %s \
// RUN:   --split-input-file --mlir-print-local-scope | FileCheck %s

func.func @extract_constant_map_offset(%id: index) -> index {
  %0 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%id]
  return %0 : index
}

// CHECK-LABEL: func @extract_constant_map_offset
// CHECK-SAME:    %[[ID:.+]]: index
// CHECK:         %[[C32:.+]] = arith.constant 32 : index
// CHECK:         %[[ADD:.+]] = arith.addi %[[ID]], %[[C32]] overflow<nsw> : index
// CHECK:         return %[[ADD]]

// -----

func.func @extract_offset_unsimplified_map(%id: index) -> index {
  %0 = affine.apply affine_map<()[s0] -> (32 + s0)>()[%id]
  return %0 : index
}

// CHECK-LABEL: func @extract_offset_unsimplified_map
// CHECK-SAME:    %[[ID:.+]]: index
// CHECK:         %[[C32:.+]] = arith.constant 32 : index
// CHECK:         %[[ADD:.+]] = arith.addi %[[ID]], %[[C32]] overflow<nsw> : index
// CHECK:         return %[[ADD]]

// -----

func.func @fold_add_into_map(%id0: index, %id1: index) -> index {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %a0 = arith.addi %id0, %c1 overflow<nsw> : index
  %a1 = arith.addi %id1, %c2 overflow<nsw, nuw> : index
  %0 = affine.apply affine_map<(d0)[s0] -> (s0 + d0)>(%a0)[%a1]
  return %0 : index
}

// CHECK-LABEL: func @fold_add_into_map
// CHECK-SAME:    %[[ID0:[A-Za-z0-9]+]]: index
// CHECK-SAME:    %[[ID1:[A-Za-z0-9]+]]: index
// CHECK:         %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[AFFINE_ADD:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%[[ID1]], %[[ID0]]]
// CHECK:         %[[ADD:.+]] = arith.addi %[[AFFINE_ADD]], %[[C3]] overflow<nsw> : index
// CHECK:         return %[[ADD]]

// -----

func.func @propagate_constant_offsets_through_linearize(%id0: index, %id1: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c13 = arith.constant 13 : index
  %a0 = arith.addi %id0, %c1 overflow<nsw> : index
  %a1 = arith.addi %id1, %c2 overflow<nsw, nuw> : index
  %0 = affine.linearize_index [%a0, %c0, %a1, %c13] by (3, 5, 7, 11) : index
  return %0 : index
}

// CHECK-LABEL: func @propagate_constant_offsets_through_linearize
// CHECK-SAME:    %[[ID0:[A-Za-z0-9]+]]: index
// CHECK-SAME:    %[[ID1:[A-Za-z0-9]+]]: index

// Total constant offset = 1 * 5 * 7 * 11 + 2 * 11 + 13 = 420
// CHECK:         %[[C420:.+]] = arith.constant 420 : index
// CHECK:         %[[LINEARIZE:.+]] = affine.linearize_index [%[[ID0]], %c0, %[[ID1]], %c0] by (3, 5, 7, 11) : index
// CHECK:         %[[ADD:.+]] = arith.addi %[[LINEARIZE]], %[[C420]] overflow<nsw> : index
// CHECK:         return %[[ADD]]

// -----

func.func @propagate_constant_offsets_unbounded(%id: index) -> index {
  %c2 = arith.constant 2 : index
  %c13 = arith.constant 13 : index
  %a = arith.addi %id, %c2 overflow<nsw> : index
  %0 = affine.linearize_index [%a, %c13] by (11) : index
  return %0 : index
}

// CHECK-LABEL: func @propagate_constant_offsets_unbounded
// CHECK-SAME:    %[[ID:[A-Za-z0-9]+]]: index

// Total constant offset = 2 * 11 + 13 = 35
// CHECK:         %[[C35:.+]] = arith.constant 35 : index
// CHECK:         %[[LINEARIZE:.+]] = affine.linearize_index [%[[ID]], %c0] by (11) : index
// CHECK:         %[[ADD:.+]] = arith.addi %[[LINEARIZE]], %[[C35]] overflow<nsw> : index
// CHECK:         return %[[ADD]]

// -----

func.func @propagate_constant_offsets_stop_dynamic(%id0: index, %id1: index, %s1: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c13 = arith.constant 13 : index
  %a0 = arith.addi %id0, %c1 overflow<nsw> : index
  %a1 = arith.addi %id1, %c2 overflow<nsw, nuw> : index
  %0 = affine.linearize_index disjoint [%a0, %c0, %a1, %c13] by (3, %s1, 7, 11) : index
  return %0 : index
}

// CHECK-LABEL: func @propagate_constant_offsets_stop_dynamic
// CHECK-SAME:    %[[ID0:[A-Za-z0-9]+]]: index
// CHECK-SAME:    %[[ID1:[A-Za-z0-9]+]]: index
// CHECK-SAME:    %[[S1:[A-Za-z0-9]+]]: index

// Total constant offset before dynamic = 2 * 11 + 13 = 35
// CHECK:         %[[C35:.+]] = arith.constant 35 : index
// CHECK:         %[[ADD0:.+]] = arith.addi %[[ID0]], %c1 overflow<nsw> : index
// CHECK:         %[[LINEARIZE:.+]] = affine.linearize_index disjoint [%[[ADD0]], %c0, %[[ID1]], %c0] by (3, %[[S1]], 7, 11) : index
// CHECK:         %[[ADD:.+]] = arith.addi %[[LINEARIZE]], %[[C35]] overflow<nsw> : index
// CHECK:         return %[[ADD]]

// -----

func.func @fold_mul_into_linearize(%id0: index, %id1: index) -> index {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %a0 = arith.muli %id0, %c2 overflow<nsw> : index
  %a1 = arith.muli %id1, %c4 overflow<nsw, nuw> : index
  %0 = affine.linearize_index disjoint [%a0, %a1] by (16) : index
  return %0 : index
}

// CHECK-LABEL: func @fold_mul_into_linearize
// CHECK-SAME:    %[[ID0:[A-Za-z0-9]+]]: index
// CHECK-SAME:    %[[ID1:[A-Za-z0-9]+]]: index
// CHECK:         %[[LINEARIZE:.+]] = affine.linearize_index disjoint [%[[ID0]], %c0, %[[ID1]], %c0] by (2, 4, 4) : index
// CHECK:         return %[[LINEARIZE]]

// -----

func.func @nofold_mul_indivisible(%id0: index, %id1: index) -> index {
  %c7 = arith.constant 7 : index
  %a = arith.muli %id1, %c7 overflow<nsw> : index
  %0 = affine.linearize_index [%id0, %a] by (16) : index
  return %0 : index
}

// CHECK-LABEL: func @nofold_mul_indivisible
// CHECK-SAME:    %[[ID0:[A-Za-z0-9]+]]: index
// CHECK-SAME:    %[[ID1:[A-Za-z0-9]+]]: index
// CHECK:         %[[MUL:.+]] = arith.muli %[[ID1]], %c7 overflow<nsw> : index
// CHECK:         %[[LINEARIZE:.+]] = affine.linearize_index [%[[ID0]], %[[MUL]]] by (16) : index
// CHECK:         return %[[LINEARIZE]]
