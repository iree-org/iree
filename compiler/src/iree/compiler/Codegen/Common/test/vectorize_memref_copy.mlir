// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-vectorize-memref-copy))" %s | FileCheck %s

func.func @memref_copy(%source: memref<2x2xf32>, %dest: memref<2x2xf32>) {
  memref.copy %source, %dest : memref<2x2xf32> to memref<2x2xf32>
  return
}

// CHECK-LABEL: func.func @memref_copy
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<2x2xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<2x2xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[POISON:.+]] = ub.poison : f32
//       CHECK:   %[[RD:.+]] = vector.transfer_read %[[SOURCE]][%[[C0]], %[[C0]]], %[[POISON]] {in_bounds = [true, true]} : memref<2x2xf32>, vector<2x2xf32>
//       CHECK:   vector.transfer_write %[[RD]], %[[DEST]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<2x2xf32>, memref<2x2xf32>

// -----

func.func @linalg_copy(%source: memref<2x2xf32>, %dest: memref<2x2xf32>) {
  linalg.copy ins(%source : memref<2x2xf32>) outs(%dest : memref<2x2xf32>)
  return
}

// CHECK-LABEL: func.func @linalg_copy
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<2x2xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<2x2xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[POISON:.+]] = ub.poison : f32
//       CHECK:   %[[RD:.+]] = vector.transfer_read %[[SOURCE]][%[[C0]], %[[C0]]], %[[POISON]] {in_bounds = [true, true]} : memref<2x2xf32>, vector<2x2xf32>
//       CHECK:   vector.transfer_write %[[RD]], %[[DEST]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<2x2xf32>, memref<2x2xf32>

// -----

// Test with the last dimension larger than and not a multiple of the preferred number of copy elements.

func.func @memref_copy_not_multiple_of_preferred(%source: memref<2x6xf32>, %dest: memref<2x6xf32>) {
  memref.copy %source, %dest : memref<2x6xf32> to memref<2x6xf32>
  return
}

// CHECK-LABEL: func.func @memref_copy_not_multiple_of_preferred
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<2x6xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<2x6xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
//       CHECK:   scf.for %[[ARG2:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
//       CHECK:     scf.for %[[ARG3:.+]] = %[[C0]] to %[[C6]] step %[[C4]]
//       CHECK:       %[[MIN:.+]] = affine.min affine_map<(d0) -> (-d0 + 6, 4)>(%[[ARG3]])
//       CHECK:       %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[ARG2]], %[[ARG3]]] [1, %[[MIN]]] [1, 1]
//       CHECK:       %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[ARG2]], %[[ARG3]]] [1, %[[MIN]]] [1, 1]
//       CHECK:       memref.copy %[[SOURCE_SUBVIEW]], %[[DEST_SUBVIEW]]

// -----

// Test with the penultimate dimension larger than and not a multiple of the preferred number of copy elements on that dimension.

func.func @memref_copy_not_multiple_on_penultimate_dim(%source: memref<3x2xf32>, %dest: memref<3x2xf32>) {
  memref.copy %source, %dest : memref<3x2xf32> to memref<3x2xf32>
  return
}
// CHECK-LABEL: func.func @memref_copy_not_multiple_on_penultimate_dim
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<3x2xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<3x2xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//       CHECK:   scf.for %[[ARG2:.+]] = %[[C0]] to %[[C3]] step %[[C2]]
//       CHECK:     %[[MIN:.+]] = affine.min affine_map<(d0) -> (-d0 + 3, 2)>(%[[ARG2]])
//       CHECK:     %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[ARG2]], 0] [%[[MIN]], 2] [1, 1]
//       CHECK:     %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[ARG2]], 0] [%[[MIN]], 2] [1, 1]
//       CHECK:     memref.copy %[[SOURCE_SUBVIEW]], %[[DEST_SUBVIEW]]

// -----

func.func @memref_copy_dynamic(%source: memref<?x4xf32>, %dest: memref<?x4xf32>) {
  memref.copy %source, %dest : memref<?x4xf32> to memref<?x4xf32>
  return
}
// CHECK-LABEL: func.func @memref_copy_dynamic
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<?x4xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<?x4xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[DIM:.+]] = memref.dim %[[SOURCE]], %[[C0]] : memref<?x4xf32>
//       CHECK:   scf.for %[[ARG2:.+]] = %[[C0]] to %[[DIM]] step %[[C1]]
//       CHECK:     %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[ARG2]], 0] [1, 4] [1, 1]
//       CHECK:     %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[ARG2]], 0] [1, 4] [1, 1]
//       CHECK:     %[[RD:.+]] = vector.transfer_read %[[SOURCE_SUBVIEW]]
//       CHECK:     vector.transfer_write %[[RD]], %[[DEST_SUBVIEW]]

// -----

func.func @memref_copy_dynamic_inner_dim(%source: memref<4x?xf32>, %dest: memref<4x?xf32>) {
  memref.copy %source, %dest : memref<4x?xf32> to memref<4x?xf32>
  return
}
// CHECK-LABEL: func.func @memref_copy_dynamic_inner_dim
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<4x?xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<4x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[DIM:.+]] = memref.dim %[[SOURCE]], %[[C1]] : memref<4x?xf32>
//       CHECK:   scf.for %[[ARG2:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
//       CHECK:     scf.for %[[ARG3:.+]] = %[[C0]] to %[[DIM]] step %[[C4]]
//       CHECK:       %[[MIN:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%[[ARG3]])[%[[DIM]]]
//       CHECK:       %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[ARG2]], %[[ARG3]]] [1, %[[MIN]]] [1, 1]
//       CHECK:       %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[ARG2]], %[[ARG3]]] [1, %[[MIN]]] [1, 1]
//       CHECK:       memref.copy %[[SOURCE_SUBVIEW]], %[[DEST_SUBVIEW]]

// -----

// Test that the single iteration loops are removed and the subview ops are canonicalized
// (`memref<1x?xbf16, strided<[4, 1]>` instead of `memref<1x?xbf16, strided<[4, 1], offset: ?>`).

func.func @memref_copy_fully_dynamic(%source: memref<1x4xbf16>, %dest: memref<32x?xbf16, strided<[40, 1], offset: ?>>, %dim: index) {
  %c0 = arith.constant 0 : index
  scf.forall (%arg0) in (3) {
    %0 = affine.min affine_map<(d0) -> (d0 * -16 + 40, 16)>(%arg0)
    %1:2 = affine.delinearize_index %dim into (2, 64) : index, index
    %2:3 = affine.delinearize_index %1#1 into (4, 16) : index, index, index
    %3 = affine.linearize_index disjoint [%2#1, %c0] by (4, 4) : index
    %4 = affine.linearize_index disjoint [%1#0, %2#2] by (2, 16) : index
    %5 = affine.max affine_map<()[s0] -> (-s0 + 32, 0)>()[%4]
    %6 = affine.min affine_map<()[s0] -> (1, s0)>()[%5]
    %7 = affine.max affine_map<(d0)[s0] -> (0, d0 - s0)>(%0)[%3]
    %8 = affine.min affine_map<(d0) -> (4, d0)>(%7)
    %subview_0 = memref.subview %source[0, 0] [%6, %8] [1, 1] : memref<1x4xbf16> to memref<?x?xbf16, strided<[4, 1]>>
    %subview_1 = memref.subview %dest[%4, %3] [%6, %8] [1, 1] : memref<32x?xbf16, strided<[40, 1], offset: ?>> to memref<?x?xbf16, strided<[40, 1], offset: ?>>
    memref.copy %subview_0, %subview_1 : memref<?x?xbf16, strided<[4, 1]>> to memref<?x?xbf16, strided<[40, 1], offset: ?>>
  }
  return
}
// CHECK-LABEL: func.func @memref_copy_fully_dynamic
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<1x4xbf16>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<32x?xbf16, strided<[40, 1], offset: ?>>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[LIN_0:.+]] = affine.linearize_index disjoint [%{{.+}}, %{{.+}}] by (4, 4) : index
//   CHECK-DAG:   %[[LIN_1:.+]] = affine.linearize_index disjoint [%{{.+}}, %{{.+}}] by (2, 16) : index
//   CHECK-DAG:   %[[MIN_0:.+]] = affine.min affine_map<()[s0] -> (1, s0)>()[%{{.+}}]
//   CHECK-DAG:   %[[MIN_1:.+]] = affine.min affine_map<(d0) -> (4, d0)>(%{{.+}})
//   CHECK-DAG:   %[[SUBVIEW_0:.+]] = memref.subview %[[SOURCE]][0, 0] [%[[MIN_0]], %[[MIN_1]]] [1, 1]
//  CHECK-SAME:   memref<1x4xbf16> to memref<?x?xbf16, strided<[4, 1]>>
//   CHECK-DAG:   %[[SUBVIEW_1:.+]] = memref.subview %[[DEST]][%[[LIN_1]], %[[LIN_0]]] [%[[MIN_0]], %[[MIN_1]]] [1, 1]
//  CHECK-SAME:   memref<32x?xbf16, strided<[40, 1], offset: ?>> to memref<?x?xbf16, strided<[40, 1], offset: ?>>
//   CHECK-DAG:   %[[CMP_0:.+]] = arith.cmpi sgt, %[[MIN_0]], %[[C0]] : index
//       CHECK:   scf.if %[[CMP_0]] {
//       CHECK:     %[[CMP_1:.+]] = arith.cmpi sgt, %[[MIN_1]], %[[C0]] : index
//       CHECK:     scf.if %[[CMP_1]] {
//       CHECK:       %[[MIN_2:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%[[C0]])[%[[MIN_1]]]
//       CHECK:       %[[SUBVIEW_2:.+]] = memref.subview %[[SUBVIEW_0]][0, 0] [1, %[[MIN_2]]] [1, 1]
//  CHECK-SAME:       memref<?x?xbf16, strided<[4, 1]>> to memref<1x?xbf16, strided<[4, 1]>>
//       CHECK:       %[[SUBVIEW_3:.+]] = memref.subview %[[SUBVIEW_1]][0, 0] [1, %[[MIN_2]]] [1, 1]
//  CHECK-SAME:       memref<?x?xbf16, strided<[40, 1], offset: ?>> to memref<1x?xbf16, strided<[40, 1], offset: ?>>
//       CHECK:       memref.copy %[[SUBVIEW_2]], %[[SUBVIEW_3]]

// -----

func.func @memref_copy_dynamic_outer_dim(%source: memref<?x1xf32>, %dest: memref<?x1xf32>) {
  memref.copy %source, %dest : memref<?x1xf32> to memref<?x1xf32>
  return
}
// CHECK-LABEL: func.func @memref_copy_dynamic_outer_dim
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<?x1xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<?x1xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[DIM:.+]] = memref.dim %[[SOURCE]], %[[C0]]
//       CHECK:   scf.for %[[ARG:.+]] = %[[C0]] to %[[DIM]] step %[[C4]]
//       CHECK:     %[[MIN:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%[[ARG]])[%[[DIM]]]
//       CHECK:     %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[ARG]], 0] [%[[MIN]], 1] [1, 1]
//       CHECK:     %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[ARG]], 0] [%[[MIN]], 1] [1, 1]
//       CHECK:     memref.copy %[[SOURCE_SUBVIEW]], %[[DEST_SUBVIEW]]

// -----

// Test that scf.for operations with `_is_tiled` attribute are simplified. The `memref.copy` should still be vectorized as well.

func.func @for_with_tiled_attr(%source: memref<4x?xf32>, %dest: memref<4x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg0 = %c0 to %c1 step %c1 {
    %subview_0 = memref.subview %source[%arg0, 0] [4, 1] [1, 1] : memref<4x?xf32> to memref<4x1xf32, strided<[?, 1], offset: ?>>
    %subview_1 = memref.subview %dest[%arg0, 0] [4, 1] [1, 1] : memref<4x?xf32> to memref<4x1xf32, strided<[?, 1], offset: ?>>
    memref.copy %subview_0, %subview_1 : memref<4x1xf32, strided<[?, 1], offset: ?>> to memref<4x1xf32, strided<[?, 1], offset: ?>>
  } {_is_tiled}
  return
}
// CHECK-LABEL: func.func @for_with_tiled_attr
//   CHECK-NOT:   scf.for
//       CHECK:   vector.transfer_read
//       CHECK:   vector.transfer_write
