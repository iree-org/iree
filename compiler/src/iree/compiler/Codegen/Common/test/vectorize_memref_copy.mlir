// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-vectorize-memref-copy))" %s | FileCheck %s

func.func @memref_copy(%source: memref<2x2xf32>, %dest: memref<2x2xf32>) {
  memref.copy %source, %dest : memref<2x2xf32> to memref<2x2xf32>
  return
}

// CHECK-LABEL: func.func @memref_copy
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<2x2xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<2x2xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   scf.for %[[ARG2:.+]] = %[[C0]] to %[[C2]] step %[[C2]]
//       CHECK:     scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C2]]
//       CHECK:       %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[ARG2]], %[[ARG3]]] [2, 2] [1, 1]
//       CHECK:       %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[ARG2]], %[[ARG3]]] [2, 2] [1, 1]
//       CHECK:       %[[RD:.+]] = vector.transfer_read %[[SOURCE_SUBVIEW]]
//       CHECK:       vector.transfer_write %[[RD]], %[[DEST_SUBVIEW]]

// -----

func.func @linalg_copy(%source: memref<2x2xf32>, %dest: memref<2x2xf32>) {
  linalg.copy ins(%source : memref<2x2xf32>) outs(%dest : memref<2x2xf32>)
  return
}

// CHECK-LABEL: func.func @linalg_copy
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<2x2xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<2x2xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   scf.for %[[ARG2:.+]] = %[[C0]] to %[[C2]] step %[[C2]]
//       CHECK:     scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C2]]
//       CHECK:       %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[ARG2]], %[[ARG3]]] [2, 2] [1, 1]
//       CHECK:       %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[ARG2]], %[[ARG3]]] [2, 2] [1, 1]
//       CHECK:       %[[RD:.+]] = vector.transfer_read %[[SOURCE_SUBVIEW]]
//       CHECK:       vector.transfer_write %[[RD]], %[[DEST_SUBVIEW]]

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
//       CHECK:     scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C2]]
//       CHECK:       %[[MIN:.+]] = affine.min affine_map<(d0) -> (-d0 + 3, 2)>(%[[ARG2]])
//       CHECK:       %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[ARG2]], %[[ARG3]]] [%[[MIN]], 2] [1, 1]
//       CHECK:       %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[ARG2]], %[[ARG3]]] [%[[MIN]], 2] [1, 1]
//       CHECK:       memref.copy %[[SOURCE_SUBVIEW]], %[[DEST_SUBVIEW]]

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
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[DIM:.+]] = memref.dim %[[SOURCE]], %[[C0]] : memref<?x4xf32>
//       CHECK:   scf.for %[[ARG2:.+]] = %[[C0]] to %[[DIM]] step %[[C1]]
//       CHECK:     scf.for %[[ARG3:.+]] = %[[C0]] to %[[C4]] step %[[C4]]
//       CHECK:       %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[ARG2]], %[[ARG3]]] [1, 4] [1, 1]
//       CHECK:       %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[ARG2]], %[[ARG3]]] [1, 4] [1, 1]
//       CHECK:       %[[RD:.+]] = vector.transfer_read %[[SOURCE_SUBVIEW]]
//       CHECK:       vector.transfer_write %[[RD]], %[[DEST_SUBVIEW]]

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
