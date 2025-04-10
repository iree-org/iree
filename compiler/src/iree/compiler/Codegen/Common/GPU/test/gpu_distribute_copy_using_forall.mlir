// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-codegen-gpu-distribute-copy-using-forall))' %s | FileCheck %s

func.func @static_copy(%src : memref<56x32xf32>, %target : memref<56x32xf32>) {
  memref.copy %src, %target : memref<56x32xf32> to memref<56x32xf32>
  return
}

// CHECK-LABEL: func.func @static_copy
//  CHECK-SAME: (%[[SRC:.+]]: memref<56x32xf32>, %[[TARGET:.+]]: memref<56x32xf32>)

//       CHECK:   scf.forall (%[[IV0:[A-Za-z0-9]+]], %[[IV1:[A-Za-z0-9]+]]) = (0, 0) to (56, 32) step (1, 4) {
//   CHECK-DAG:       %[[SRC_SUBVIEW:.+]] = memref.subview %[[SRC]][%[[IV0]], %[[IV1]]] [1, 4] [1, 1]
//   CHECK-DAG:       %[[TARGET_SUBVIEW:.+]] = memref.subview %[[TARGET]][%[[IV0]], %[[IV1]]] [1, 4] [1, 1]
//       CHECK:       memref.copy %[[SRC_SUBVIEW]], %[[TARGET_SUBVIEW]]
//       CHECK:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

func.func @unaligned_copy(%src : memref<56x31xf32>, %target : memref<56x31xf32>) {
  memref.copy %src, %target : memref<56x31xf32> to memref<56x31xf32>
  return
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
// CHECK-LABEL: func.func @unaligned_copy
//  CHECK-SAME: (%[[SRC:.+]]: memref<56x31xf32>, %[[TARGET:.+]]: memref<56x31xf32>)

//       CHECK:   scf.forall (%[[IV0:[A-Za-z0-9]+]], %[[IV1:[A-Za-z0-9]+]]) = (0, 0) to (56, 31) step (1, 4) {
//       CHECK:       %[[MIN:.+]] = affine.min #[[$MAP]](%c4, %c31, %[[IV1]])
//   CHECK-DAG:       %[[SRC_SUBVIEW:.+]] = memref.subview %[[SRC]][%[[IV0]], %[[IV1]]] [1, %[[MIN]]]
//   CHECK-DAG:       %[[TARGET_SUBVIEW:.+]] = memref.subview %[[TARGET]][%[[IV0]], %[[IV1]]] [1, %[[MIN]]]
//       CHECK:       memref.copy %[[SRC_SUBVIEW]], %[[TARGET_SUBVIEW]]
//       CHECK:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

func.func @dynamic_copy(%src : memref<?x?xf32>, %target : memref<?x?xf32>) {
  memref.copy %src, %target : memref<?x?xf32> to memref<?x?xf32>
  return
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
// CHECK-LABEL: func.func @dynamic_copy
//  CHECK-SAME: (%[[SRC:.+]]: memref<?x?xf32>, %[[TARGET:.+]]: memref<?x?xf32>)

//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[SRC]], %c0 : memref<?x?xf32>
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[SRC]], %c1 : memref<?x?xf32>
//       CHECK:   scf.forall (%[[IV0:[A-Za-z0-9]+]], %[[IV1:[A-Za-z0-9]+]]) = (0, 0) to (%[[D0]], %[[D1]]) step (1, 4) {
//       CHECK:       %[[MIN:.+]] = affine.min #[[$MAP]](%c4, %[[D1]], %[[IV1]])
//   CHECK-DAG:       %[[SRC_SUBVIEW:.+]] = memref.subview %[[SRC]][%[[IV0]], %[[IV1]]] [1, %[[MIN]]]
//   CHECK-DAG:       %[[TARGET_SUBVIEW:.+]] = memref.subview %[[TARGET]][%[[IV0]], %[[IV1]]] [1, %[[MIN]]]
//       CHECK:       memref.copy %[[SRC_SUBVIEW]], %[[TARGET_SUBVIEW]]
//       CHECK:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

func.func @f16_copy(%src : memref<56x32xf16>, %target : memref<56x32xf16>) {
  memref.copy %src, %target : memref<56x32xf16> to memref<56x32xf16>
  return
}

// CHECK-LABEL: func.func @f16_copy
//  CHECK-SAME: (%[[SRC:.+]]: memref<56x32xf16>, %[[TARGET:.+]]: memref<56x32xf16>)

//       CHECK:   scf.forall (%[[IV0:[A-Za-z0-9]+]], %[[IV1:[A-Za-z0-9]+]]) = (0, 0) to (56, 32) step (1, 8) {
//   CHECK-DAG:       %[[SRC_SUBVIEW:.+]] = memref.subview %[[SRC]][%[[IV0]], %[[IV1]]] [1, 8]
//   CHECK-DAG:       %[[TARGET_SUBVIEW:.+]] = memref.subview %[[TARGET]][%[[IV0]], %[[IV1]]] [1, 8]
//       CHECK:       memref.copy %[[SRC_SUBVIEW]], %[[TARGET_SUBVIEW]]
//       CHECK:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

func.func @rank_0_copy(%src : memref<f32>, %target : memref<f32>) {
  memref.copy %src, %target : memref<f32> to memref<f32>
  return
}

// CHECK-LABEL: func.func @rank_0_copy
//  CHECK-SAME: (%[[SRC:.+]]: memref<f32>, %[[TARGET:.+]]: memref<f32>)

//       CHECK:   scf.forall (%{{.*}}) in (1) {
//       CHECK:       memref.copy %[[SRC]], %[[TARGET]]
//       CHECK:   mapping = [#gpu.thread<linear_dim_0>]

// -----

func.func @already_distributed_copy(%src : memref<56x32xf32>, %target : memref<56x32xf32>) {
  scf.forall (%arg2) in (1) {
    memref.copy %src, %target : memref<56x32xf32> to memref<56x32xf32>
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return
}

// CHECK-LABEL: func.func @already_distributed_copy
//  CHECK-SAME: (%[[SRC:.+]]: memref<56x32xf32>, %[[TARGET:.+]]: memref<56x32xf32>)

//       CHECK:   scf.forall (%{{.*}}) in (1) {
//       CHECK:       memref.copy %[[SRC]], %[[TARGET]]
//       CHECK:   mapping = [#gpu.thread<linear_dim_0>]
